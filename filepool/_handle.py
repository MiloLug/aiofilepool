"""Logical file handle implementation for the async file pool."""

from __future__ import annotations

import asyncio
import os
import warnings
from io import IOBase
from typing import Awaitable, Callable, TypeVar

from ._errors import HandleClosedError, PoolStateError
from ._modes import ModeSpec, validate_read_args, validate_write_args

T = TypeVar("T")


class FileHandle:
    """Virtual file handle that proxies I/O via the descriptor manager."""

    def __init__(
        self,
        *,
        pool,
        handle_id: int,
        path: str,
        mode_spec: ModeSpec,
        encoding: str | None,
        errors: str | None,
        newline: str | None,
        initial_position: int,
    ) -> None:
        self._pool = pool
        self._handle_id = handle_id
        self._path = path
        self._mode_spec = mode_spec
        self._encoding = encoding
        self._errors = errors
        self._newline = newline

        self._position = initial_position
        self._closed = False
        self._op_lock = asyncio.Lock()

    async def read(self, size: int = -1, *, offset: int | None = None) -> bytes | str:
        validate_read_args(self._mode_spec, offset=offset)
        async with self._op_lock:
            self._ensure_open()
            target = self._position if offset is None else offset

            async def operation(fd: IOBase) -> tuple[bytes | str, bool | None]:
                if self._pool.uses_threads:
                    data, new_pos = await self._pool._run_blocking(
                        self._read_sync,
                        fd,
                        target,
                        size,
                    )
                else:
                    data, new_pos = await self._read_cooperative(fd, target, size)

                if offset is None:
                    self._position = new_pos
                return data, None

            return await self._run_descriptor_op(operation, dirty_on_error=False)

    async def write(self, data: bytes | str, *, offset: int | None = None) -> int:
        validate_write_args(self._mode_spec, offset=offset, data=data)
        async with self._op_lock:
            self._ensure_open()

            target = self._position if offset is None else offset
            payload = (
                bytes(data)
                if self._mode_spec.binary and isinstance(data, memoryview)
                else data
            )

            async def operation(fd: IOBase) -> tuple[int, bool | None]:
                if self._pool.uses_threads:
                    written, new_pos = await self._pool._run_blocking(
                        self._write_sync,
                        fd,
                        target,
                        payload,
                        self._pool.fsync_on_write,
                    )
                else:
                    written, new_pos = await self._write_cooperative(
                        fd, target, payload
                    )

                if offset is None:
                    self._position = new_pos
                return written, False

            return await self._run_descriptor_op(operation, dirty_on_error=True)

    async def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        async with self._op_lock:
            self._ensure_open()

            async def operation(fd: IOBase) -> tuple[int, bool | None]:
                if self._pool.uses_threads:
                    new_pos = await self._pool._run_blocking(
                        self._seek_sync,
                        fd,
                        self._position,
                        offset,
                        whence,
                    )
                else:
                    new_pos = self._seek_sync(fd, self._position, offset, whence)
                self._position = new_pos
                return new_pos, None

            return await self._run_descriptor_op(operation, dirty_on_error=False)

    async def tell(self) -> int:
        async with self._op_lock:
            self._ensure_open()
            return self._position

    async def flush(self) -> None:
        async with self._op_lock:
            self._ensure_open()

            async def operation(fd: IOBase) -> tuple[None, bool | None]:
                if self._pool.uses_threads:
                    await self._pool._run_blocking(self._flush_sync, fd)
                else:
                    self._flush_sync(fd)
                return None, False

            await self._run_descriptor_op(operation, dirty_on_error=True)

    async def truncate(self, size: int | None = None) -> int:
        async with self._op_lock:
            self._ensure_open()
            target_size = self._position if size is None else size
            if target_size < 0:
                raise ValueError("truncate size must be >= 0")

            async def operation(fd: IOBase) -> tuple[int, bool | None]:
                if self._pool.uses_threads:
                    new_size, new_pos = await self._pool._run_blocking(
                        self._truncate_sync,
                        fd,
                        self._position,
                        target_size,
                        self._pool.fsync_on_write,
                    )
                else:
                    new_size, new_pos = self._truncate_sync(
                        fd,
                        self._position,
                        target_size,
                        self._pool.fsync_on_write,
                    )
                self._position = new_pos
                return new_size, False

            return await self._run_descriptor_op(operation, dirty_on_error=True)

    async def close(self) -> None:
        async with self._op_lock:
            if self._closed:
                return
            self._closed = True
            await self._pool.manager.unregister(self._handle_id)

    async def __aenter__(self) -> "FileHandle":
        self._ensure_open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def name(self) -> str:
        return self._path

    @property
    def mode(self) -> str:
        return self._mode_spec.raw

    async def _run_descriptor_op(
        self,
        operation: Callable[[IOBase], Awaitable[tuple[T, bool | None]]],
        *,
        dirty_on_error: bool,
    ) -> T:
        await self._pool._begin_operation()
        fd: IOBase | None = None
        dirty: bool | None = None
        try:
            fd = await self._pool.manager.acquire(self._handle_id)
            result, dirty = await operation(fd)
            return result
        except Exception:
            if dirty_on_error and dirty is None:
                dirty = True
            raise
        finally:
            if fd is not None:
                await self._pool.manager.release(
                    self._handle_id,
                    dirty=dirty,
                    position=self._position,
                )
            await self._pool._end_operation()

    @staticmethod
    def _read_sync(fd: IOBase, target: int, size: int) -> tuple[bytes | str, int]:
        fd.seek(target)
        data = fd.read(size)
        new_pos = fd.tell()
        return data, int(new_pos)

    async def _read_cooperative(
        self, fd: IOBase, target: int, size: int
    ) -> tuple[bytes | str, int]:
        fd.seek(target)
        chunk_size = self._pool.chunk_size

        if size == 0:
            empty = b"" if self._mode_spec.binary else ""
            return empty, int(fd.tell())

        chunks: list[bytes | str] = []
        if size < 0:
            while True:
                chunk = fd.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
                await asyncio.sleep(0)
        else:
            remaining = size
            while remaining > 0:
                chunk = fd.read(min(chunk_size, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
                if remaining > 0:
                    await asyncio.sleep(0)

        if self._mode_spec.binary:
            data: bytes | str = b"".join(chunks) if chunks else b""
        else:
            data = "".join(chunks) if chunks else ""
        return data, int(fd.tell())

    @staticmethod
    def _write_sync(
        fd: IOBase,
        target: int,
        data: bytes | str | bytearray,
        fsync_on_write: bool,
    ) -> tuple[int, int]:
        fd.seek(target)
        written = fd.write(data)
        if written is None:
            written = len(data)
        fd.flush()
        if fsync_on_write:
            os.fsync(fd.fileno())
        return int(written), int(fd.tell())

    async def _write_cooperative(
        self,
        fd: IOBase,
        target: int,
        data: bytes | str | bytearray,
    ) -> tuple[int, int]:
        fd.seek(target)
        chunk_size = self._pool.chunk_size
        total = 0

        if self._mode_spec.binary:
            payload = bytes(data)
            view = memoryview(payload)
            index = 0
            while index < len(view):
                piece = view[index : index + chunk_size]
                wrote = fd.write(piece)
                if wrote is None:
                    wrote = len(piece)
                if wrote <= 0:
                    raise PoolStateError("write returned non-positive progress")
                index += wrote
                total += wrote
                if index < len(view):
                    await asyncio.sleep(0)
        else:
            text = str(data)
            index = 0
            while index < len(text):
                piece = text[index : index + chunk_size]
                wrote = fd.write(piece)
                if wrote is None:
                    wrote = len(piece)
                if wrote <= 0:
                    raise PoolStateError("write returned non-positive progress")
                index += wrote
                total += wrote
                if index < len(text):
                    await asyncio.sleep(0)

        fd.flush()
        if self._pool.fsync_on_write:
            os.fsync(fd.fileno())
        return int(total), int(fd.tell())

    @staticmethod
    def _seek_sync(fd: IOBase, current_position: int, offset: int, whence: int) -> int:
        fd.seek(current_position)
        return int(fd.seek(offset, whence))

    @staticmethod
    def _flush_sync(fd: IOBase) -> None:
        fd.flush()

    @staticmethod
    def _truncate_sync(
        fd: IOBase,
        current_position: int,
        size: int,
        fsync_on_write: bool,
    ) -> tuple[int, int]:
        fd.seek(current_position)
        new_size = fd.truncate(size)
        fd.flush()
        if fsync_on_write:
            os.fsync(fd.fileno())
        return int(new_size), int(fd.tell())

    def _ensure_open(self) -> None:
        if self._closed:
            raise HandleClosedError("file handle is closed")
        self._pool._ensure_open()

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        warnings.warn(
            (
                f"FileHandle for '{getattr(self, '_path', '<unknown>')}' was not closed; "
                "use 'async with' or call 'await handle.close()'"
            ),
            ResourceWarning,
            stacklevel=2,
        )
        pool = getattr(self, "_pool", None)
        if pool is not None:
            try:
                pool._schedule_best_effort_cleanup(getattr(self, "_handle_id", -1))
            except Exception:
                pass
