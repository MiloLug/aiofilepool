import asyncio
import os
from typing import TYPE_CHECKING, AsyncGenerator

from aiofilepool._base_io import AsyncBinaryIO, AsyncIOState
from aiofilepool._chunking import Chunker
from aiofilepool._modes import ModeSpec
from aiofilepool._types import StrOrBytesPath
from aiofilepool.errors import (
    IONotOpenError,
    IOInitializedError,
    InvalidFileModeError,
    InvalidPositionError,
)

if TYPE_CHECKING:
    from aiofilepool._pool import FilePool


_IS_POSIX = os.name == "posix"


class FileHandle(AsyncBinaryIO):
    def __init__(self, pool: "FilePool", path: StrOrBytesPath, mode: ModeSpec):
        self._pool = pool
        self._path: str | bytes = os.fspath(path)
        self._mode = mode
        self._position = 0
        self._size = 0
        self._state = AsyncIOState.UNINITIALIZED
        self._op_lock = asyncio.Lock()

    def __await__(self):
        return self._initialize().__await__()

    async def __aenter__(self):
        return await self._initialize()

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def _initialize(self):
        if self._state != AsyncIOState.UNINITIALIZED:
            raise IOInitializedError()
        if self._mode.truncate:
            self._size = 0
        else:
            stats = await self._pool.fs.stat(self._path)
            self._size = stats.st_size
        self._state = AsyncIOState.OPEN
        return self

    def _acquire_fd(self):
        return self._pool._fd_manager.acquire(self)

    async def read(self, size: int | None = None, offset: int | None = None) -> bytes:
        if not self._mode.read:
            raise InvalidFileModeError("file is not readable")

        if offset is not None and offset < 0:
            raise InvalidPositionError("offset must be >= 0")

        if size is not None and size < 0:
            raise InvalidPositionError("size must be >= 0")

        async with self._op_lock:
            if self._state != AsyncIOState.OPEN:
                raise IONotOpenError()

            resolved_offset = (
                self._position
                if offset is None
                else offset
                if offset < self._size
                else self._size
            )
            resolved_size = self._size - resolved_offset if size is None else size

            if resolved_size == 0:
                self._position = resolved_offset
                return bytes()

            async with self._acquire_fd() as fd:
                fd.seek(resolved_offset)
                if resolved_size < self._pool._chunking_threshold:
                    data = await self._pool._run_blocking(fd.read, resolved_size)
                else:
                    data = bytearray()
                    for chunk_size in self._pool._chunker(resolved_size):
                        data.extend(await self._pool._run_blocking(fd.read, chunk_size))
                self._position = resolved_offset + resolved_size
                return bytes(data)

    async def write(self, data: bytes, offset: int | None = None) -> int:
        if not self._mode.write:
            raise InvalidFileModeError("file is not writable")

        if offset is not None:
            if offset < 0:
                raise InvalidPositionError("offset must be >= 0")

        async with self._op_lock:
            if self._state != AsyncIOState.OPEN:
                raise IONotOpenError()

            resolved_offset = (
                self._position
                if offset is None
                else offset
                if offset < self._size
                else self._size
            )
            written = 0
            error: BaseException | None = None

            async with self._acquire_fd() as fd:
                try:
                    fd.seek(resolved_offset)
                    if len(data) < self._pool._chunking_threshold:
                        written = await self._pool._run_blocking(fd.write, data)
                    else:
                        for chunk_size in self._pool._chunker(len(data)):
                            written += await self._pool._run_blocking(
                                fd.write, data[written : written + chunk_size]
                            )
                except BaseException as e:
                    error = e

                self._size = max(self._size, resolved_offset + written)
                self._position = resolved_offset + written
                if error:
                    raise error
                return written

    async def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        position = self._position
        match whence:
            case os.SEEK_SET:
                position = offset
            case os.SEEK_CUR:
                position += offset
            case os.SEEK_END:
                position = self._size + offset
            case _:
                raise InvalidPositionError("invalid whence")

        if position < 0:
            raise InvalidPositionError(
                f"invalid position = {position} after seeking by {offset}"
            )

        if position > self._size:
            position = self._size

        self._position = position
        return position

    async def tell(self) -> int:
        return self._position

    async def size(self) -> int:
        return self._size

    async def truncate(self, size: int | None = None) -> None:
        if not self._mode.write:
            raise InvalidFileModeError("file is not writable")
        if size is not None and size < 0:
            raise InvalidPositionError("truncate size must be >= 0")

        async with self._op_lock:
            if self._state != AsyncIOState.OPEN:
                raise IONotOpenError()

            resolved_size = self._position if size is None else size

            async with self._acquire_fd() as fd:
                await self._pool._run_blocking(fd.truncate, resolved_size)
                self._size = self._position = resolved_size

    async def chunks(
        self,
        size: int | None = None,
        offset: int | None = None,
        chunker: Chunker | None = None,
        chunking_threshold: int | None = None,
    ) -> AsyncGenerator[bytes, None]:
        if not self._mode.read:
            raise InvalidFileModeError("file is not readable")

        if offset is not None and (offset < 0 or offset > self._size):
            raise InvalidPositionError("offset must be in [0, file size]")

        if size is not None and size < 0:
            raise InvalidPositionError("size must be >= 0")

        async with self._op_lock:
            if self._state != AsyncIOState.OPEN:
                raise IONotOpenError()

            resolved_offset = (
                self._position
                if offset is None
                else offset
                if offset < self._size
                else self._size
            )
            if resolved_offset >= self._size:
                self._position = self._size
                return

            resolved_size = self._size - resolved_offset if size is None else size
            resolved_chunker = self._pool._chunker if chunker is None else chunker
            resolved_threshold = (
                self._pool._chunking_threshold
                if chunking_threshold is None
                else chunking_threshold
            )

            async with self._acquire_fd() as fd:
                fd.seek(resolved_offset)
                consumed_size = 0
                try:
                    if resolved_size < resolved_threshold:
                        data = await self._pool._run_blocking(fd.read, resolved_size)
                        consumed_size += len(data)
                        yield data
                    else:
                        for chunk_size in resolved_chunker(resolved_size):
                            data = await self._pool._run_blocking(fd.read, chunk_size)
                            consumed_size += len(data)
                            yield data
                finally:
                    self._position = resolved_offset + consumed_size

    async def allocate(self, length: int) -> None:
        if not self._mode.write:
            raise InvalidFileModeError("file is not writable")
        if length < 0:
            raise InvalidPositionError("length must be >= 0")
        async with self._op_lock:
            if self._state != AsyncIOState.OPEN:
                raise IONotOpenError()
            if length <= self._size:
                return

            async with self._acquire_fd() as fd:
                if _IS_POSIX:
                    await self._pool._run_blocking(
                        os.posix_fallocate,  # type: ignore[attr-defined]
                        fd.fileno(),
                        0,
                        length,
                    )
                else:
                    await self._pool._run_blocking(fd.truncate, length)
                self._size = length

    async def close(self) -> None:
        async with self._op_lock:
            if self._state != AsyncIOState.OPEN:
                return
            await self._pool._fd_manager.discard(self)
            self._state = AsyncIOState.CLOSED

    def __hash__(self) -> int:
        return id(self)
