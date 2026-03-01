import asyncio
import math
import os
from typing import Self

from aiofilepool._fd_manager import FileDescriptorManager
from aiofilepool._handle import FileHandle

from ._modes import ModeSpec


class FilePool:
    def __init__(
        self,
        descriptor_pool_size: int = 256,
        thread_pool_size: int = 4,
        chunk_size: int = 64 * 1024 * 1024,
        fsync_on_write: bool = False,
    ):
        self._thread_pool_size = thread_pool_size
        self._chunk_size = chunk_size
        self._fsync_on_write = fsync_on_write
        self._fd_manager = FileDescriptorManager(max_descriptors=descriptor_pool_size)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        self._fd_manager.close_all()

    async def _close_handle(self, handle: FileHandle) -> None:
        self._fd_manager.close(handle)

    async def _get_stats(self, path: str) -> os.stat_result:
        stat = os.stat(path)
        return stat

    async def _read(self, handle: FileHandle, size: int, offset: int) -> bytes:
        fd = self._fd_manager.acquire(handle)
        try:
            fd.seek(offset)
            data = bytearray()
            chunks = math.ceil(size / self._chunk_size)
            rest = size % self._chunk_size
            for _ in range(chunks - 1):
                data.extend(fd.read(self._chunk_size))
                await asyncio.sleep(0)
            if rest > 0:
                data.extend(fd.read(rest))
        finally:
            self._fd_manager.release(handle)

        return bytes(data)

    async def _write(
        self, handle: FileHandle, data: bytes, offset: int
    ) -> tuple[int, Exception | None]:
        written = 0
        error = None
        chunks = math.ceil(len(data) / self._chunk_size)
        rest = len(data) % self._chunk_size

        fd = self._fd_manager.acquire(handle)
        fd.seek(offset)
        try:
            for i in range(chunks - 1):
                written += fd.write(
                    data[i * self._chunk_size : (i + 1) * self._chunk_size]
                )
                await asyncio.sleep(0)
            if rest > 0:
                written += fd.write(data[-rest:])
        except Exception as e:
            error = e
        finally:
            self._fd_manager.release(handle)

        return written, error

    async def _truncate(self, handle: FileHandle, size: int) -> None:
        fd = self._fd_manager.acquire(handle)
        try:
            fd.truncate(size)
        finally:
            self._fd_manager.release(handle)

    def open(
        self,
        path: str | os.PathLike[str],
        mode: str,
    ) -> FileHandle:
        mode_spec = ModeSpec.from_str(mode)
        return FileHandle(
            pool=self,
            path=os.fspath(path),
            mode=mode_spec,
        )
