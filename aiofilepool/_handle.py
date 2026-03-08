import asyncio
import os
from typing import TYPE_CHECKING, AsyncGenerator

from aiofilepool._base_io import AsyncBinaryIO, AsyncIOState
from aiofilepool._chunking import Chunker
from aiofilepool._modes import ModeSpec
from aiofilepool._types import StrPath
from aiofilepool.errors import (
    IONotOpenError,
    IOInitializedError,
    InvalidFileModeError,
    InvalidPositionError,
)

if TYPE_CHECKING:
    from aiofilepool._pool import FilePool


class FileHandle(AsyncBinaryIO):
    def __init__(self, pool: "FilePool", path: StrPath, mode: ModeSpec):
        self._pool = pool
        self._path: str = os.fspath(path)
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
            stats = await self._pool.stat(self._path)
            self._size = stats.st_size
        self._state = AsyncIOState.OPEN
        return self

    def _acquire_fd(self):
        return self._pool._fd_manager.acquire(self)

    async def read(self, size: int | None = None, offset: int | None = None) -> bytes:
        if not self._mode.read:
            raise InvalidFileModeError("file is not readable")
        if self._state != AsyncIOState.OPEN:
            raise IONotOpenError()

        if offset is not None:
            if offset < 0:
                raise InvalidPositionError("offset must be >= 0")
        else:
            offset = self._position

        if size is not None:
            if size < 0:
                raise InvalidPositionError("size must be >= 0")
        else:
            size = self._size - offset

        async with self._op_lock, self._acquire_fd() as fd:
            fd.seek(offset)
            if size < self._pool._chunking_threshold:
                data = await self._pool._run_blocking(fd.read, size)
            else:
                data = bytearray()
                for chunk_size in self._pool._chunker(size):
                    data.extend(await self._pool._run_blocking(fd.read, chunk_size))
            self._position = offset + size
            return bytes(data)

    async def write(self, data: bytes, offset: int | None = None) -> int:
        if not self._mode.write:
            raise InvalidFileModeError("file is not writable")
        if self._state != AsyncIOState.OPEN:
            raise IONotOpenError()

        if offset is not None:
            if offset < 0:
                raise InvalidPositionError("offset must be >= 0")
        else:
            offset = self._position

        written = 0
        error: BaseException | None = None

        async with self._op_lock, self._acquire_fd() as fd:
            try:
                fd.seek(offset)
                if len(data) < self._pool._chunking_threshold:
                    written = await self._pool._run_blocking(fd.write, data)
                else:
                    for chunk_size in self._pool._chunker(len(data)):
                        written += await self._pool._run_blocking(
                            fd.write, data[written : written + chunk_size]
                        )
            except BaseException as e:
                error = e

            self._size = max(self._size, offset + written)
            self._position = offset + written
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

        if position < 0 or position > self._size:
            raise InvalidPositionError(
                f"invalid position = {position} after seeking by {offset}"
            )
        self._position = position
        return position

    async def tell(self) -> int:
        return self._position

    async def size(self) -> int:
        return self._size

    async def truncate(self, size: int | None = None) -> None:
        if not self._mode.write:
            raise InvalidFileModeError("file is not writable")
        if self._state != AsyncIOState.OPEN:
            raise IONotOpenError()

        size = size if size is not None else self._position
        if size < 0:
            raise InvalidPositionError("truncate size must be >= 0")

        async with self._op_lock, self._acquire_fd() as fd:
            await self._pool._run_blocking(fd.truncate, size)
            self._size = self._position = size

    async def chunks(
        self,
        size: int | None = None,
        offset: int | None = None,
        chunker: Chunker | None = None,
        chunking_threshold: int | None = None,
    ) -> AsyncGenerator[bytes, None]:
        if not self._mode.read:
            raise InvalidFileModeError("file is not readable")
        if self._state != AsyncIOState.OPEN:
            raise IONotOpenError()

        if offset is not None:
            if offset < 0:
                raise InvalidPositionError("offset must be >= 0")
        else:
            offset = self._position

        if size is not None:
            if size < 0:
                raise InvalidPositionError("size must be >= 0")
        else:
            size = self._size - offset

        if chunker is None:
            chunker = self._pool._chunker

        if chunking_threshold is None:
            chunking_threshold = self._pool._chunking_threshold

        async with self._op_lock, self._acquire_fd() as fd:
            fd.seek(offset)
            consumed_size = 0
            try:
                if size < chunking_threshold:
                    data = await self._pool._run_blocking(fd.read, size)
                    consumed_size += len(data)
                    yield data
                else:
                    for chunk_size in chunker(size):
                        data = await self._pool._run_blocking(fd.read, chunk_size)
                        consumed_size += len(data)
                        yield data
            finally:
                self._position = offset + consumed_size

    async def close(self) -> None:
        async with self._op_lock:
            if self._state != AsyncIOState.OPEN:
                return
            await self._pool._fd_manager.discard(self)
            self._state = AsyncIOState.CLOSED

    def __hash__(self) -> int:
        return id(self)
