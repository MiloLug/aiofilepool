import os
from typing import TYPE_CHECKING

from aiofilepool._modes import ModeSpec
from aiofilepool.errors import (
    HandleClosedError,
    InvalidFileModeError,
    InvalidFilePositionError,
    InvalidFileSizeError,
)

if TYPE_CHECKING:
    from aiofilepool._pool import FilePool


class FileHandle:
    def __init__(self, pool: "FilePool", path: str, mode: ModeSpec):
        self._pool = pool
        self._path = path
        self._mode = mode
        self._position = 0
        self._initialized = False
        self._size = 0

    def __await__(self):
        if not self._initialized:
            return self._initialize().__await__()
        return self

    async def __aenter__(self):
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def _initialize(self):
        stats = await self._pool._get_stats(self._path)
        self._size = stats.st_size
        self._initialized = True

    async def read(self, size: int | None = None, offset: int | None = None) -> bytes:
        if not self._mode.read:
            raise InvalidFileModeError("file is not readable")
        if not self._initialized:
            raise HandleClosedError("file is closed")

        if offset is not None:
            if offset < 0:
                raise InvalidFilePositionError("offset must be >= 0")
        else:
            offset = self._position

        if size is not None:
            if size < 0:
                raise InvalidFileSizeError("size must be >= 0")
        else:
            size = self._size - offset

        self._position = offset + size
        return await self._pool._read(self, size, offset)

    async def write(self, data: bytes, offset: int | None = None) -> int:
        if not self._mode.write:
            raise InvalidFileModeError("file is not writable")
        if not self._initialized:
            raise HandleClosedError("file is closed")

        if offset is not None:
            if offset < 0:
                raise InvalidFilePositionError("offset must be >= 0")
        else:
            offset = self._position

        written_size, error = await self._pool._write(self, data, offset)
        self._size = max(self._size, offset + written_size)
        self._position = offset + written_size
        if error:
            raise error
        return written_size

    async def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        match whence:
            case os.SEEK_SET:
                position = offset
            case os.SEEK_CUR:
                position += offset
            case os.SEEK_END:
                position = self._size - offset
            case _:
                raise ValueError("invalid whence")

        if position < 0 or position > self._size:
            raise InvalidFilePositionError(
                f"invalid position = {position} after seeking by {offset}"
            )
        self._position = position
        return position

    async def tell(self) -> int:
        return self._position

    async def truncate(self, size: int | None = None) -> None:
        if not self._mode.write:
            raise InvalidFileModeError("file is not writable")
        if not self._initialized:
            raise HandleClosedError("file is closed")

        size = size if size is not None else self._position
        if size < 0:
            raise InvalidFileSizeError("truncate size must be >= 0")

        await self._pool._truncate(self, size)
        self._size = self._position = size

    async def close(self) -> None:
        if not self._initialized:
            return
        self._closed = True
        await self._pool._close_handle(self)

    def __hash__(self) -> int:
        return id(self)
