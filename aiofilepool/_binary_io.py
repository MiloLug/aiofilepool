import asyncio
import os
from typing import AsyncGenerator, BinaryIO, TYPE_CHECKING

from aiofilepool._base_io import AsyncBinaryIO, AsyncIOState
from aiofilepool._chunking import Chunker
from aiofilepool.errors import IOInitializedError, IONotOpenError, InvalidPositionError

if TYPE_CHECKING:
    from aiofilepool._pool import FilePool


class BinaryIOAdapter(AsyncBinaryIO):
    def __init__(self, pool: "FilePool", io: BinaryIO):
        self._pool = pool
        self._io = io
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

    def _get_size(self) -> int:
        self._io.seek(0, os.SEEK_END)
        size = self._io.tell()
        self._io.seek(0, os.SEEK_SET)
        return size

    async def _initialize(self):
        if self._state != AsyncIOState.UNINITIALIZED:
            raise IOInitializedError()
        self._size = await self._pool._run_blocking(self._get_size)
        self._state = AsyncIOState.OPEN
        return self

    async def read(self, size: int | None = None, offset: int | None = None) -> bytes:
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

            self._io.seek(resolved_offset)
            if resolved_size < self._pool._chunking_threshold:
                data = await self._pool._run_blocking(self._io.read, resolved_size)
            else:
                data = bytearray()
                for chunk_size in self._pool._chunker(resolved_size):
                    data.extend(
                        await self._pool._run_blocking(self._io.read, chunk_size)
                    )

            self._position = resolved_offset + resolved_size
            return bytes(data)

    async def write(self, data: bytes, offset: int | None = None) -> int:
        if offset is not None and offset < 0:
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
            try:
                self._io.seek(resolved_offset)
                if len(data) < self._pool._chunking_threshold:
                    written = await self._pool._run_blocking(self._io.write, data)
                else:
                    for chunk_size in self._pool._chunker(len(data)):
                        written += await self._pool._run_blocking(
                            self._io.write, data[written : written + chunk_size]
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

    async def chunks(
        self,
        size: int | None = None,
        offset: int | None = None,
        chunker: Chunker | None = None,
        chunking_threshold: int | None = None,
    ) -> AsyncGenerator[bytes, None]:
        if offset is not None and offset < 0:
            raise InvalidPositionError("offset must be >= 0")

        if size is not None and size < 0:
            raise InvalidPositionError("size must be >= 0")

        if chunker is None:
            chunker = self._pool._chunker

        if chunking_threshold is None:
            chunking_threshold = self._pool._chunking_threshold

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

            self._io.seek(resolved_offset)
            consumed_size = 0
            try:
                if resolved_size < chunking_threshold:
                    data = await self._pool._run_blocking(self._io.read, resolved_size)
                    consumed_size += len(data)
                    yield data
                else:
                    for chunk_size in chunker(resolved_size):
                        data = await self._pool._run_blocking(self._io.read, chunk_size)
                        consumed_size += len(data)
                        yield data
            finally:
                self._position = resolved_offset + consumed_size

    async def fsync(self) -> None:
        async with self._op_lock:
            if self._state != AsyncIOState.OPEN:
                raise IONotOpenError()
            await self._pool._run_blocking(self._io.flush)

    async def close(self) -> None:
        if self._state != AsyncIOState.OPEN:
            return
        self._state = AsyncIOState.CLOSED

    def __hash__(self) -> int:
        return id(self)
