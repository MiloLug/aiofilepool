from abc import ABC, abstractmethod
from enum import IntEnum
import os
from typing import AsyncGenerator

from aiofilepool._chunking import Chunker


class AsyncIOState(IntEnum):
    UNINITIALIZED = 0
    CLOSED = 1
    OPEN = 2


class AsyncBinaryIO(ABC):
    @abstractmethod
    async def read(
        self, size: int | None = None, offset: int | None = None
    ) -> bytes: ...

    @abstractmethod
    async def write(self, data: bytes, offset: int | None = None) -> int: ...

    @abstractmethod
    async def seek(self, offset: int, whence: int = os.SEEK_SET) -> int: ...

    @abstractmethod
    async def tell(self) -> int: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def size(self) -> int: ...

    @abstractmethod
    def chunks(
        self,
        size: int | None = None,
        offset: int | None = None,
        chunker: Chunker | None = None,
        chunking_threshold: int | None = None,
    ) -> AsyncGenerator[bytes, None]: ...
