from abc import ABC, abstractmethod
from enum import IntEnum
import os
from typing import AsyncGenerator

from aiofilepool._chunking import Chunker
from aiofilepool._types import Whence


class AsyncIOState(IntEnum):
    UNINITIALIZED = 0
    CLOSED = 1
    OPEN = 2


class AsyncBinaryIO(ABC):
    @abstractmethod
    async def read(self, size: int | None = None, offset: int | None = None) -> bytes:
        """
        Read bytes from the file. Changes the current position.

        Args:
            size: The maximum number of bytes to read. None -> read to the end of the file.
            offset: The offset to read from. None -> read from the current position.
        Returns:
            The bytes read from the file.
        """

    @abstractmethod
    async def write(self, data: bytes, offset: int | None = None) -> int:
        """
        Write bytes to the file. Changes the current position.

        Args:
            data: The bytes to write.
            offset: The offset to write to. None -> write to the current position.
        Returns:
            The number of bytes written.
        """

    @abstractmethod
    async def seek(self, offset: int, whence: Whence = os.SEEK_SET) -> int:
        """
        Seek to a new position in the file.

        Args:
            offset: The offset to seek to.
            whence: The reference point for the offset.
                os.SEEK_SET | os.SEEK_CUR | os.SEEK_END
        Returns:
            The new position in the file.
        """

    @abstractmethod
    async def tell(self) -> int:
        """Get the current position in the file."""

    @abstractmethod
    async def close(self) -> None:
        """Close the file."""

    @abstractmethod
    async def fsync(self) -> None:
        """Flush the file to disk."""

    @abstractmethod
    async def size(self) -> int:
        """Get the size of the file."""

    @abstractmethod
    def chunks(
        self,
        size: int | None = None,
        offset: int | None = None,
        chunker: Chunker | None = None,
        chunking_threshold: int | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Get a chunked reader of bytes from the file.

        Args:
            size: The maximum number of bytes to read.
                None -> read to the end of the file.
            offset: The offset to read from.
                None -> read from the current position.
            chunker: The chunker to use.
                None -> use the default chunker.
            chunking_threshold: The threshold at which to switch to chunking.
                None -> use the default threshold.
        Returns:
            A generator of bytes from the file.
        """
