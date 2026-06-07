from abc import ABC, abstractmethod
import time
from collections.abc import Iterator


class Chunker(ABC):
    @abstractmethod
    def __call__(self, data_size: int) -> Iterator[int]: ...


class FixedChunker(Chunker):
    """
    A chunker that returns a fixed chunk size.
    """

    __slots__ = ("_chunk_size",)

    def __init__(self, chunk_size: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self._chunk_size = chunk_size

    def __call__(self, data_size: int) -> Iterator[int]:
        while data_size > self._chunk_size:
            yield self._chunk_size
            data_size -= self._chunk_size

        if data_size > 0:
            yield data_size


class BalancedChunker(Chunker):
    """
    A chunker that balances the chunk size based on the time it takes to read the chunk.
    """

    __slots__ = ("_scale_up", "_scale_down", "_min_chunk_size", "_max_chunk_size")

    def __init__(
        self,
        scale_up: float = 1.2,
        scale_down: float = 0.9,
        min_chunk_size: int = 1024 * 1024,
        max_chunk_size: int = 512 * 1024 * 1024,
    ):
        """
        Args:
            scale_up: The factor to scale the chunk size up.
            scale_down: The factor to scale the chunk size down.
            min_chunk_size: The minimum chunk size.
            max_chunk_size: The maximum chunk size.
        """

        if scale_up <= 1:
            raise ValueError("scale_up must be > 1")
        if not (0 < scale_down < 1):
            raise ValueError("scale_down must be between 0 and 1")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be > 0")
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be > 0")
        if min_chunk_size > max_chunk_size:
            raise ValueError("min_chunk_size must be <= max_chunk_size")
        if int(min_chunk_size * scale_up) == min_chunk_size:
            raise ValueError("min_chunk_size is too small to scale up")

        self._scale_up = scale_up
        self._scale_down = scale_down
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size

    def __call__(self, data_size: int) -> Iterator[int]:
        chunk_size = self._min_chunk_size
        prev_time: float = 0
        prev_chunk_rate: float = 0

        while data_size > chunk_size * self._scale_up:
            current_time = time.perf_counter_ns()
            chunk_rate = (current_time - prev_time) / chunk_size

            if chunk_size < self._max_chunk_size and chunk_rate >= prev_chunk_rate:
                chunk_size = min(
                    self._max_chunk_size,
                    int(chunk_size * self._scale_up),
                )
            elif chunk_size > self._min_chunk_size:
                chunk_size = max(
                    self._min_chunk_size,
                    int(chunk_size * self._scale_down),
                )

            prev_time = current_time
            prev_chunk_rate = chunk_rate

            yield chunk_size
            data_size -= chunk_size

        while data_size > self._max_chunk_size:
            yield self._max_chunk_size
            data_size -= self._max_chunk_size

        if data_size > 0:
            yield data_size
