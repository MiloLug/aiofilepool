import time
from typing import Iterator


def fixed_chunks(data_size: int, chunk_size: int) -> Iterator[int]:
    while data_size > chunk_size:
        yield chunk_size
        data_size -= chunk_size

    if data_size > 0:
        yield data_size


def balanced_chunks(
    data_size: int,
    chunk_size: int,
    scale: float = 1.1,
    min_chunk_size: int = 1024,
    max_chunk_size: int = 512 * 1024 * 1024,
) -> Iterator[int]:
    prev_time: float = 0
    prev_chunk_rate: float = 0

    while data_size > chunk_size * scale:
        current_time = time.perf_counter_ns()
        chunk_rate = (current_time - prev_time) / chunk_size

        if chunk_size < max_chunk_size and chunk_rate >= prev_chunk_rate:
            chunk_size = int(chunk_size * scale)
            print(f"Increasing chunk size to {chunk_size}")
        elif chunk_size > min_chunk_size:
            print(f"Decreasing chunk size to {chunk_size}")
            chunk_size = int(chunk_size / scale)

        prev_time = current_time
        prev_chunk_rate = chunk_rate

        yield chunk_size
        data_size -= chunk_size

    if data_size > 0:
        yield data_size
