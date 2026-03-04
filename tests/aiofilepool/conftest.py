import io
from pathlib import Path

import pytest

from aiofilepool import FilePool


@pytest.fixture
def file_writer(tmp_path: Path):
    def _write(name: str, data: bytes) -> Path:
        path = tmp_path / name
        path.write_bytes(data)
        return path

    return _write


@pytest.fixture
def pool_factory():
    def _make_pool(
        *,
        descriptor_pool_size: int = 8,
        thread_pool_size: int = 2,
        chunking_threshold: int = 1024,
        chunker=None,
    ) -> FilePool:
        kwargs = {
            "descriptor_pool_size": descriptor_pool_size,
            "thread_pool_size": thread_pool_size,
            "chunking_threshold": chunking_threshold,
        }
        if chunker is not None:
            kwargs["chunker"] = chunker
        return FilePool(**kwargs)  # type: ignore[arg-type]

    return _make_pool


class RecordingChunker:
    def __init__(self, chunks: list[int]):
        self._chunks = chunks
        self.calls: list[int] = []

    def __call__(self, data_size: int):
        self.calls.append(data_size)
        return iter(self._chunks)


@pytest.fixture
def binary_io_factory():
    def _make(pool: FilePool, data: bytes = b""):
        bio = io.BytesIO(data)
        return pool.manage(bio), bio

    return _make
