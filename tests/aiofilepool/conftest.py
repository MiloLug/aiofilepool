import asyncio
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

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


class AsyncGate:
    def __init__(self):
        self.entered = asyncio.Event()
        self._released = asyncio.Event()

    async def wait(self) -> None:
        self.entered.set()
        await self._released.wait()

    def release(self) -> None:
        self._released.set()


class FailingIO(io.BytesIO):
    def __init__(
        self,
        data: bytes = b"",
        *,
        fail_on: dict[str, int] | None = None,
    ):
        super().__init__(data)
        self.calls: dict[str, int] = defaultdict(int)
        self._fail_on = fail_on or {}
        self.close_attempted = False

    def _maybe_fail(self, operation: str) -> None:
        self.calls[operation] += 1
        if self._fail_on.get(operation) == self.calls[operation]:
            raise RuntimeError(f"{operation} failed")

    def read(self, size: int = -1) -> bytes:
        self._maybe_fail("read")
        return super().read(size)

    def write(self, data: bytes) -> int:
        self._maybe_fail("write")
        return super().write(data)

    def truncate(self, size: int | None = None) -> int:
        self._maybe_fail("truncate")
        return super().truncate(size)

    def flush(self) -> None:
        self._maybe_fail("flush")

    def close(self) -> None:
        self.close_attempted = True
        self._maybe_fail("close")
        super().close()


@pytest.fixture
def binary_io_factory():
    def _make(pool: FilePool, data: bytes = b"", io_obj: io.BytesIO | None = None):
        bio = io_obj or io.BytesIO(data)
        return pool.manage(bio), bio

    return _make


@pytest.fixture
def stressed_pool_factory(pool_factory):
    def _make(**kwargs: Any) -> FilePool:
        return pool_factory(
            descriptor_pool_size=1,
            thread_pool_size=0,
            chunking_threshold=4,
            **kwargs,
        )

    return _make
