import asyncio
import errno
import io
import os
from collections import defaultdict
from collections.abc import Buffer, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Literal

import pytest
from hypothesis import strategies as st

from aiofilepool import AsyncBinaryIO, FilePool, StrOrBytesPath


# --- Path / fspath plumbing ---------------------------------------------------


class _PathBytes:
    """Object implementing __fspath__ → bytes (covers PEP 519 bytes-via-fspath case)."""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def __fspath__(self) -> bytes:
        return self._raw


_PATH_FACTORIES = [
    pytest.param(lambda p: p, id="path"),
    pytest.param(lambda p: str(p), id="str"),
    pytest.param(lambda p: os.fsencode(p), id="bytes"),
    pytest.param(lambda p: _PathBytes(os.fsencode(p)), id="path-bytes"),
]


@pytest.fixture(params=_PATH_FACTORIES)
def path_factory(request) -> Callable[[Path], StrOrBytesPath]:
    """Parametric: coerces a Path to each of the StrOrBytesPath shapes the pool must accept."""
    return request.param


# --- Basic IO fixtures --------------------------------------------------------


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


@pytest.fixture
def binary_io_factory():
    def _make(pool: FilePool, data: bytes = b"", io_obj: io.BytesIO | None = None):
        bio = io_obj or io.BytesIO(data)
        return pool.manage(bio), bio

    return _make


@dataclass
class AsyncIOCase:
    kind: Literal["handle", "adapter"]
    pool: FilePool
    io: AsyncBinaryIO
    backing: Path | BinaryIO


@pytest.fixture(params=["handle", "adapter"], ids=["handle", "adapter"])
def async_io_case_factory(request, pool_factory, file_writer, monkeypatch):
    """Parametric IO factory: yields a case backed by either FileHandle or BinaryIOAdapter."""
    kind = request.param

    @asynccontextmanager
    async def _make(
        *,
        data: bytes = b"",
        handle_mode: str = "r+",
        initialize: bool = True,
        thread_pool_size: int = 2,
        chunking_threshold: int = 1024,
        chunker=None,
        io_obj: BinaryIO | None = None,
        filename: str = "contract.bin",
    ) -> AsyncIterator[AsyncIOCase]:
        async with pool_factory(
            thread_pool_size=thread_pool_size,
            chunking_threshold=chunking_threshold,
            chunker=chunker,
        ) as pool:
            if kind == "handle":
                path = file_writer(filename, data)
                if io_obj is not None:
                    patch_descriptor_open(monkeypatch, lambda path, mode: io_obj)
                subject = pool.open(path, handle_mode)
                backing: Path | BinaryIO = path
            else:
                backing = io_obj or io.BytesIO(data)
                subject = pool.manage(backing)

            if initialize:
                subject = await subject

            yield AsyncIOCase(kind=kind, pool=pool, io=subject, backing=backing)

    return _make


# --- Test doubles -------------------------------------------------------------


class RecordingChunker:
    """Chunker that returns a fixed schedule and records every invocation's data_size."""

    def __init__(self, chunks: list[int]):
        self._chunks = chunks
        self.calls: list[int] = []

    def __call__(self, data_size: int):
        self.calls.append(data_size)
        return iter(self._chunks)


class AsyncGate:
    """One-shot gate: `wait()` blocks until `release()`; `entered` signals first arrival."""

    def __init__(self):
        self.entered = asyncio.Event()
        self._released = asyncio.Event()

    async def wait(self) -> None:
        self.entered.set()
        await self._released.wait()

    def release(self) -> None:
        self._released.set()


class FailingIO(io.BytesIO):
    """BytesIO raising RuntimeError on the n-th call to a chosen operation."""

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

    def read(self, size: int | None = -1) -> bytes:
        self._maybe_fail("read")
        return super().read(size)

    def write(self, data: bytes | Buffer) -> int:
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


class OSErrorIO(io.BytesIO):
    """BytesIO raising a chosen OSError on the n-th call to a chosen operation.

    Real-world failure modes are OSError(ENOSPC), PermissionError, FileNotFoundError.
    Use this (not FailingIO) when tests must verify behavior under real OS errors.
    """

    def __init__(
        self,
        data: bytes = b"",
        *,
        fail_on: str = "write",
        on_call: int = 1,
        exc_factory: Callable[[], BaseException] = lambda: OSError(
            errno.ENOSPC, "No space left on device"
        ),
    ) -> None:
        super().__init__(data)
        self.calls: dict[str, int] = defaultdict(int)
        self._fail_on = fail_on
        self._on_call = on_call
        self._exc_factory = exc_factory

    def _maybe_fail(self, op: str) -> None:
        self.calls[op] += 1
        if op == self._fail_on and self.calls[op] == self._on_call:
            raise self._exc_factory()

    def read(self, size: int | None = -1) -> bytes:
        self._maybe_fail("read")
        return super().read(size)

    def write(self, data: bytes | Buffer) -> int:
        self._maybe_fail("write")
        return super().write(data)

    def flush(self) -> None:
        self._maybe_fail("flush")

    def close(self) -> None:
        self._maybe_fail("close")
        super().close()


# --- Hypothesis strategies / oracles ------------------------------------------


@st.composite
def disjoint_intervals(
    draw,
    *,
    max_total_size: int = 4096,
    max_intervals: int = 16,
    payload_max: int = 256,
) -> list[tuple[int, bytes]]:
    """Sorted, non-overlapping (offset, payload) intervals fitting in [0, max_total_size].

    Used to drive scatter-write / scatter-read property tests against the assembled-buffer
    oracle (see assemble_intervals). Overlaps from the raw draw are dropped (later wins-by-skip)
    rather than coalesced, so the resulting set is provably disjoint.
    """
    raw = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=max(0, max_total_size - 1)),
                st.binary(min_size=1, max_size=payload_max),
            ),
            min_size=1,
            max_size=max_intervals,
        )
    )
    raw.sort(key=lambda x: x[0])
    intervals: list[tuple[int, bytes]] = []
    cursor = 0
    for offset, payload in raw:
        if offset < cursor:
            continue
        if offset + len(payload) > max_total_size:
            keep = max_total_size - offset
            if keep <= 0:
                continue
            payload = payload[:keep]
        intervals.append((offset, payload))
        cursor = offset + len(payload)
    return intervals


def assemble_intervals(
    intervals: list[tuple[int, bytes]], total_size: int | None = None
) -> bytes:
    """Expected on-disk bytes for a set of disjoint (offset, payload) writes.

    Gaps are zero-padded (matching `allocate()` + scattered `write(offset=)` semantics).
    """
    size = (
        total_size
        if total_size is not None
        else max((off + len(data) for off, data in intervals), default=0)
    )
    buf = bytearray(size)
    for offset, data in intervals:
        end = offset + len(data)
        if end > size:
            data = data[: size - offset]
            end = size
        if offset < size:
            buf[offset:end] = data
    return bytes(buf)


# --- Observability helpers ----------------------------------------------------


def patch_descriptor_open(
    monkeypatch: pytest.MonkeyPatch, opener: Callable[..., Any]
) -> None:
    """Intercept the descriptor-opening `open()` inside `FileHandle._open_fd`.

    Descriptors are opened via the module-global `open` in `aiofilepool._handle`;
    patch that name (never `FileHandle._open_fd` itself) so the mode-selection and
    `_fd_materialized` logic stays under test.
    """
    monkeypatch.setattr("aiofilepool._handle.open", opener, raising=False)


def count_fd(pool: FilePool) -> int:
    """Snapshot the count of currently-open OS file descriptors held by the pool."""
    return len(pool._fd_manager._descriptors)  # noqa: SLF001


def count_blocking_calls(pool: FilePool, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Spy on pool._run_blocking; return a mutable list of dispatched function names.

    Tests that assert "no dispatch happened" (e.g. allocate idempotency) consult the list
    after the operation to verify the no-dispatch contract.
    """
    calls: list[str] = []
    original_run_blocking = pool._run_blocking

    async def spy(func, *args):
        calls.append(getattr(func, "__name__", repr(func)))
        return await original_run_blocking(func, *args)

    monkeypatch.setattr(pool, "_run_blocking", spy)
    return calls
