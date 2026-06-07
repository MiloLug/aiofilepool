"""`fsync` durability primitive on the async IO contract.

`FileHandle.fsync()` must flush the buffered userspace fd FIRST, then call
`os.fsync` on its descriptor — flushing alone leaves bytes in the kernel page
cache, and fsync-without-flush misses the userspace buffer. The download
pipeline relies on this ordering to make "part row deleted ⇒ bytes durable"
hold across a crash.

`BinaryIOAdapter.fsync()` wraps a caller-owned in-memory/arbitrary BinaryIO with
no real descriptor, so it flushes only.
"""

import io
import os

import pytest

from aiofilepool.errors import IONotOpenError


pytestmark = pytest.mark.asyncio


class _RecordingFd(io.BytesIO):
    """A fake fd that records flush() and exposes a sentinel fileno()."""

    def __init__(self, events: list[object]) -> None:
        super().__init__()
        self._events = events

    def flush(self) -> None:
        self._events.append("flush")

    def fileno(self) -> int:
        return 4242


async def test_handle_fsync_flushes_buffer_then_fsyncs_fd(
    pool_factory, file_writer, monkeypatch
) -> None:
    events: list[object] = []
    rec = _RecordingFd(events)
    monkeypatch.setattr(
        "aiofilepool._fd_manager.open", lambda path, mode: rec, raising=False
    )
    monkeypatch.setattr(os, "fsync", lambda fileno: events.append(("fsync", fileno)))

    path = file_writer("fsync.bin", b"")
    async with pool_factory(thread_pool_size=0) as pool:
        handle = await pool.open(path, "w+b")
        await handle.write(b"abc")
        events.clear()  # ignore anything emitted by the write/cool path

        await handle.fsync()

        assert events == ["flush", ("fsync", 4242)]
        await handle.close()


async def test_adapter_fsync_flushes_underlying_only(pool_factory) -> None:
    events: list[str] = []

    class _Rec(io.BytesIO):
        def flush(self) -> None:
            events.append("flush")

    async with pool_factory() as pool:
        adapter = await pool.manage(_Rec())
        await adapter.write(b"abc")
        events.clear()

        await adapter.fsync()

        assert events == ["flush"]


async def test_fsync_on_closed_handle_raises(pool_factory, file_writer) -> None:
    path = file_writer("fsync-closed.bin", b"")
    async with pool_factory() as pool:
        handle = await pool.open(path, "w+b")
        await handle.close()

        with pytest.raises(IONotOpenError):
            await handle.fsync()
