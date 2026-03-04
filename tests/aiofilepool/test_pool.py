import asyncio

import pytest

from aiofilepool._pool import FilePoolState
from aiofilepool.errors import FilePoolNotOpenError


pytestmark = pytest.mark.asyncio


class RecordingChunker:
    def __init__(self, chunks: list[int]):
        self._chunks = chunks
        self.calls: list[int] = []

    def __call__(self, data_size: int):
        self.calls.append(data_size)
        return iter(self._chunks)


async def test_pool_context_manager_closes_and_close_is_idempotent(
    pool_factory, file_writer
) -> None:
    path = file_writer("ctx-close.bin", b"")
    pool = pool_factory()

    async with pool:
        handle = await pool.open(path, "w+")
        await handle.write(b"abc")

    with pytest.raises(FilePoolNotOpenError):
        pool.open(path, "r")

    await pool.close()


async def test_open_after_close_raises_not_open(pool_factory, file_writer) -> None:
    path = file_writer("closed-open.bin", b"")

    pool = pool_factory()
    await pool.close()

    with pytest.raises(FilePoolNotOpenError):
        pool.open(path, "w+")


async def test_threadless_mode_read_write_still_works(
    pool_factory, file_writer
) -> None:
    path = file_writer("no-thread.bin", b"")

    async with pool_factory(thread_pool_size=0, chunking_threshold=8) as pool:
        handle = await pool.open(path, "w+")
        payload = b"threadless-io"

        assert await handle.write(payload) == len(payload)
        assert await handle.read(offset=0) == payload


async def test_chunker_is_used_above_threshold_for_read_and_write(
    pool_factory, file_writer
) -> None:
    path = file_writer("chunked.bin", b"")
    payload = b"abcdefghijk"
    chunker = RecordingChunker([3, 3, 5])

    async with pool_factory(chunking_threshold=4, chunker=chunker) as pool:
        handle = await pool.open(path, "w+")
        written = await handle.write(payload)
        assert written == len(payload)

        data = await handle.read(offset=0)
        assert data == payload

    assert chunker.calls == [len(payload), len(payload)]


async def test_close_cancelled_waiter_can_still_reach_closed_state(
    pool_factory, monkeypatch
) -> None:
    pool = pool_factory()
    entered_close = asyncio.Event()

    original_close = pool._fd_manager.close

    async def delayed_close(timeout=None):
        entered_close.set()
        await asyncio.sleep(0.05)
        await original_close(timeout=timeout)

    monkeypatch.setattr(pool._fd_manager, "close", delayed_close)

    first_waiter = asyncio.create_task(pool.close())
    await entered_close.wait()
    first_waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await first_waiter

    await pool.close()
    assert pool._state == FilePoolState.CLOSED
