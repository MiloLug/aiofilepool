import asyncio
import io
import threading
from pathlib import Path

import pytest

from aiofilepool import FilePool, StrPath
from aiofilepool._pool import FilePoolState
from aiofilepool.errors import FilePoolNotOpenError

from .conftest import AsyncGate, FailingIO, RecordingChunker


pytestmark = pytest.mark.asyncio


def _as_path(path: Path) -> StrPath:
    return path


def _as_str(path: Path) -> StrPath:
    return str(path)


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


async def test_stat_returns_size_for_existing_file(pool_factory, file_writer) -> None:
    path = file_writer("stats.bin", b"abcdef")

    async with pool_factory() as pool:
        stat = await pool.stat(path)
        assert stat.st_size == 6


async def test_package_exports_strpath_alias() -> None:
    assert StrPath is not None


@pytest.mark.parametrize(
    "path_factory",
    [
        pytest.param(_as_path, id="path"),
        pytest.param(_as_str, id="str"),
    ],
)
async def test_open_accepts_strpath_inputs(
    pool_factory, file_writer, path_factory
) -> None:
    path = file_writer("strpath-open.bin", b"")

    async with pool_factory(thread_pool_size=0) as pool:
        handle = await pool.open(path_factory(path), "w+")

        assert isinstance(handle._path, str)
        assert await handle.write(b"payload") == 7
        assert await handle.read(offset=0) == b"payload"


@pytest.mark.parametrize(
    "path_factory",
    [
        pytest.param(_as_path, id="path"),
        pytest.param(_as_str, id="str"),
    ],
)
async def test_stat_accepts_strpath_inputs(
    pool_factory, file_writer, path_factory
) -> None:
    path = file_writer("strpath-stat.bin", b"abcdef")

    async with pool_factory() as pool:
        stat = await pool.stat(path_factory(path))

    assert stat.st_size == 6


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


async def test_concurrent_close_callers_share_a_single_close_task(
    pool_factory, monkeypatch
) -> None:
    pool = pool_factory()
    gate = AsyncGate()
    close_calls = 0
    original_close = pool._fd_manager.close

    async def delayed_close(timeout=None):
        nonlocal close_calls
        close_calls += 1
        await gate.wait()
        await original_close(timeout=timeout)

    monkeypatch.setattr(pool._fd_manager, "close", delayed_close)

    waiters = [asyncio.create_task(pool.close()) for _ in range(3)]
    await gate.entered.wait()
    assert close_calls == 1
    assert pool._state == FilePoolState.CLOSING

    gate.release()

    await asyncio.gather(*waiters)
    assert pool._state == FilePoolState.CLOSED
    assert pool._close_task is not None


async def test_close_timeout_can_retry_and_finish(pool_factory, monkeypatch) -> None:
    pool = pool_factory()
    gate = AsyncGate()
    original_close = pool._fd_manager.close

    async def delayed_close(timeout=None):
        await gate.wait()
        await original_close(timeout=timeout)

    monkeypatch.setattr(pool._fd_manager, "close", delayed_close)

    with pytest.raises(asyncio.TimeoutError):
        await pool.close(timeout=0.01)

    assert pool._state == FilePoolState.CLOSING
    assert pool._close_task is not None
    gate.release()

    await pool.close()
    assert pool._state == FilePoolState.CLOSED


async def test_run_blocking_waits_for_executor_work_before_cancellation_propagates(
    pool_factory,
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_call() -> str:
        started.set()
        release.wait(timeout=1)
        finished.set()
        return "done"

    async with pool_factory(thread_pool_size=1) as pool:
        task = asyncio.create_task(pool._run_blocking(blocking_call))
        assert await asyncio.to_thread(started.wait, 1) is True

        task.cancel()
        await asyncio.sleep(0)
        assert finished.is_set() is False

        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert await asyncio.to_thread(finished.wait, 1) is True


async def test_pool_close_surfaces_cold_descriptor_flush_errors_after_other_cleanup(
    pool_factory, file_writer, monkeypatch
) -> None:
    path_one = file_writer("flush-error-one.bin", b"")
    path_two = file_writer("flush-error-two.bin", b"")
    bad_io = FailingIO(fail_on={"flush": 1})
    good_io = FailingIO()
    opened_ios = iter([bad_io, good_io])

    def fake_open(path, mode):
        return next(opened_ios)

    monkeypatch.setattr("aiofilepool._fd_manager.open", fake_open, raising=False)

    pool = pool_factory(descriptor_pool_size=2, thread_pool_size=0)
    first = await pool.open(path_one, "w+")
    second = await pool.open(path_two, "w+")
    await first.write(b"abc")
    await second.write(b"xyz")

    with pytest.raises(RuntimeError, match="flush failed"):
        await pool.close()

    assert good_io.closed is True


async def test_file_pool_constructor_rejects_invalid_descriptor_pool_size() -> None:
    with pytest.raises(ValueError, match="max_descriptors must be >= 1"):
        FilePool(descriptor_pool_size=0)


async def test_stat_still_works_after_pool_close(pool_factory, file_writer) -> None:
    path = file_writer("post-close-stat.bin", b"abcdef")
    pool = pool_factory()
    await pool.close()

    stat = await pool.stat(path)

    assert stat.st_size == 6


async def test_manage_still_works_after_pool_close(pool_factory) -> None:
    pool = pool_factory()
    await pool.close()

    adapter = await pool.manage(io.BytesIO(b"abc"))

    assert await adapter.read() == b"abc"
