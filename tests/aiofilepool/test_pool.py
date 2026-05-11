import asyncio
import os
import threading
from pathlib import Path

import pytest

from aiofilepool import FilePool, StrOrBytesPath
from aiofilepool._fs import AsyncFileSystem
from aiofilepool.errors import FilePoolNotOpenError

from .conftest import AsyncGate, FailingIO


pytestmark = pytest.mark.asyncio


class _PathBytes:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def __fspath__(self) -> bytes:
        return self._raw


def _as_path(path: Path) -> StrOrBytesPath:
    return path


def _as_str(path: Path) -> StrOrBytesPath:
    return str(path)


def _as_bytes(path: Path) -> StrOrBytesPath:
    return os.fsencode(path)


def _as_path_bytes(path: Path) -> StrOrBytesPath:
    return _PathBytes(os.fsencode(path))


_PATH_FACTORIES = [
    pytest.param(_as_path, id="path"),
    pytest.param(_as_str, id="str"),
    pytest.param(_as_bytes, id="bytes"),
    pytest.param(_as_path_bytes, id="path-bytes"),
]


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


@pytest.mark.parametrize("path_factory", _PATH_FACTORIES)
async def test_open_accepts_strorbytes_path_inputs(
    pool_factory, file_writer, path_factory
) -> None:
    path = file_writer("strpath-open.bin", b"")

    async with pool_factory(thread_pool_size=0) as pool:
        handle = await pool.open(path_factory(path), "w+")

        assert await handle.write(b"payload") == 7
        assert await handle.read(offset=0) == b"payload"


async def test_threadless_mode_read_write_still_works(
    pool_factory, file_writer
) -> None:
    path = file_writer("no-thread.bin", b"")

    async with pool_factory(thread_pool_size=0, chunking_threshold=8) as pool:
        handle = await pool.open(path, "w+")
        payload = b"threadless-io"

        assert await handle.write(payload) == len(payload)
        assert await handle.read(offset=0) == payload


async def test_close_cancelled_waiter_can_still_reach_closed_state(
    pool_factory, file_writer, monkeypatch
) -> None:
    pool = pool_factory()
    entered_close = asyncio.Event()
    path = file_writer("closed-after-retry.bin", b"")

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

    with pytest.raises(FilePoolNotOpenError):
        pool.open(path, "r")


async def test_concurrent_close_callers_share_a_single_close_task(
    pool_factory, file_writer, monkeypatch
) -> None:
    pool = pool_factory()
    gate = AsyncGate()
    close_calls = 0
    path = file_writer("closed-concurrently.bin", b"")
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

    gate.release()

    await asyncio.gather(*waiters)

    with pytest.raises(FilePoolNotOpenError):
        pool.open(path, "r")


async def test_close_timeout_can_retry_and_finish_with_active_operation(
    pool_factory, file_writer, monkeypatch
) -> None:
    pool = pool_factory(thread_pool_size=1)
    gate = AsyncGate()
    path = file_writer("close-timeout-active.bin", b"")
    handle = await pool.open(path, "w+")
    original_run_blocking = pool._run_blocking
    call_count = 0

    async def gated_run_blocking(func, *args):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await gate.wait()
        return await original_run_blocking(func, *args)

    monkeypatch.setattr(pool, "_run_blocking", gated_run_blocking)

    write_task = asyncio.create_task(handle.write(b"abc"))
    await gate.entered.wait()

    with pytest.raises(asyncio.TimeoutError):
        await pool.close(timeout=0.01)

    gate.release()

    assert await write_task == 3
    await pool.close()
    with pytest.raises(FilePoolNotOpenError):
        pool.open(path, "r")


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


async def test_default_fs_is_async_file_system_bound_to_pool(pool_factory) -> None:
    async with pool_factory() as pool:
        assert isinstance(pool.fs, AsyncFileSystem)
        assert pool.fs._pool is pool


async def test_custom_fs_is_honored_via_constructor_kwarg(file_writer) -> None:
    path = file_writer("custom-fs.bin", b"abcdef")
    stat_calls: list[str | bytes] = []

    class _SpyFS(AsyncFileSystem):
        def __init__(self) -> None:
            self._pool = None  # type: ignore[assignment]
            self.stat_calls = stat_calls

        async def stat(self, p):
            self.stat_calls.append(p)
            return os.stat(p)

    spy = _SpyFS()
    pool = FilePool(descriptor_pool_size=4, thread_pool_size=1, fs=spy)

    async with pool:
        assert pool.fs is spy
        handle = await pool.open(path, "r")
        assert stat_calls == [os.fspath(path)]
        assert await handle.size() == 6
