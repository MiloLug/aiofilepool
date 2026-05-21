"""`FilePool` lifecycle, cancellation safety, and `AsyncFileSystem` delegation.

Pins:
* state machine: OPEN → CLOSING → CLOSED, idempotent close, post-close `open()` rejects
* concurrent close callers share one underlying `_close_task` (assert by identity)
* `close(timeout=)` raises `TimeoutError` without aborting; next `close()` finishes
* cancellation: cancelled first close waiter still drives the pool to CLOSED;
  `_run_blocking` shields executor work from cancellation
* threadless mode supports the full IO surface
* constructor rejects `descriptor_pool_size=0`
* `AsyncFileSystem` delegates `stat / rename / move / copyfile / mkdir / exists /
  is_file / is_dir / remove` to the pool's executor
"""

import asyncio
import os
import shutil
import threading
from pathlib import Path

import pytest

from aiofilepool import FilePool
from aiofilepool._fs import AsyncFileSystem
from aiofilepool.errors import FilePoolNotOpenError

from .conftest import AsyncGate, FailingIO


pytestmark = pytest.mark.asyncio


# --- Lifecycle ----------------------------------------------------------------


async def test_async_context_manager_closes_and_close_is_idempotent(
    pool_factory, file_writer
) -> None:
    path = file_writer("ctx-close.bin", b"")
    pool = pool_factory()

    async with pool:
        handle = await pool.open(path, "w+")
        await handle.write(b"abc")

    # After context exit, pool is closed; further open() rejects.
    with pytest.raises(FilePoolNotOpenError):
        pool.open(path, "r")

    # Second close() is a no-op (returns without raising).
    await pool.close()


async def test_open_after_close_raises_not_open(pool_factory, file_writer) -> None:
    path = file_writer("closed-open.bin", b"")

    pool = pool_factory()
    await pool.close()

    with pytest.raises(FilePoolNotOpenError):
        pool.open(path, "w+")


async def test_threadless_mode_supports_full_io_surface(
    pool_factory, file_writer
) -> None:
    """`thread_pool_size=0` falls back to direct synchronous dispatch in `_run_blocking`."""
    path = file_writer("no-thread.bin", b"")

    async with pool_factory(thread_pool_size=0, chunking_threshold=8) as pool:
        handle = await pool.open(path, "w+")
        payload = b"threadless-io"

        assert await handle.write(payload) == len(payload)
        assert await handle.read(offset=0) == payload
        assert await handle.size() == len(payload)
        await handle.truncate(4)
        assert await handle.read(offset=0) == b"thre"


async def test_constructor_rejects_zero_descriptor_pool_size() -> None:
    with pytest.raises(ValueError, match="max_descriptors must be >= 1"):
        FilePool(descriptor_pool_size=0)


# --- Concurrent close + cancellation safety -----------------------------------


async def test_concurrent_close_callers_share_one_underlying_task(
    pool_factory, file_writer, monkeypatch
) -> None:
    """All concurrent `close()` callers must observe the same `_close_task` object."""
    pool = pool_factory()
    gate = AsyncGate()
    close_calls = 0
    path = file_writer("close-shared.bin", b"")
    original_close = pool._fd_manager.close

    async def delayed_close(timeout=None):
        nonlocal close_calls
        close_calls += 1
        await gate.wait()
        await original_close(timeout=timeout)

    monkeypatch.setattr(pool._fd_manager, "close", delayed_close)

    waiters = [asyncio.create_task(pool.close()) for _ in range(3)]
    await gate.entered.wait()

    # Only one underlying close ran, and pool._close_task is set to the single task.
    assert close_calls == 1
    assert pool._close_task is not None
    task_identity = pool._close_task

    gate.release()
    await asyncio.gather(*waiters)

    # The task object is preserved (not reset) after completion.
    assert pool._close_task is task_identity
    with pytest.raises(FilePoolNotOpenError):
        pool.open(path, "r")


async def test_cancelled_first_close_waiter_still_drives_pool_to_closed(
    pool_factory, file_writer, monkeypatch
) -> None:
    """Cancelling a `close()` waiter must still complete the underlying close eventually."""
    pool = pool_factory()
    entered_close = asyncio.Event()
    path = file_writer("close-after-cancel.bin", b"")

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

    # Subsequent close() finishes the drain and transitions to CLOSED.
    await pool.close()

    with pytest.raises(FilePoolNotOpenError):
        pool.open(path, "r")


async def test_close_timeout_raises_without_aborting_inflight_op(
    pool_factory, file_writer, monkeypatch
) -> None:
    """`close(timeout=)` surfaces `TimeoutError` but does not cancel the inflight op;
    a subsequent `close()` returns once the op finishes."""
    pool = pool_factory(thread_pool_size=1)
    gate = AsyncGate()
    path = file_writer("close-timeout.bin", b"")
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


async def test_run_blocking_shields_executor_from_cancellation(pool_factory) -> None:
    """A cancel on `_run_blocking` must let the executor finish its work before the
    `CancelledError` propagates — preserves on-disk state integrity."""
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
        # Cancel must not interrupt the blocking call mid-flight.
        assert finished.is_set() is False

        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert await asyncio.to_thread(finished.wait, 1) is True


async def test_close_surfaces_cold_descriptor_flush_error_after_other_cleanup(
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

    # The non-failing descriptor still got closed during cleanup.
    assert good_io.closed is True


async def test_custom_fs_is_honored_via_constructor(file_writer) -> None:
    """A user-supplied `AsyncFileSystem` replaces the default and is what `_initialize` calls."""
    path = file_writer("custom-fs.bin", b"abcdef")
    stat_calls: list[str | bytes] = []

    class _SpyFS(AsyncFileSystem):
        async def stat(self, p):
            stat_calls.append(p)
            return os.stat(p)

    spy = _SpyFS(pool=None)  # type: ignore[arg-type]
    pool = FilePool(descriptor_pool_size=4, thread_pool_size=1, fs=spy)

    async with pool:
        assert pool.fs is spy
        handle = await pool.open(path, "r")
        assert stat_calls == [os.fspath(path)]
        assert await handle.size() == 6


# --- AsyncFileSystem delegation -----------------------------------------------


async def test_fs_stat_returns_size_for_existing_file(
    pool_factory, file_writer
) -> None:
    path = file_writer("fs-stat.bin", b"abcdef")

    async with pool_factory() as pool:
        stat = await pool.fs.stat(path)
        assert stat.st_size == 6


async def test_fs_stat_raises_filenotfound_for_missing_path(
    pool_factory, tmp_path
) -> None:
    async with pool_factory() as pool:
        with pytest.raises(FileNotFoundError):
            await pool.fs.stat(tmp_path / "missing.bin")


async def test_fs_rename_moves_file_and_preserves_contents(
    pool_factory, file_writer, tmp_path
) -> None:
    old = file_writer("rename-old.bin", b"payload")
    new = tmp_path / "rename-new.bin"

    async with pool_factory() as pool:
        await pool.fs.rename(old, new)

        assert await pool.fs.exists(old) is False
        assert await pool.fs.exists(new) is True
        assert new.read_bytes() == b"payload"


async def test_fs_rename_raises_when_source_missing(pool_factory, tmp_path) -> None:
    async with pool_factory() as pool:
        with pytest.raises(FileNotFoundError):
            await pool.fs.rename(tmp_path / "missing.bin", tmp_path / "target.bin")


async def test_fs_exists_distinguishes_present_from_missing(
    pool_factory, file_writer, tmp_path
) -> None:
    path = file_writer("exists.bin", b"x")

    async with pool_factory() as pool:
        assert await pool.fs.exists(path) is True
        assert await pool.fs.exists(tmp_path) is True
        assert await pool.fs.exists(tmp_path / "missing.bin") is False


async def test_fs_is_file_and_is_dir_distinguish_kinds(
    pool_factory, file_writer, tmp_path
) -> None:
    path = file_writer("is-file.bin", b"x")

    async with pool_factory() as pool:
        assert await pool.fs.is_file(path) is True
        assert await pool.fs.is_dir(tmp_path) is True
        assert await pool.fs.is_file(tmp_path) is False
        assert await pool.fs.is_dir(path) is False
        assert await pool.fs.is_file(tmp_path / "missing.bin") is False


async def test_fs_move_relocates_file_across_directories(
    pool_factory, file_writer, tmp_path
) -> None:
    src = file_writer("move-src.bin", b"payload")
    target_dir = tmp_path / "subdir"
    target_dir.mkdir()
    dst = target_dir / "move-dst.bin"

    async with pool_factory() as pool:
        await pool.fs.move(src, dst)

        assert src.exists() is False
        assert dst.read_bytes() == b"payload"


async def test_fs_copyfile_duplicates_contents(
    pool_factory, file_writer, tmp_path
) -> None:
    src = file_writer("copy-src.bin", b"payload")
    dst = tmp_path / "copy-dst.bin"

    async with pool_factory() as pool:
        await pool.fs.copyfile(src, dst)

        assert src.read_bytes() == b"payload"
        assert dst.read_bytes() == b"payload"


async def test_fs_remove_deletes_file(pool_factory, file_writer) -> None:
    path = file_writer("remove.bin", b"x")

    async with pool_factory() as pool:
        await pool.fs.remove(path)
        assert path.exists() is False


async def test_fs_mkdir_creates_directory(pool_factory, tmp_path) -> None:
    target = tmp_path / "made"

    async with pool_factory() as pool:
        await pool.fs.mkdir(target)
        assert target.is_dir() is True


async def test_fs_mkdir_exists_ok_no_error_on_existing(pool_factory, tmp_path) -> None:
    target = tmp_path / "made"
    target.mkdir()

    async with pool_factory() as pool:
        # Without exist_ok, FileExistsError surfaces.
        with pytest.raises(FileExistsError):
            await pool.fs.mkdir(target)
        # With exist_ok=True, the call succeeds idempotently.
        await pool.fs.mkdir(target, exist_ok=True)


async def test_fs_mkdir_parents_creates_intermediate_directories(
    pool_factory, tmp_path
) -> None:
    target = tmp_path / "a" / "b" / "c"

    async with pool_factory() as pool:
        await pool.fs.mkdir(target, parents=True)
        assert target.is_dir() is True


async def test_fs_methods_dispatch_via_pool_run_blocking(
    pool_factory, file_writer, tmp_path, monkeypatch
) -> None:
    """Every fs method funnels through `pool._run_blocking` (so cancellation and
    executor lifecycle are managed centrally)."""
    path = file_writer("dispatch.bin", b"abc")
    target = tmp_path / "dispatch-renamed.bin"

    async with pool_factory() as pool:
        recorded: list[str] = []
        original_run_blocking = pool._run_blocking

        async def spy(func, *args):
            recorded.append(func.__name__)
            return await original_run_blocking(func, *args)

        monkeypatch.setattr(pool, "_run_blocking", spy)

        await pool.fs.stat(path)
        await pool.fs.exists(path)
        await pool.fs.is_file(path)
        await pool.fs.is_dir(path)
        await pool.fs.rename(path, target)

        assert recorded == [
            os.stat.__name__,
            os.path.exists.__name__,
            os.path.isfile.__name__,
            os.path.isdir.__name__,
            os.rename.__name__,
        ]


async def test_fs_move_and_copyfile_dispatch_via_run_blocking(
    pool_factory, file_writer, tmp_path, monkeypatch
) -> None:
    src = file_writer("dispatch-mc.bin", b"payload")
    moved = tmp_path / "moved.bin"
    copied = tmp_path / "copied.bin"

    async with pool_factory() as pool:
        recorded: list[str] = []
        original_run_blocking = pool._run_blocking

        async def spy(func, *args):
            recorded.append(func.__name__)
            return await original_run_blocking(func, *args)

        monkeypatch.setattr(pool, "_run_blocking", spy)

        await pool.fs.copyfile(src, copied)
        await pool.fs.move(src, moved)

        assert recorded == [shutil.copyfile.__name__, shutil.move.__name__]


@pytest.mark.parametrize("path_factory_value", ["path", "str", "bytes"])
async def test_fs_stat_accepts_str_path_and_bytes_inputs(
    pool_factory, file_writer, path_factory_value: str
) -> None:
    """The fs surface accepts the same path shapes the pool's `open()` accepts."""
    raw = file_writer("strpath-fs.bin", b"abcdef")
    coerced: str | bytes | Path
    coerced = (
        raw
        if path_factory_value == "path"
        else (str(raw) if path_factory_value == "str" else os.fsencode(raw))
    )

    async with pool_factory() as pool:
        stat = await pool.fs.stat(coerced)
        assert stat.st_size == 6
