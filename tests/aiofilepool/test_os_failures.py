"""Real OS-failure paths through the pool.

Renamed from `test_fault_injection.py`. The shared `OSErrorIO` test double now
lives in `conftest.py`. This module pins behavior under failures that production
actually encounters:

* `OSError(ENOSPC)` during `write()` — partial commit, no leaked fd
* `PermissionError` during a renewal `open()` — slot released, sibling progresses
* `FileNotFoundError` during initial open-for-read — propagates from `__await__`,
  pool stays usable
* `OSError(EIO)` on one concurrent open — does not deadlock the other
* `OSError(ENOSPC)` from `posix_fallocate` — `_size` unchanged, retry succeeds
* `CancelledError` mid-`allocate` — pool consistent, slot returned, handle reusable
* `ThreadPoolExecutor` pre-shutdown — clean error, `pool.close()` still terminates
"""

import asyncio
import errno
import os

import pytest

from aiofilepool import FilePool

from .conftest import OSErrorIO


pytestmark = pytest.mark.asyncio


# --- Write-time OS errors -----------------------------------------------------


async def test_enospc_during_write_propagates_without_leaking_descriptor() -> None:
    """ENOSPC during a write surfaces the OSError and leaves descriptor state consistent."""
    failing = OSErrorIO(
        b"",
        fail_on="write",
        on_call=1,
        exc_factory=lambda: OSError(errno.ENOSPC, "No space left on device"),
    )

    async with FilePool(
        descriptor_pool_size=2, thread_pool_size=0, chunking_threshold=1024
    ) as pool:
        adapter = await pool.manage(failing)
        with pytest.raises(OSError) as exc_info:
            await adapter.write(b"data")
        assert exc_info.value.errno == errno.ENOSPC
        # Adapter does not own an fd in the pool.
        assert len(pool._fd_manager._descriptors) == 0
        await adapter.close()


async def test_enospc_chunked_write_commits_prefix(tmp_path) -> None:
    """A chunked write that fails mid-stream commits the bytes written before the
    failure; `_size` and `_position` reflect that prefix."""
    failing = OSErrorIO(
        b"",
        fail_on="write",
        on_call=2,
        exc_factory=lambda: OSError(errno.ENOSPC, "ENOSPC"),
    )

    async with FilePool(
        descriptor_pool_size=2, thread_pool_size=0, chunking_threshold=4
    ) as pool:
        adapter = await pool.manage(failing)

        from aiofilepool._chunking import FixedChunker

        pool._chunker = FixedChunker(2)  # type: ignore[assignment]

        with pytest.raises(OSError):
            await adapter.write(b"abcdef")

        # First 2 bytes committed (one chunk write succeeded; the second raised).
        assert await adapter.size() == 2
        assert await adapter.tell() == 2
        assert failing.getvalue() == b"ab"


# --- Reopen / activation failures ---------------------------------------------


async def test_permission_error_during_reopen_releases_slot(
    tmp_path, monkeypatch
) -> None:
    """`PermissionError` on the renewal `open()` must still release the slot
    so a sibling handle's acquire completes."""
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    path_a.write_bytes(b"")
    path_b.write_bytes(b"")

    async with FilePool(
        descriptor_pool_size=1, thread_pool_size=0, chunking_threshold=1024
    ) as pool:
        handle_a = await pool.open(path_a, "r+b")
        handle_b = await pool.open(path_b, "r+b")
        await handle_a.write(b"alpha")

        original_open = open
        call_count = {"n": 0}

        def _failing_open(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise PermissionError(
                    errno.EACCES, "Permission denied (injected)", str(args[0])
                )
            return original_open(*args, **kwargs)

        monkeypatch.setattr(
            "aiofilepool._fd_manager.open", _failing_open, raising=False
        )

        with pytest.raises(PermissionError):
            await handle_b.write(b"beta")

        # Subsequent op on handle_a succeeds — proves no leaked slot.
        monkeypatch.setattr(
            "aiofilepool._fd_manager.open", original_open, raising=False
        )
        await handle_a.write(b"second-write")
        await handle_a.close()
        await handle_b.close()


async def test_filenotfound_on_initial_open_propagates_and_pool_remains_usable(
    tmp_path,
) -> None:
    missing = tmp_path / "does-not-exist.bin"

    async with FilePool(
        descriptor_pool_size=2, thread_pool_size=0, chunking_threshold=1024
    ) as pool:
        with pytest.raises(FileNotFoundError):
            await pool.open(missing, "rb")

        existing = tmp_path / "existing.bin"
        existing.write_bytes(b"hello")
        handle = await pool.open(existing, "rb")
        assert await handle.read() == b"hello"
        await handle.close()


async def test_concurrent_acquire_after_failed_open_does_not_deadlock(
    tmp_path, monkeypatch
) -> None:
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    path_a.write_bytes(b"alpha")
    path_b.write_bytes(b"beta")

    async with FilePool(
        descriptor_pool_size=1, thread_pool_size=0, chunking_threshold=1024
    ) as pool:
        original_open = open
        call_count = {"n": 0}

        def _flaky_open(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError(errno.EIO, "I/O error (injected)", str(args[0]))
            return original_open(*args, **kwargs)

        monkeypatch.setattr("aiofilepool._fd_manager.open", _flaky_open, raising=False)

        async def _try_open_a():
            handle = await pool.open(path_a, "rb")
            await handle.read()
            await handle.close()

        async def _try_open_b():
            handle = await pool.open(path_b, "rb")
            data = await handle.read()
            await handle.close()
            return data

        with pytest.raises(OSError):
            await _try_open_a()

        result = await asyncio.wait_for(_try_open_b(), timeout=2.0)
        assert result == b"beta"


# --- allocate() failure paths -------------------------------------------------


async def test_posix_fallocate_enospc_does_not_advance_size(
    tmp_path, monkeypatch
) -> None:
    """If `posix_fallocate` raises ENOSPC, the handle's `_size` must NOT be advanced
    (it's set only after `_run_blocking` returns successfully), and the slot must
    be released so a smaller retry succeeds."""
    monkeypatch.setattr("aiofilepool._handle._IS_POSIX", True)

    path = tmp_path / "alloc-fail.bin"
    path.write_bytes(b"")

    call_count = {"n": 0}
    if not hasattr(os, "posix_fallocate"):
        # Provide a stub on platforms that lack it (Windows test runners).
        def _real_posix_fallocate(fd, offset, length):
            os.ftruncate(fd, length)

        monkeypatch.setattr(os, "posix_fallocate", _real_posix_fallocate, raising=False)
    real_posix_fallocate = os.posix_fallocate  # type: ignore[attr-defined]

    def _failing_fallocate(fd, offset, length):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError(errno.ENOSPC, "ENOSPC (injected)")
        return real_posix_fallocate(fd, offset, length)

    monkeypatch.setattr(os, "posix_fallocate", _failing_fallocate, raising=False)

    async with FilePool(
        descriptor_pool_size=2, thread_pool_size=0, chunking_threshold=1024
    ) as pool:
        handle = await pool.open(path, "w+b")

        with pytest.raises(OSError):
            await handle.allocate(1_000_000)

        # _size not advanced past actual file size.
        assert await handle.size() == 0

        # Retry with a smaller allocate succeeds (slot was returned).
        await handle.allocate(64)
        assert await handle.size() == 64
        await handle.close()


async def test_cancellation_mid_allocate_keeps_pool_consistent(
    tmp_path, monkeypatch
) -> None:
    """Cancelling an `allocate()` mid-flight must leave the pool consistent and the
    handle usable for a retry."""
    monkeypatch.setattr("aiofilepool._handle._IS_POSIX", False)

    path = tmp_path / "alloc-cancel.bin"
    path.write_bytes(b"")

    async with FilePool(
        descriptor_pool_size=2, thread_pool_size=1, chunking_threshold=1024
    ) as pool:
        handle = await pool.open(path, "w+b")

        gate = asyncio.Event()
        original_run_blocking = pool._run_blocking

        async def gated_run_blocking(func, *args):
            if func is os.truncate or getattr(func, "__name__", "") in {
                "posix_fallocate",
                "truncate",
            }:
                await gate.wait()
            return await original_run_blocking(func, *args)

        monkeypatch.setattr(pool, "_run_blocking", gated_run_blocking)

        alloc_task = asyncio.create_task(handle.allocate(1024))
        await asyncio.sleep(0)
        alloc_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await alloc_task

        gate.set()
        # Restore plain dispatch for the retry.
        monkeypatch.setattr(pool, "_run_blocking", original_run_blocking)

        # Pool must still be usable: retry should succeed.
        await handle.allocate(128)
        assert await handle.size() == 128
        await handle.close()


# --- Executor shutdown surfaces a clean error ---------------------------------


async def test_executor_pre_shutdown_surfaces_clean_error_and_close_still_terminates(
    tmp_path,
) -> None:
    """Forcing the executor down before an op surfaces a `RuntimeError` from
    `_run_blocking`. The pool must remain shutdownable (no hang)."""
    path = tmp_path / "exec-down.bin"
    path.write_bytes(b"")

    pool = FilePool(descriptor_pool_size=2, thread_pool_size=2, chunking_threshold=1024)
    handle = await pool.open(path, "w+b")

    assert pool._executor is not None
    pool._executor.shutdown(wait=True)

    with pytest.raises(RuntimeError):
        await handle.write(b"abc")

    # close() still terminates within a reasonable time.
    await asyncio.wait_for(pool.close(), timeout=5.0)
