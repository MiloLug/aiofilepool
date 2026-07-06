"""`FileDescriptorManager` mechanics under capacity pressure.

Pins the slot-accounting and hot/cold/inactive transitions of the descriptor
manager:

* round-robin under `cap=1`: two handles share one slot without data loss
* waiters on `_ensure_slot` wake on release AND on pool close (with
  `FilePoolNotOpenError`)
* failed cold-descriptor flush during eviction surfaces the error but still
  releases the slot (no deadlock for future waiters)
* failed `open()` during reopen (activation from inactive) releases the slot
  so a different handle can still acquire
* discarding a cooled handle takes the `cold_handles` branch (slot is not
  re-released, because cooling already released it)
* `close()` drains cold descriptors and then waits for `_descriptors` empty
* `close()` surfaces the first cleanup error while still draining the rest
* a handle's `close()` waits for an inflight op before the descriptor is freed
"""

import asyncio
import io

import pytest

from aiofilepool.errors import FilePoolNotOpenError, IONotOpenError

from .conftest import AsyncGate, FailingIO, patch_descriptor_open


pytestmark = pytest.mark.asyncio


# --- Round-robin under cap=1 --------------------------------------------------


async def test_two_handles_round_robin_under_cap_of_one(
    pool_factory, file_writer
) -> None:
    """`cap=1` forces the manager to swap fds between two handles; bytes on both files
    must reflect every write."""
    path_one = file_writer("one.bin", b"")
    path_two = file_writer("two.bin", b"")

    async with pool_factory(descriptor_pool_size=1, thread_pool_size=0) as pool:
        handle_one = await pool.open(path_one, "w+")
        handle_two = await pool.open(path_two, "w+")

        await handle_one.write(b"one")
        await handle_two.write(b"two")

        await handle_one.seek(0)
        assert await handle_one.read() == b"one"

        await handle_one.seek(3)
        await handle_one.write(b"!")
        await handle_one.seek(0)
        assert await handle_one.read() == b"one!"

        await handle_two.seek(0)
        assert await handle_two.read() == b"two"


async def test_reopened_inactive_handle_remains_read_write_capable(
    pool_factory, file_writer
) -> None:
    """A handle whose descriptor was evicted and later reopened via `renewal_mode`
    must regain full read/write access (no silent demotion to read-only)."""
    path_one = file_writer("reopen-one.bin", b"")
    path_two = file_writer("reopen-two.bin", b"")

    async with pool_factory(descriptor_pool_size=1, thread_pool_size=0) as pool:
        handle_one = await pool.open(path_one, "w+")
        handle_two = await pool.open(path_two, "w+")

        await handle_one.write(b"abc")
        await handle_two.write(b"xyz")

        # Reactivate handle_one — evicts handle_two's fd.
        await handle_one.seek(0)
        assert await handle_one.read() == b"abc"
        await handle_one.seek(3)
        assert await handle_one.write(b"d") == 1
        await handle_one.seek(0)
        assert await handle_one.read() == b"abcd"


async def test_closed_handle_descriptor_is_discarded_for_future_handles(
    pool_factory, file_writer
) -> None:
    """After a handle is closed, its descriptor is fully discarded — a new handle
    on the same path opens a fresh descriptor and sees the persisted bytes."""
    path = file_writer("discard.bin", b"")

    async with pool_factory(descriptor_pool_size=1, thread_pool_size=0) as pool:
        first = await pool.open(path, "w+")
        await first.write(b"persisted")
        await first.close()

        second = await pool.open(path, "r")
        assert await second.read() == b"persisted"


# --- Waiters: wake on release AND on pool close -------------------------------


async def test_waiter_resumes_after_active_handle_releases_descriptor(
    stressed_pool_factory, file_writer, monkeypatch
) -> None:
    path_one = file_writer("wait-one.bin", b"")
    path_two = file_writer("wait-two.bin", b"")
    gate = AsyncGate()
    second_started = asyncio.Event()

    async with stressed_pool_factory() as pool:
        handle_one = await pool.open(path_one, "w+")
        handle_two = await pool.open(path_two, "w+")
        original_run_blocking = pool._run_blocking
        call_count = 0

        async def gated_run_blocking(func, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await gate.wait()
            else:
                second_started.set()
            return await original_run_blocking(func, *args)

        monkeypatch.setattr(pool, "_run_blocking", gated_run_blocking)

        first_task = asyncio.create_task(handle_one.write(b"one"))
        await gate.entered.wait()

        second_task = asyncio.create_task(handle_two.write(b"two"))
        await asyncio.sleep(0)
        # Second task is blocked waiting for a slot until the first finishes.
        assert second_started.is_set() is False

        gate.release()

        assert await first_task == 3
        assert await second_task == 3
        assert second_started.is_set() is True

    assert path_one.read_bytes() == b"one"
    assert path_two.read_bytes() == b"two"


async def test_pool_close_unblocks_waiter_with_not_open_error(
    stressed_pool_factory, file_writer, monkeypatch
) -> None:
    """A task waiting for a slot must wake with `FilePoolNotOpenError` when the
    pool transitions to closed."""
    path_one = file_writer("close-waits-one.bin", b"")
    path_two = file_writer("close-waits-two.bin", b"")
    gate = AsyncGate()

    async with stressed_pool_factory() as pool:
        handle_one = await pool.open(path_one, "w+")
        handle_two = await pool.open(path_two, "w+")
        original_run_blocking = pool._run_blocking
        call_count = 0

        async def gated_run_blocking(func, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await gate.wait()
            return await original_run_blocking(func, *args)

        monkeypatch.setattr(pool, "_run_blocking", gated_run_blocking)

        first_task = asyncio.create_task(handle_one.write(b"one"))
        await gate.entered.wait()

        second_task = asyncio.create_task(handle_two.write(b"two"))
        await asyncio.sleep(0)

        close_task = asyncio.create_task(pool.close())

        with pytest.raises(FilePoolNotOpenError):
            await second_task

        gate.release()

        assert await first_task == 3
        await close_task
        assert path_one.read_bytes() == b"one"


# --- Per-handle op serialization ----------------------------------------------


async def test_handle_op_lock_serializes_same_handle_operations(
    pool_factory, file_writer, monkeypatch
) -> None:
    """`_op_lock` ensures that two writes on the same handle execute strictly serially."""
    path = file_writer("serialize.bin", b"")
    gate = AsyncGate()
    second_started = asyncio.Event()

    async with pool_factory(thread_pool_size=0) as pool:
        handle = await pool.open(path, "w+")
        original_run_blocking = pool._run_blocking
        call_count = 0

        async def gated_run_blocking(func, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await gate.wait()
            else:
                second_started.set()
            return await original_run_blocking(func, *args)

        monkeypatch.setattr(pool, "_run_blocking", gated_run_blocking)

        first_task = asyncio.create_task(handle.write(b"one"))
        await gate.entered.wait()

        second_task = asyncio.create_task(handle.write(b"two"))
        await asyncio.sleep(0)
        assert second_started.is_set() is False

        gate.release()

        assert await first_task == 3
        assert await second_task == 3
        assert await handle.read(offset=0) == b"onetwo"


async def test_cancelling_handle_write_leaves_handle_usable(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("cancel-handle-write.bin", b"")
    gate = AsyncGate()

    async with pool_factory(thread_pool_size=1) as pool:
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

        write_task.cancel()
        await asyncio.sleep(0)
        gate.release()

        with pytest.raises(asyncio.CancelledError):
            await write_task

        # The write was shielded — the bytes did land — but the handle remains usable
        # for a subsequent overwrite at offset=0.
        assert await handle.write(b"xyz", offset=0) == 3
        assert await handle.read(offset=0) == b"xyz"


async def test_cancelling_adapter_write_leaves_adapter_usable(
    pool_factory, monkeypatch
) -> None:
    gate = AsyncGate()

    async with pool_factory(thread_pool_size=1) as pool:
        adapter = await pool.manage(io.BytesIO())
        original_run_blocking = pool._run_blocking
        call_count = 0

        async def gated_run_blocking(func, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await gate.wait()
            return await original_run_blocking(func, *args)

        monkeypatch.setattr(pool, "_run_blocking", gated_run_blocking)

        write_task = asyncio.create_task(adapter.write(b"abc"))
        await gate.entered.wait()

        write_task.cancel()
        await asyncio.sleep(0)
        gate.release()

        with pytest.raises(asyncio.CancelledError):
            await write_task

        assert await adapter.write(b"xyz", offset=0) == 3
        assert await adapter.read(offset=0) == b"xyz"


async def test_handle_close_waits_for_inflight_operation_before_freeing_descriptor(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("close-vs-write.bin", b"")
    gate = AsyncGate()

    async with pool_factory(thread_pool_size=1) as pool:
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

        close_task = asyncio.create_task(handle.close())
        await asyncio.sleep(0)
        assert close_task.done() is False

        gate.release()

        assert await write_task == 3
        await close_task

        with pytest.raises(IONotOpenError):
            await handle.read(1, 0)

    assert path.read_bytes() == b"abc"


async def test_adapter_close_completes_while_write_inflight(
    pool_factory, monkeypatch
) -> None:
    """Adapters don't hold a pool descriptor — `close()` need not wait for an inflight
    write to release a resource. But the write is allowed to finish."""
    gate = AsyncGate()
    backing = io.BytesIO()

    async with pool_factory(thread_pool_size=1) as pool:
        adapter = await pool.manage(backing)
        original_run_blocking = pool._run_blocking
        call_count = 0

        async def gated_run_blocking(func, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await gate.wait()
            return await original_run_blocking(func, *args)

        monkeypatch.setattr(pool, "_run_blocking", gated_run_blocking)

        write_task = asyncio.create_task(adapter.write(b"abc"))
        await gate.entered.wait()

        await adapter.close()
        gate.release()

        assert await write_task == 3

        with pytest.raises(IONotOpenError):
            await adapter.read(1, 0)

    assert backing.getvalue() == b"abc"


# --- Eviction failure paths ---------------------------------------------------


async def test_cold_eviction_failure_does_not_deadlock_future_operations(
    pool_factory, file_writer, monkeypatch
) -> None:
    path_one = file_writer("evict-one.bin", b"")
    path_two = file_writer("evict-two.bin", b"")
    bad_io = FailingIO(fail_on={"flush": 1})
    good_io = FailingIO()
    opened_ios = iter([bad_io, good_io])

    patch_descriptor_open(monkeypatch, lambda path, mode: next(opened_ios))

    async with pool_factory(descriptor_pool_size=1, thread_pool_size=0) as pool:
        handle_one = await pool.open(path_one, "w+")
        handle_two = await pool.open(path_two, "w+")

        assert await handle_one.write(b"one") == 3
        with pytest.raises(RuntimeError, match="flush failed"):
            await handle_two.write(b"two")

        # Retry must succeed — proves slot was released even on the failure path.
        assert await asyncio.wait_for(handle_two.write(b"two"), timeout=0.2) == 3


async def test_failed_open_on_reopen_releases_slot(
    pool_factory, file_writer, monkeypatch
) -> None:
    """When reopening an evicted handle fails (the renewal `open()` raises), the slot
    must be released so a sibling handle can still acquire."""
    path_a = file_writer("reopen-fail-a.bin", b"a")
    path_b = file_writer("reopen-fail-b.bin", b"b")

    async with pool_factory(descriptor_pool_size=1, thread_pool_size=0) as pool:
        handle_a = await pool.open(path_a, "r+b")
        handle_b = await pool.open(path_b, "r+b")

        # Touch a to bring it active, then touch b to make a inactive.
        await handle_a.read(offset=0)
        await handle_b.read(offset=0)
        # Now a is evicted (materialized but no live descriptor), b is in _cold_handles.
        assert handle_a not in pool._fd_manager._descriptors
        assert handle_a._fd_materialized
        assert handle_b in pool._fd_manager._cold_handles

        # Inject failure on the next open() (which will be a's renewal).
        original_open = open
        call_count = {"n": 0}

        def failing_open(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("injected reopen failure")
            return original_open(*args, **kwargs)

        patch_descriptor_open(monkeypatch, failing_open)

        with pytest.raises(OSError, match="injected reopen failure"):
            await handle_a.read(offset=0)

        # If the slot leaked, this would deadlock under wait_for.
        assert await asyncio.wait_for(handle_b.read(offset=0), timeout=2.0) == b"b"


async def test_discarding_cold_handle_does_not_re_release_slot(
    pool_factory, file_writer
) -> None:
    """A handle that was cooled has already released its slot; discarding it must
    not release a second time (which would corrupt the slot count)."""
    path_a = file_writer("cold-discard-a.bin", b"")
    path_b = file_writer("cold-discard-b.bin", b"")

    async with pool_factory(descriptor_pool_size=1, thread_pool_size=0) as pool:
        handle_a = await pool.open(path_a, "w+")
        await handle_a.write(b"a-data")

        assert handle_a in pool._fd_manager._cold_handles
        slots_before = pool._fd_manager._slots

        await handle_a.close()

        # Slots count is unchanged (the cooled state had already counted as available).
        assert pool._fd_manager._slots == slots_before
        assert handle_a not in pool._fd_manager._cold_handles
        assert handle_a not in pool._fd_manager._descriptors

        # A subsequent acquire still works and slot count stays bounded.
        handle_b = await pool.open(path_b, "w+")
        await handle_b.write(b"b-data")
        assert await handle_b.read(offset=0) == b"b-data"
        assert pool._fd_manager._slots <= 1


async def test_handle_close_after_discard_failure_can_be_retried(
    pool_factory, file_writer, monkeypatch
) -> None:
    """If a handle's underlying flush fails on `close()`, the error surfaces, but a
    retry succeeds (the descriptor was popped before the flush attempt)."""
    path = file_writer("discard-fail.bin", b"")
    failing_io = FailingIO(fail_on={"flush": 1})

    patch_descriptor_open(monkeypatch, lambda path, mode: failing_io)

    async with pool_factory(thread_pool_size=0) as pool:
        handle = await pool.open(path, "w+")
        await handle.write(b"abc")

        with pytest.raises(RuntimeError, match="flush failed"):
            await handle.close()

        # Retry close → idempotent return (state already CLOSED).
        await handle.close()

        with pytest.raises(IONotOpenError):
            await handle.write(b"x")
