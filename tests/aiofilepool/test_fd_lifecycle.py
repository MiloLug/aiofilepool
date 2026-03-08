import asyncio
import io

import pytest

from aiofilepool.errors import FilePoolNotOpenError, IONotOpenError

from .conftest import AsyncGate, FailingIO


pytestmark = pytest.mark.asyncio


async def test_descriptor_pool_size_one_reuses_handles_without_data_loss(
    pool_factory, file_writer
) -> None:
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


async def test_reopening_inactive_handle_path_remains_read_write_capable(
    pool_factory, file_writer
) -> None:
    path_one = file_writer("reopen-one.bin", b"")
    path_two = file_writer("reopen-two.bin", b"")

    async with pool_factory(descriptor_pool_size=1, thread_pool_size=0) as pool:
        handle_one = await pool.open(path_one, "w+")
        handle_two = await pool.open(path_two, "w+")

        await handle_one.write(b"abc")
        await handle_two.write(b"xyz")

        await handle_one.seek(0)
        assert await handle_one.read() == b"abc"
        await handle_one.seek(3)
        assert await handle_one.write(b"d") == 1
        await handle_one.seek(0)
        assert await handle_one.read() == b"abcd"


async def test_closed_handle_descriptor_is_discarded_for_future_handles(
    pool_factory, file_writer
) -> None:
    path = file_writer("discard.bin", b"")

    async with pool_factory(descriptor_pool_size=1, thread_pool_size=0) as pool:
        first = await pool.open(path, "w+")
        await first.write(b"persisted")
        await first.close()

        second = await pool.open(path, "r")
        assert await second.read() == b"persisted"


async def test_waiting_handle_resumes_after_active_handle_releases_descriptor(
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
        assert second_started.is_set() is False

        gate.release()

        assert await first_task == 3
        assert await second_task == 3
        assert second_started.is_set() is True

    assert path_one.read_bytes() == b"one"
    assert path_two.read_bytes() == b"two"


async def test_pool_close_unblocks_waiting_handle_with_not_open_error(
    stressed_pool_factory, file_writer, monkeypatch
) -> None:
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


async def test_same_handle_operations_are_serialized_until_current_op_finishes(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("same-handle-serialization.bin", b"")
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


async def test_handle_close_waits_for_inflight_operation_before_closing(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("handle-close-race.bin", b"")
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


async def test_adapter_close_can_finish_while_write_is_in_flight(
    pool_factory, monkeypatch
) -> None:
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


async def test_cold_descriptor_eviction_failure_does_not_block_future_operations(
    pool_factory, file_writer, monkeypatch
) -> None:
    path_one = file_writer("evict-one.bin", b"")
    path_two = file_writer("evict-two.bin", b"")
    bad_io = FailingIO(fail_on={"flush": 1})
    good_io = FailingIO()
    opened_ios = iter([bad_io, good_io])

    monkeypatch.setattr(
        "aiofilepool._fd_manager.open",
        lambda path, mode: next(opened_ios),
        raising=False,
    )

    async with pool_factory(descriptor_pool_size=1, thread_pool_size=0) as pool:
        handle_one = await pool.open(path_one, "w+")
        handle_two = await pool.open(path_two, "w+")

        assert await handle_one.write(b"one") == 3
        with pytest.raises(RuntimeError, match="flush failed"):
            await handle_two.write(b"two")

        assert await asyncio.wait_for(handle_two.write(b"two"), timeout=0.2) == 3


async def test_handle_close_can_be_retried_after_discard_failure(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("discard-failure.bin", b"")
    failing_io = FailingIO(fail_on={"flush": 1})

    monkeypatch.setattr(
        "aiofilepool._fd_manager.open",
        lambda path, mode: failing_io,
        raising=False,
    )

    async with pool_factory(thread_pool_size=0) as pool:
        handle = await pool.open(path, "w+")
        await handle.write(b"abc")

        with pytest.raises(RuntimeError, match="flush failed"):
            await handle.close()

        await handle.close()

        with pytest.raises(IONotOpenError):
            await handle.write(b"x")
