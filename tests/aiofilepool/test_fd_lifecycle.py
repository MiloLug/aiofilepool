import asyncio

import pytest

from aiofilepool.errors import FilePoolNotOpenError

from .conftest import AsyncGate


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
