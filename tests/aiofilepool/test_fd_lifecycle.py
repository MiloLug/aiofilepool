import pytest


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
