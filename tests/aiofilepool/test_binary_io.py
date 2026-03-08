import io

import pytest

from aiofilepool.errors import IONotOpenError


pytestmark = pytest.mark.asyncio


async def test_adapter_context_manager_initializes_and_closes_without_closing_io(
    pool_factory,
) -> None:
    async with pool_factory() as pool:
        bio = io.BytesIO(b"context")

        async with pool.manage(bio) as adapter:
            assert await adapter.read(3) == b"con"

        assert bio.closed is False
        with pytest.raises(IONotOpenError):
            await adapter.read(1)


async def test_adapter_close_is_idempotent_and_does_not_close_underlying_io(
    pool_factory, binary_io_factory
) -> None:
    async with pool_factory() as pool:
        adapter, bio = binary_io_factory(pool, b"abc")
        adapter = await adapter

        await adapter.close()
        await adapter.close()

        assert bio.closed is False
        with pytest.raises(IONotOpenError):
            await adapter.read(1, 0)
        with pytest.raises(IONotOpenError):
            await adapter.write(b"x")


async def test_initialized_adapter_remains_usable_after_pool_close(
    pool_factory,
) -> None:
    pool = pool_factory()
    adapter = await pool.manage(io.BytesIO(b"abc"))

    await pool.close()

    assert await adapter.read(offset=0) == b"abc"
    assert await adapter.write(b"XY", offset=1) == 2
    assert await adapter.read(offset=0) == b"aXY"
