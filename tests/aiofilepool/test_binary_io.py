import io
import os

import pytest

from aiofilepool import BinaryIOAdapter
from aiofilepool.errors import IOInitializedError, IONotOpenError, InvalidPositionError

from .conftest import RecordingChunker


pytestmark = pytest.mark.asyncio


async def test_manage_returns_binary_io_adapter(pool_factory) -> None:
    async with pool_factory() as pool:
        adapter = pool.manage(io.BytesIO())
        assert isinstance(adapter, BinaryIOAdapter)


async def test_adapter_initialization_detects_existing_size(
    pool_factory, binary_io_factory
) -> None:
    async with pool_factory() as pool:
        adapter, _ = binary_io_factory(pool, b"hello")
        adapter = await adapter

        assert await adapter.tell() == 0
        assert await adapter.read() == b"hello"


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


async def test_adapter_double_initialization_raises(
    pool_factory, binary_io_factory
) -> None:
    async with pool_factory() as pool:
        adapter, _ = binary_io_factory(pool, b"abc")
        await adapter

        with pytest.raises(IOInitializedError):
            await adapter


async def test_adapter_read_write_cursor_and_offsets(
    pool_factory, binary_io_factory
) -> None:
    async with pool_factory() as pool:
        adapter, _ = binary_io_factory(pool)
        adapter = await adapter

        assert await adapter.write(b"abcdef") == 6
        assert await adapter.tell() == 6

        assert await adapter.seek(0) == 0
        assert await adapter.read(3) == b"abc"
        assert await adapter.tell() == 3

        assert await adapter.write(b"XY") == 2
        assert await adapter.tell() == 5
        assert await adapter.read(offset=0) == b"abcXYf"


async def test_adapter_seek_whence_variants_and_invalid_positions(
    pool_factory, binary_io_factory
) -> None:
    async with pool_factory() as pool:
        adapter, _ = binary_io_factory(pool, b"abcdef")
        adapter = await adapter

        assert await adapter.seek(2, os.SEEK_SET) == 2
        assert await adapter.seek(1, os.SEEK_CUR) == 3
        assert await adapter.seek(-1, os.SEEK_END) == 5

        with pytest.raises(InvalidPositionError, match="invalid whence"):
            await adapter.seek(0, 999)
        with pytest.raises(InvalidPositionError, match="invalid position"):
            await adapter.seek(-1, os.SEEK_SET)
        with pytest.raises(InvalidPositionError, match="invalid position"):
            await adapter.seek(1, os.SEEK_END)


async def test_adapter_negative_offsets_and_sizes_raise_invalid_position(
    pool_factory, binary_io_factory
) -> None:
    async with pool_factory() as pool:
        adapter, _ = binary_io_factory(pool, b"abc")
        adapter = await adapter

        with pytest.raises(InvalidPositionError, match="offset must be >= 0"):
            await adapter.read(1, -1)
        with pytest.raises(InvalidPositionError, match="size must be >= 0"):
            await adapter.read(-1, 0)
        with pytest.raises(InvalidPositionError, match="offset must be >= 0"):
            await adapter.write(b"x", -1)


async def test_adapter_chunking_path_is_used_for_large_read_and_write(
    pool_factory, binary_io_factory
) -> None:
    payload = b"abcdefghijk"
    chunker = RecordingChunker([3, 3, 5])

    async with pool_factory(chunking_threshold=4, chunker=chunker) as pool:
        adapter, _ = binary_io_factory(pool)
        adapter = await adapter

        assert await adapter.write(payload) == len(payload)
        assert await adapter.read(offset=0) == payload

    assert chunker.calls == [len(payload), len(payload)]


@pytest.mark.parametrize("thread_pool_size", [0, 2])
async def test_adapter_roundtrip_in_threaded_and_threadless_modes(
    thread_pool_size: int, pool_factory, binary_io_factory
) -> None:
    async with pool_factory(
        thread_pool_size=thread_pool_size, chunking_threshold=8
    ) as pool:
        adapter, _ = binary_io_factory(pool)
        adapter = await adapter
        payload = b"thread-mode"

        assert await adapter.write(payload) == len(payload)
        assert await adapter.read(offset=0) == payload


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
