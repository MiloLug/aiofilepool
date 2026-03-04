import os

import pytest

from aiofilepool.errors import (
    FileHandleInitializedError,
    FileHandleNotOpenError,
    InvalidFileModeError,
    InvalidFilePositionError,
)


pytestmark = pytest.mark.asyncio


async def test_handle_can_be_initialized_via_await(pool_factory, file_writer) -> None:
    path = file_writer("data.bin", b"hello")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r")
        assert await handle.tell() == 0
        assert await handle.read() == b"hello"


async def test_handle_context_manager_initializes_and_closes(
    pool_factory, file_writer
) -> None:
    path = file_writer("ctx.bin", b"context")

    async with pool_factory() as pool:
        async with pool.open(path, "r") as handle:
            assert await handle.read(3) == b"con"

        with pytest.raises(FileHandleNotOpenError):
            await handle.read(1)


async def test_double_initialization_raises(pool_factory, file_writer) -> None:
    path = file_writer("double-init.bin", b"abc")

    async with pool_factory() as pool:
        handle = pool.open(path, "r")
        await handle
        with pytest.raises(FileHandleInitializedError):
            await handle


async def test_handle_read_write_cursor_and_offsets(pool_factory, file_writer) -> None:
    path = file_writer("rw.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")

        assert await handle.write(b"abcdef") == 6
        assert await handle.tell() == 6

        assert await handle.seek(0) == 0
        assert await handle.read(3) == b"abc"
        assert await handle.tell() == 3

        assert await handle.write(b"XY") == 2
        assert await handle.tell() == 5
        assert await handle.read(offset=0) == b"abcXYf"


async def test_seek_whence_variants_and_invalid_positions(
    pool_factory, file_writer
) -> None:
    path = file_writer("seek.bin", b"abcdef")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r")

        assert await handle.seek(2, os.SEEK_SET) == 2
        assert await handle.seek(1, os.SEEK_CUR) == 3
        assert await handle.seek(-1, os.SEEK_END) == 5

        with pytest.raises(InvalidFilePositionError, match="invalid whence"):
            await handle.seek(0, 999)
        with pytest.raises(InvalidFilePositionError, match="invalid position"):
            await handle.seek(-1, os.SEEK_SET)
        with pytest.raises(InvalidFilePositionError, match="invalid position"):
            await handle.seek(1, os.SEEK_END)


async def test_truncate_defaults_to_current_position_and_accepts_explicit_size(
    pool_factory, file_writer
) -> None:
    path = file_writer("truncate.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.write(b"abcdef")

        await handle.seek(4)
        await handle.truncate()
        assert await handle.tell() == 4
        assert await handle.read(offset=0) == b"abcd"

        await handle.truncate(2)
        assert await handle.tell() == 2
        assert await handle.read(offset=0) == b"ab"


async def test_mode_errors_for_unreadable_and_unwritable_handles(
    pool_factory, file_writer
) -> None:
    path = file_writer("mode-errors.bin", b"abc")

    async with pool_factory() as pool:
        write_only = await pool.open(path, "w")
        with pytest.raises(InvalidFileModeError, match="not readable"):
            await write_only.read(1, 0)

        read_only = await pool.open(path, "r")
        with pytest.raises(InvalidFileModeError, match="not writable"):
            await read_only.write(b"x")


async def test_closed_handle_rejects_read_write_and_truncate(
    pool_factory, file_writer
) -> None:
    path = file_writer("closed.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.close()

        with pytest.raises(FileHandleNotOpenError):
            await handle.read(1, 0)
        with pytest.raises(FileHandleNotOpenError):
            await handle.write(b"x")
        with pytest.raises(FileHandleNotOpenError):
            await handle.truncate(0)


async def test_negative_offsets_and_sizes_raise_invalid_position(
    pool_factory, file_writer
) -> None:
    path = file_writer("positions.bin", b"abc")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r+")

        with pytest.raises(InvalidFilePositionError, match="offset must be >= 0"):
            await handle.read(1, -1)
        with pytest.raises(InvalidFilePositionError, match="size must be >= 0"):
            await handle.read(-1, 0)
        with pytest.raises(InvalidFilePositionError, match="offset must be >= 0"):
            await handle.write(b"x", -1)
        with pytest.raises(
            InvalidFilePositionError, match="truncate size must be >= 0"
        ):
            await handle.truncate(-1)
