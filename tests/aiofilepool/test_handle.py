from pathlib import Path

import pytest

from aiofilepool import FileHandle, ModeSpec, StrPath
from aiofilepool.errors import (
    IONotOpenError,
    FilePoolNotOpenError,
    InvalidFileModeError,
    InvalidPositionError,
)


pytestmark = pytest.mark.asyncio


def _as_path(path: Path) -> StrPath:
    return path


def _as_str(path: Path) -> StrPath:
    return str(path)


@pytest.mark.parametrize(
    "path_factory",
    [
        pytest.param(_as_path, id="path"),
        pytest.param(_as_str, id="str"),
    ],
)
async def test_direct_filehandle_constructor_accepts_strpath_for_read_mode(
    pool_factory, file_writer, path_factory
) -> None:
    path = file_writer("direct-read-handle.bin", b"hello")

    async with pool_factory() as pool:
        handle = FileHandle(pool, path_factory(path), ModeSpec.from_str("r"))

        await handle
        assert await handle.read() == b"hello"


@pytest.mark.parametrize(
    "path_factory",
    [
        pytest.param(_as_path, id="path"),
        pytest.param(_as_str, id="str"),
    ],
)
async def test_direct_filehandle_constructor_normalizes_truncate_modes(
    pool_factory, file_writer, path_factory
) -> None:
    path = file_writer("direct-write-handle.bin", b"discarded")

    async with pool_factory(thread_pool_size=0) as pool:
        handle = await FileHandle(pool, path_factory(path), ModeSpec.from_str("w+"))

        assert await handle.size() == 0
        assert await handle.write(b"abc") == 3
        assert await handle.read(offset=0) == b"abc"


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


async def test_truncate_rejects_negative_size(pool_factory, file_writer) -> None:
    path = file_writer("truncate-negative.bin", b"abc")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r+")

        with pytest.raises(InvalidPositionError, match="truncate size must be >= 0"):
            await handle.truncate(-1)


async def test_handle_truncate_can_grow_file_with_zero_padding(
    pool_factory, file_writer
) -> None:
    path = file_writer("truncate-grow.bin", b"ab")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r+")
        await handle.truncate(5)

        assert await handle.tell() == 5
        assert await handle.size() == 5
        assert await handle.read(offset=0) == b"ab\x00\x00\x00"


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

        with pytest.raises(IONotOpenError):
            await handle.read(1, 0)
        with pytest.raises(IONotOpenError):
            await handle.write(b"x")
        with pytest.raises(IONotOpenError):
            await handle.truncate(0)


async def test_open_handle_operations_fail_after_pool_close(
    pool_factory, file_writer
) -> None:
    path = file_writer("pool-close-open-handle.bin", b"")
    pool = pool_factory()
    handle = await pool.open(path, "w+")

    await pool.close()

    with pytest.raises(FilePoolNotOpenError):
        await handle.write(b"x")
    with pytest.raises(FilePoolNotOpenError):
        await handle.read(1, 0)


async def test_handle_chunks_write_only_handle_raises_invalid_mode(
    pool_factory, file_writer
) -> None:
    path = file_writer("chunks-write-only.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w")
        with pytest.raises(InvalidFileModeError, match="file is not readable"):
            [chunk async for chunk in handle.chunks()]


async def test_missing_path_read_handle_raises_during_initialization(
    pool_factory, tmp_path
) -> None:
    missing_path = tmp_path / "missing-read-handle.bin"

    async with pool_factory() as pool:
        handle = pool.open(missing_path, "r")
        with pytest.raises(FileNotFoundError):
            await handle
