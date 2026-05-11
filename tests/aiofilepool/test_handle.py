import os
from pathlib import Path

import pytest

from aiofilepool import FileHandle, ModeSpec, StrOrBytesPath
from aiofilepool._fs import AsyncFileSystem
from aiofilepool.errors import (
    IONotOpenError,
    FilePoolNotOpenError,
    InvalidFileModeError,
    InvalidPositionError,
)


pytestmark = pytest.mark.asyncio


class _PathBytes:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def __fspath__(self) -> bytes:
        return self._raw


def _as_path(path: Path) -> StrOrBytesPath:
    return path


def _as_str(path: Path) -> StrOrBytesPath:
    return str(path)


def _as_bytes(path: Path) -> StrOrBytesPath:
    return os.fsencode(path)


def _as_path_bytes(path: Path) -> StrOrBytesPath:
    return _PathBytes(os.fsencode(path))


_PATH_FACTORIES = [
    pytest.param(_as_path, id="path"),
    pytest.param(_as_str, id="str"),
    pytest.param(_as_bytes, id="bytes"),
    pytest.param(_as_path_bytes, id="path-bytes"),
]


@pytest.mark.parametrize("path_factory", _PATH_FACTORIES)
async def test_direct_filehandle_constructor_accepts_strpath_for_read_mode(
    pool_factory, file_writer, path_factory
) -> None:
    path = file_writer("direct-read-handle.bin", b"hello")

    async with pool_factory() as pool:
        handle = FileHandle(pool, path_factory(path), ModeSpec.from_str("r"))

        await handle
        assert await handle.read() == b"hello"


@pytest.mark.parametrize("path_factory", _PATH_FACTORIES)
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


def _ensure_posix_fallocate(monkeypatch) -> None:
    if not hasattr(os, "posix_fallocate"):

        def posix_fallocate(fd, offset, length):
            os.ftruncate(fd, length)

        monkeypatch.setattr(os, "posix_fallocate", posix_fallocate, raising=False)


async def test_allocate_grows_empty_file_posix_branch(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("allocate-posix.bin", b"")
    monkeypatch.setattr("aiofilepool._handle._IS_POSIX", True)
    _ensure_posix_fallocate(monkeypatch)

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.allocate(1024)
        await handle.close()

    assert path.stat().st_size == 1024


async def test_allocate_grows_empty_file_windows_branch(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("allocate-win.bin", b"")
    monkeypatch.setattr("aiofilepool._handle._IS_POSIX", False)

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.allocate(2048)
        await handle.close()

    assert path.stat().st_size == 2048


async def test_allocate_posix_branch_calls_posix_fallocate(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("allocate-spy-posix.bin", b"")
    monkeypatch.setattr("aiofilepool._handle._IS_POSIX", True)
    _ensure_posix_fallocate(monkeypatch)

    async with pool_factory() as pool:
        recorded: list[str] = []
        original_run_blocking = pool._run_blocking

        async def spy(func, *args):
            recorded.append(func.__name__)
            return await original_run_blocking(func, *args)

        monkeypatch.setattr(pool, "_run_blocking", spy)

        handle = await pool.open(path, "w+")
        await handle.allocate(512)
        await handle.close()

        assert "posix_fallocate" in recorded
        assert "truncate" not in recorded


async def test_allocate_windows_branch_calls_truncate(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("allocate-spy-win.bin", b"")
    monkeypatch.setattr("aiofilepool._handle._IS_POSIX", False)

    async with pool_factory() as pool:
        recorded: list[str] = []
        original_run_blocking = pool._run_blocking

        async def spy(func, *args):
            recorded.append(func.__name__)
            return await original_run_blocking(func, *args)

        monkeypatch.setattr(pool, "_run_blocking", spy)

        handle = await pool.open(path, "w+")
        await handle.allocate(512)
        await handle.close()

        assert "truncate" in recorded
        assert "posix_fallocate" not in recorded


@pytest.mark.parametrize("is_posix", [True, False], ids=["posix", "windows"])
async def test_allocate_is_noop_when_length_below_current_size(
    pool_factory, file_writer, monkeypatch, is_posix
) -> None:
    path = file_writer("allocate-noop.bin", b"0123456789")
    monkeypatch.setattr("aiofilepool._handle._IS_POSIX", is_posix)
    if is_posix:
        _ensure_posix_fallocate(monkeypatch)

    async with pool_factory() as pool:
        handle = await pool.open(path, "r+")

        recorded: list[str] = []
        original_run_blocking = pool._run_blocking

        async def spy(func, *args):
            recorded.append(func.__name__)
            return await original_run_blocking(func, *args)

        monkeypatch.setattr(pool, "_run_blocking", spy)

        await handle.allocate(3)
        await handle.close()

    assert "posix_fallocate" not in recorded
    assert "truncate" not in recorded
    assert path.stat().st_size == 10


async def test_allocate_rejects_read_only_mode(pool_factory, file_writer) -> None:
    path = file_writer("allocate-readonly.bin", b"data")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r")
        with pytest.raises(InvalidFileModeError, match="file is not writable"):
            await handle.allocate(100)


async def test_allocate_rejects_negative_length(pool_factory, file_writer) -> None:
    path = file_writer("allocate-negative.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        with pytest.raises(InvalidPositionError, match="length must be >= 0"):
            await handle.allocate(-1)


async def test_allocate_rejects_closed_handle(pool_factory, file_writer) -> None:
    path = file_writer("allocate-closed.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.close()

        with pytest.raises(IONotOpenError):
            await handle.allocate(8)


async def test_allocate_rejects_uninitialized_handle(pool_factory, file_writer) -> None:
    path = file_writer("allocate-uninit.bin", b"")

    async with pool_factory() as pool:
        handle = pool.open(path, "w+")
        with pytest.raises(IONotOpenError):
            await handle.allocate(8)


async def test_allocate_updates_handle_size(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("allocate-size.bin", b"")
    monkeypatch.setattr("aiofilepool._handle._IS_POSIX", False)

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.allocate(64)
        assert await handle.size() == 64
        assert await handle.read(offset=0) == b"\x00" * 64
        assert await handle.tell() == 64


async def test_initialize_uses_pool_fs_stat(pool_factory, file_writer) -> None:
    path = file_writer("init-uses-fs.bin", b"hello world")
    stat_calls: list[str | bytes] = []

    class _SpyFS(AsyncFileSystem):
        async def stat(self, p):
            stat_calls.append(p)
            return await super().stat(p)

    async with pool_factory() as pool:
        pool.fs = _SpyFS(pool)

        handle = await pool.open(path, "r")

        assert stat_calls == [os.fspath(path)]
        assert await handle.size() == 11
