import os
from pathlib import Path

import pytest

from aiofilepool import StrOrBytesPath


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
async def test_fs_stat_returns_size_for_existing_file(
    pool_factory, file_writer, path_factory
) -> None:
    path = file_writer("fs-stat.bin", b"abcdef")

    async with pool_factory() as pool:
        stat = await pool.fs.stat(path_factory(path))
        assert stat.st_size == 6


async def test_fs_stat_raises_filenotfound_for_missing_path(
    pool_factory, tmp_path
) -> None:
    async with pool_factory() as pool:
        with pytest.raises(FileNotFoundError):
            await pool.fs.stat(tmp_path / "missing.bin")


@pytest.mark.parametrize("path_factory", _PATH_FACTORIES)
async def test_fs_rename_moves_file_and_preserves_contents(
    pool_factory, file_writer, tmp_path, path_factory
) -> None:
    old = file_writer("rename-old.bin", b"payload")
    new = tmp_path / "rename-new.bin"

    async with pool_factory() as pool:
        await pool.fs.rename(path_factory(old), path_factory(new))

        assert await pool.fs.exists(path_factory(old)) is False
        assert await pool.fs.exists(path_factory(new)) is True
        assert new.read_bytes() == b"payload"


async def test_fs_rename_raises_when_source_missing(pool_factory, tmp_path) -> None:
    async with pool_factory() as pool:
        with pytest.raises(FileNotFoundError):
            await pool.fs.rename(tmp_path / "missing.bin", tmp_path / "target.bin")


@pytest.mark.parametrize("path_factory", _PATH_FACTORIES)
async def test_fs_exists_returns_true_for_file_false_for_missing(
    pool_factory, file_writer, tmp_path, path_factory
) -> None:
    path = file_writer("exists.bin", b"x")

    async with pool_factory() as pool:
        assert await pool.fs.exists(path_factory(path)) is True
        assert await pool.fs.exists(path_factory(tmp_path)) is True
        assert await pool.fs.exists(path_factory(tmp_path / "missing.bin")) is False


@pytest.mark.parametrize("path_factory", _PATH_FACTORIES)
async def test_fs_is_file_true_for_regular_file_false_for_directory(
    pool_factory, file_writer, tmp_path, path_factory
) -> None:
    path = file_writer("is-file.bin", b"x")

    async with pool_factory() as pool:
        assert await pool.fs.is_file(path_factory(path)) is True
        assert await pool.fs.is_file(path_factory(tmp_path)) is False
        assert await pool.fs.is_file(path_factory(tmp_path / "missing.bin")) is False


@pytest.mark.parametrize("path_factory", _PATH_FACTORIES)
async def test_fs_is_dir_true_for_directory_false_for_file(
    pool_factory, file_writer, tmp_path, path_factory
) -> None:
    path = file_writer("is-dir.bin", b"x")

    async with pool_factory() as pool:
        assert await pool.fs.is_dir(path_factory(tmp_path)) is True
        assert await pool.fs.is_dir(path_factory(path)) is False
        assert await pool.fs.is_dir(path_factory(tmp_path / "missing.bin")) is False


async def test_fs_methods_run_on_pool_executor(
    pool_factory, file_writer, tmp_path, monkeypatch
) -> None:
    path = file_writer("dispatch.bin", b"abc")
    target = tmp_path / "dispatch-renamed.bin"

    async with pool_factory() as pool:
        recorded: list[str] = []
        original_run_blocking = pool._run_blocking

        async def spy(func, *args):
            recorded.append(func.__name__)
            return await original_run_blocking(func, *args)

        monkeypatch.setattr(pool, "_run_blocking", spy)

        await pool.fs.stat(path)
        await pool.fs.exists(path)
        await pool.fs.is_file(path)
        await pool.fs.is_dir(path)
        await pool.fs.rename(path, target)

        assert recorded == [
            os.stat.__name__,
            os.path.exists.__name__,
            os.path.isfile.__name__,
            os.path.isdir.__name__,
            os.rename.__name__,
        ]
