import os
from pathlib import Path

import pytest

from aiofilepool import FileHandle, ModeSpec, StrPath
from aiofilepool.errors import (
    IOInitializedError,
    IONotOpenError,
    FilePoolNotOpenError,
    InvalidFileModeError,
    InvalidPositionError,
)
from .conftest import FailingIO, RecordingChunker


pytestmark = pytest.mark.asyncio


def _as_path(path: Path) -> StrPath:
    return path


def _as_str(path: Path) -> StrPath:
    return str(path)


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

        with pytest.raises(IONotOpenError):
            await handle.read(1)


async def test_double_initialization_raises(pool_factory, file_writer) -> None:
    path = file_writer("double-init.bin", b"abc")

    async with pool_factory() as pool:
        handle = pool.open(path, "r")
        await handle
        with pytest.raises(IOInitializedError):
            await handle


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

        assert isinstance(handle._path, str)
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
    path = file_writer("direct-write-handle.bin", b"")

    async with pool_factory(thread_pool_size=0) as pool:
        handle = await FileHandle(pool, path_factory(path), ModeSpec.from_str("w+"))

        assert isinstance(handle._path, str)
        assert await handle.write(b"abc") == 3
        assert await handle.read(offset=0) == b"abc"


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

        with pytest.raises(InvalidPositionError, match="invalid whence"):
            await handle.seek(0, 999)
        with pytest.raises(InvalidPositionError, match="invalid position"):
            await handle.seek(-1, os.SEEK_SET)
        with pytest.raises(InvalidPositionError, match="invalid position"):
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

        with pytest.raises(IONotOpenError):
            await handle.read(1, 0)
        with pytest.raises(IONotOpenError):
            await handle.write(b"x")
        with pytest.raises(IONotOpenError):
            await handle.truncate(0)


async def test_negative_offsets_and_sizes_raise_invalid_position(
    pool_factory, file_writer
) -> None:
    path = file_writer("positions.bin", b"abc")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r+")

        with pytest.raises(InvalidPositionError, match="offset must be >= 0"):
            await handle.read(1, -1)
        with pytest.raises(InvalidPositionError, match="size must be >= 0"):
            await handle.read(-1, 0)
        with pytest.raises(InvalidPositionError, match="offset must be >= 0"):
            await handle.write(b"x", -1)
        with pytest.raises(InvalidPositionError, match="truncate size must be >= 0"):
            await handle.truncate(-1)


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


async def test_handle_chunks_small_read_returns_single_chunk_and_advances_position(
    pool_factory, file_writer
) -> None:
    path = file_writer("chunks-small.bin", b"abcdef")

    async with pool_factory(chunking_threshold=10) as pool:
        handle = await pool.open(path, "r")
        chunks = [chunk async for chunk in handle.chunks(size=3, offset=1)]

        assert chunks == [b"bcd"]
        assert await handle.tell() == 4


async def test_handle_chunks_uses_custom_chunker_for_large_reads(
    pool_factory, file_writer
) -> None:
    path = file_writer("chunks-large.bin", b"abcdef")
    chunker = RecordingChunker([2, 2, 2])

    async with pool_factory(chunking_threshold=4, chunker=chunker) as pool:
        handle = await pool.open(path, "r")
        chunks = [
            chunk
            async for chunk in handle.chunks(size=6, offset=0, chunking_threshold=4)
        ]

        assert chunks == [b"ab", b"cd", b"ef"]
        assert chunker.calls == [6]
        assert await handle.tell() == 6


async def test_handle_chunks_defaults_to_current_position_and_eof(
    pool_factory, file_writer
) -> None:
    path = file_writer("chunks-default-window.bin", b"abcdef")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r")
        await handle.seek(2)

        chunks = [chunk async for chunk in handle.chunks()]

        assert chunks == [b"cdef"]
        assert await handle.tell() == 6


async def test_handle_chunks_negative_offset_or_size_raises_invalid_position(
    pool_factory, file_writer
) -> None:
    path = file_writer("chunks-invalid-position.bin", b"abcdef")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r")

        with pytest.raises(InvalidPositionError, match="offset must be >= 0"):
            [chunk async for chunk in handle.chunks(size=1, offset=-1)]
        with pytest.raises(InvalidPositionError, match="size must be >= 0"):
            [chunk async for chunk in handle.chunks(size=-1, offset=0)]


async def test_handle_chunks_closed_handle_raises_not_open(
    pool_factory, file_writer
) -> None:
    path = file_writer("chunks-closed.bin", b"abcdef")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r")
        await handle.close()

        with pytest.raises(IONotOpenError):
            [chunk async for chunk in handle.chunks()]


async def test_handle_chunks_write_only_handle_raises_invalid_mode(
    pool_factory, file_writer
) -> None:
    path = file_writer("chunks-write-only.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w")
        with pytest.raises(InvalidFileModeError, match="file is not readable"):
            [chunk async for chunk in handle.chunks()]


async def test_uninitialized_handle_rejects_io_until_initialized(
    pool_factory, file_writer
) -> None:
    path = file_writer("uninitialized-handle.bin", b"abc")

    async with pool_factory() as pool:
        handle = pool.open(path, "r+")

        with pytest.raises(IONotOpenError):
            await handle.read(1)
        with pytest.raises(IONotOpenError):
            await handle.write(b"x")
        with pytest.raises(IONotOpenError):
            await handle.truncate(0)
        with pytest.raises(IONotOpenError):
            [chunk async for chunk in handle.chunks()]

        await handle.close()
        assert await handle.tell() == 0
        assert await handle.size() == 0

        await handle
        assert await handle.read() == b"abc"


async def test_missing_path_read_handle_raises_during_initialization(
    pool_factory, tmp_path
) -> None:
    missing_path = tmp_path / "missing-read-handle.bin"

    async with pool_factory() as pool:
        handle = pool.open(missing_path, "r")
        with pytest.raises(FileNotFoundError):
            await handle


async def test_handle_read_past_eof_advances_by_requested_window(
    pool_factory, file_writer
) -> None:
    path = file_writer("read-past-eof.bin", b"abc")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r")

        assert await handle.read(size=5, offset=2) == b"c"
        assert await handle.tell() == 7


async def test_handle_chunks_past_eof_advances_by_consumed_bytes_only(
    pool_factory, file_writer
) -> None:
    path = file_writer("chunks-past-eof.bin", b"abc")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r")
        chunks = [chunk async for chunk in handle.chunks(size=5, offset=2)]

        assert chunks == [b"c"]
        assert await handle.tell() == 3


async def test_handle_chunks_zero_size_beyond_eof_yields_empty_chunk_without_consuming(
    pool_factory, file_writer
) -> None:
    path = file_writer("chunks-zero-beyond-eof.bin", b"abc")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r")
        chunks = [chunk async for chunk in handle.chunks(size=0, offset=5)]

        assert chunks == [b""]
        assert await handle.tell() == 5


async def test_handle_chunks_early_close_updates_position_to_consumed_bytes(
    pool_factory, file_writer
) -> None:
    path = file_writer("chunks-early-close.bin", b"abcdef")
    chunker = RecordingChunker([2, 2, 2])

    async with pool_factory(chunking_threshold=4, chunker=chunker) as pool:
        handle = await pool.open(path, "r")
        chunks = handle.chunks(size=6, offset=0, chunking_threshold=4)

        assert await anext(chunks) == b"ab"
        await chunks.aclose()

        assert await handle.tell() == 2
        assert chunker.calls == [6]


async def test_handle_chunks_accepts_chunk_sizes_larger_than_remaining_data(
    pool_factory, file_writer
) -> None:
    path = file_writer("chunks-oversized-window.bin", b"abcdef")
    chunker = RecordingChunker([4, 4])

    async with pool_factory(chunking_threshold=4, chunker=chunker) as pool:
        handle = await pool.open(path, "r")
        chunks = [
            chunk
            async for chunk in handle.chunks(size=6, offset=0, chunking_threshold=4)
        ]

        assert chunks == [b"abcd", b"ef"]
        assert await handle.tell() == 6
        assert chunker.calls == [6]


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


async def test_handle_read_failure_leaves_position_unchanged(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("failing-read.bin", b"abc")
    failing_io = FailingIO(b"abc", fail_on={"read": 1})

    monkeypatch.setattr(
        "aiofilepool._fd_manager.open", lambda path, mode: failing_io, raising=False
    )

    async with pool_factory(thread_pool_size=0) as pool:
        handle = await pool.open(path, "r")

        with pytest.raises(RuntimeError, match="read failed"):
            await handle.read(1, 0)

        assert await handle.tell() == 0
        assert await handle.size() == 3


async def test_handle_chunked_write_failure_preserves_committed_prefix(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("failing-write.bin", b"")
    chunker = RecordingChunker([2, 2, 2])
    failing_io = FailingIO(fail_on={"write": 2})

    monkeypatch.setattr(
        "aiofilepool._fd_manager.open", lambda path, mode: failing_io, raising=False
    )

    async with pool_factory(
        thread_pool_size=0, chunking_threshold=4, chunker=chunker
    ) as pool:
        handle = await pool.open(path, "w+")

        with pytest.raises(RuntimeError, match="write failed"):
            await handle.write(b"abcdef")

        assert await handle.tell() == 2
        assert await handle.size() == 2
        assert failing_io.getvalue() == b"ab"
