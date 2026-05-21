"""Behavior contract shared by `FileHandle` and `BinaryIOAdapter`.

Parametrized via the `async_io_case_factory` fixture: every test marked
`(handle, adapter)` runs once against each implementation, so both subclasses
of `AsyncBinaryIO` must satisfy the same observable contract.

Pins:
* cursor advancement and `offset=` override on read / write
* `seek` whence variants, EOF soft-clamp, negative-position reject
* chunked-path activation at `chunking_threshold`, custom chunker honored
* `chunks()` past-EOF early-returns; `aclose()` mid-stream restores cursor
* mode/access matrix (read/write/chunks/truncate)
* state-machine errors: double-init, uninitialized ops, post-close ops
* `BinaryIOAdapter` quirks: does not close underlying io, survives pool close
"""

import io

import pytest

from aiofilepool.errors import (
    IOInitializedError,
    IONotOpenError,
    InvalidFileModeError,
    InvalidPositionError,
)

from .conftest import FailingIO, RecordingChunker


pytestmark = pytest.mark.asyncio


# --- Shared AsyncBinaryIO contract (handle + adapter) -------------------------


async def test_initialization_detects_existing_size(async_io_case_factory) -> None:
    async with async_io_case_factory(data=b"hello") as case:
        assert await case.io.size() == 5
        assert await case.io.read() == b"hello"


async def test_context_manager_initializes_and_closes(async_io_case_factory) -> None:
    async with async_io_case_factory(data=b"context", initialize=False) as case:
        async with case.io as managed:
            assert await managed.read(3) == b"con"

        with pytest.raises(IONotOpenError):
            await case.io.read(1)


async def test_double_initialization_raises_io_initialized(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abc", initialize=False) as case:
        await case.io

        with pytest.raises(IOInitializedError):
            await case.io


async def test_read_write_cursor_and_explicit_offset(async_io_case_factory) -> None:
    async with async_io_case_factory(data=b"") as case:
        assert await case.io.write(b"abcdef") == 6
        assert await case.io.tell() == 6

        assert await case.io.seek(0) == 0
        assert await case.io.read(3) == b"abc"
        assert await case.io.tell() == 3

        assert await case.io.write(b"XY") == 2
        assert await case.io.read(offset=0) == b"abcXYf"


async def test_seek_whence_variants_clamp_at_eof_and_reject_negative(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abcdef") as case:
        assert await case.io.seek(2) == 2
        assert await case.io.seek(1, 1) == 3
        assert await case.io.seek(-1, 2) == 5
        # Past-EOF seek soft-clamps to size.
        assert await case.io.seek(999, 2) == 6

        with pytest.raises(InvalidPositionError):
            await case.io.seek(0, 999)
        with pytest.raises(InvalidPositionError):
            await case.io.seek(-1)


async def test_negative_offset_or_size_rejected(async_io_case_factory) -> None:
    async with async_io_case_factory(data=b"abc") as case:
        with pytest.raises(InvalidPositionError):
            await case.io.read(1, -1)
        with pytest.raises(InvalidPositionError):
            await case.io.read(-1, 0)
        with pytest.raises(InvalidPositionError):
            await case.io.write(b"x", -1)


async def test_chunking_path_used_for_large_read_and_write(
    async_io_case_factory,
) -> None:
    payload = b"abcdefghijk"
    chunker = RecordingChunker([3, 3, 5])

    async with async_io_case_factory(
        chunking_threshold=4, chunker=chunker, data=b""
    ) as case:
        assert await case.io.write(payload) == len(payload)
        assert await case.io.read(offset=0) == payload

    assert chunker.calls == [len(payload), len(payload)]


@pytest.mark.parametrize("thread_pool_size", [0, 2])
async def test_roundtrip_in_threaded_and_threadless_modes(
    thread_pool_size: int, async_io_case_factory
) -> None:
    async with async_io_case_factory(
        data=b"", thread_pool_size=thread_pool_size, chunking_threshold=8
    ) as case:
        payload = b"thread-mode"
        assert await case.io.write(payload) == len(payload)
        assert await case.io.read(offset=0) == payload


async def test_zero_size_read_returns_empty_and_updates_position(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abcdef") as case:
        # offset clamps past-EOF reads to file size; size=0 still updates position.
        assert await case.io.read(size=0, offset=2) == b""
        assert await case.io.tell() == 2


async def test_chunks_small_read_leaves_remaining_bytes_available(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abcdef", chunking_threshold=10) as case:
        chunks = [chunk async for chunk in case.io.chunks(size=3, offset=1)]

        assert chunks == [b"bcd"]
        assert await case.io.read() == b"ef"


async def test_chunks_uses_custom_chunker_for_large_reads(
    async_io_case_factory,
) -> None:
    chunker = RecordingChunker([2, 2, 2])

    async with async_io_case_factory(
        data=b"abcdef", chunking_threshold=4, chunker=chunker
    ) as case:
        chunks = [
            chunk
            async for chunk in case.io.chunks(size=6, offset=0, chunking_threshold=4)
        ]

        assert chunks == [b"ab", b"cd", b"ef"]
        assert chunker.calls == [6]
        assert await case.io.read() == b""


async def test_chunks_defaults_to_current_position_and_eof(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abcdef") as case:
        await case.io.seek(2)

        chunks = [chunk async for chunk in case.io.chunks()]

        assert chunks == [b"cdef"]
        assert await case.io.read() == b""


async def test_chunks_negative_offset_or_size_rejected(async_io_case_factory) -> None:
    async with async_io_case_factory(data=b"abcdef") as case:
        with pytest.raises(InvalidPositionError):
            [chunk async for chunk in case.io.chunks(size=1, offset=-1)]
        with pytest.raises(InvalidPositionError):
            [chunk async for chunk in case.io.chunks(size=-1, offset=0)]


async def test_chunks_past_eof_yields_no_data(async_io_case_factory) -> None:
    """`chunks()` with offset past EOF early-returns: no chunks yielded, position == size."""
    async with async_io_case_factory(data=b"abc") as case:
        chunks = [chunk async for chunk in case.io.chunks(size=10, offset=100)]
        assert chunks == []
        assert await case.io.tell() == 3


async def test_chunks_partial_read_past_eof_returns_only_available_bytes(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abc") as case:
        chunks = [chunk async for chunk in case.io.chunks(size=5, offset=2)]

        assert chunks == [b"c"]
        assert await case.io.read() == b""


async def test_chunks_early_aclose_restores_position_to_consumed_bytes(
    async_io_case_factory,
) -> None:
    chunker = RecordingChunker([2, 2, 2])

    async with async_io_case_factory(
        data=b"abcdef", chunking_threshold=4, chunker=chunker
    ) as case:
        chunks = case.io.chunks(size=6, offset=0, chunking_threshold=4)

        assert await anext(chunks) == b"ab"
        await chunks.aclose()

        # Remaining bytes still readable from the cursor where aclose left it.
        assert await case.io.read() == b"cdef"
        assert chunker.calls[0] == 6


async def test_chunks_accept_oversized_chunk_schedule(async_io_case_factory) -> None:
    """A chunker that yields chunk sizes larger than remaining bytes returns short reads."""
    chunker = RecordingChunker([4, 4])

    async with async_io_case_factory(
        data=b"abcdef", chunking_threshold=4, chunker=chunker
    ) as case:
        chunks = [
            chunk
            async for chunk in case.io.chunks(size=6, offset=0, chunking_threshold=4)
        ]

        assert chunks == [b"abcd", b"ef"]
        assert await case.io.read() == b""
        assert chunker.calls == [6]


async def test_uninitialized_io_rejects_operations_until_initialized(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abc", initialize=False) as case:
        with pytest.raises(IONotOpenError):
            await case.io.read(1)
        with pytest.raises(IONotOpenError):
            await case.io.write(b"x")
        with pytest.raises(IONotOpenError):
            [chunk async for chunk in case.io.chunks()]

        # Closing an uninitialized object is a no-op; initializing after close still works.
        await case.io.close()
        await case.io

        assert await case.io.read() == b"abc"


async def test_read_failure_leaves_io_usable(async_io_case_factory) -> None:
    async with async_io_case_factory(
        data=b"abc",
        thread_pool_size=0,
        io_obj=FailingIO(b"abc", fail_on={"read": 1}),
    ) as case:
        with pytest.raises(RuntimeError, match="read failed"):
            await case.io.read(1, 0)

        assert await case.io.read(1, 0) == b"a"


async def test_chunked_write_failure_preserves_committed_prefix(
    async_io_case_factory,
) -> None:
    chunker = RecordingChunker([2, 2, 2])
    failing_io = FailingIO(fail_on={"write": 2})

    async with async_io_case_factory(
        data=b"",
        thread_pool_size=0,
        chunking_threshold=4,
        chunker=chunker,
        io_obj=failing_io,
    ) as case:
        with pytest.raises(RuntimeError, match="write failed"):
            await case.io.write(b"abcdef")

        assert await case.io.size() == 2
        assert await case.io.tell() == 2
        assert await case.io.read(offset=0) == b"ab"


# --- FileHandle-only: mode-driven access matrix -------------------------------


@pytest.mark.parametrize(
    ("mode", "method", "args"),
    [
        ("w", "read", (1, 0)),
        ("w", "chunks", ()),
        ("r", "write", (b"x",)),
        ("r", "truncate", (0,)),
        ("r", "allocate", (10,)),
    ],
)
async def test_handle_mode_rejects_operations_outside_access(
    pool_factory, file_writer, mode: str, method: str, args: tuple
) -> None:
    """ModeSpec drives the access matrix: read mode rejects writes; write mode rejects reads."""
    path = file_writer(f"mode-{mode}-{method}.bin", b"abc")

    async with pool_factory() as pool:
        handle = await pool.open(path, mode)

        target = getattr(handle, method)
        if method == "chunks":
            with pytest.raises(InvalidFileModeError):
                [chunk async for chunk in target(*args)]
        else:
            with pytest.raises(InvalidFileModeError):
                await target(*args)


async def test_handle_closed_state_rejects_all_io(pool_factory, file_writer) -> None:
    path = file_writer("closed-handle.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.close()

        with pytest.raises(IONotOpenError):
            await handle.read(1, 0)
        with pytest.raises(IONotOpenError):
            await handle.write(b"x")
        with pytest.raises(IONotOpenError):
            await handle.truncate(0)
        with pytest.raises(IONotOpenError):
            await handle.allocate(8)
        with pytest.raises(IONotOpenError):
            [chunk async for chunk in handle.chunks()]


async def test_handle_truncate_grows_with_zero_padding_and_clamps_position(
    pool_factory, file_writer
) -> None:
    path = file_writer("truncate-grow.bin", b"ab")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r+")
        await handle.truncate(5)

        assert await handle.tell() == 5
        assert await handle.size() == 5
        assert await handle.read(offset=0) == b"ab\x00\x00\x00"


async def test_handle_truncate_defaults_to_current_position(
    pool_factory, file_writer
) -> None:
    path = file_writer("truncate-pos.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.write(b"abcdef")
        await handle.seek(4)
        await handle.truncate()

        assert await handle.tell() == 4
        assert await handle.read(offset=0) == b"abcd"


async def test_handle_truncate_rejects_negative_size(pool_factory, file_writer) -> None:
    path = file_writer("truncate-neg.bin", b"abc")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r+")

        with pytest.raises(InvalidPositionError, match="truncate size must be >= 0"):
            await handle.truncate(-1)


async def test_handle_missing_path_for_read_raises_during_initialization(
    pool_factory, tmp_path
) -> None:
    missing = tmp_path / "does-not-exist.bin"

    async with pool_factory() as pool:
        handle = pool.open(missing, "r")
        with pytest.raises(FileNotFoundError):
            await handle


async def test_handle_open_accepts_every_strorbytespath_shape(
    pool_factory, file_writer, path_factory
) -> None:
    """One parametric coverage of every supported `StrOrBytesPath` shape."""
    path = file_writer("strpath-open.bin", b"")

    async with pool_factory(thread_pool_size=0) as pool:
        handle = await pool.open(path_factory(path), "w+")

        assert await handle.write(b"payload") == 7
        assert await handle.read(offset=0) == b"payload"


# --- BinaryIOAdapter-only quirks ----------------------------------------------


async def test_adapter_close_does_not_close_underlying_io(pool_factory) -> None:
    """The adapter wraps a caller-owned BinaryIO; closing the adapter must leave it open."""
    async with pool_factory() as pool:
        bio = io.BytesIO(b"context")

        async with pool.manage(bio) as adapter:
            assert await adapter.read(3) == b"con"

        assert bio.closed is False
        with pytest.raises(IONotOpenError):
            await adapter.read(1)


async def test_adapter_close_is_idempotent(pool_factory, binary_io_factory) -> None:
    async with pool_factory() as pool:
        adapter, bio = binary_io_factory(pool, b"abc")
        adapter = await adapter

        await adapter.close()
        await adapter.close()

        assert bio.closed is False
        with pytest.raises(IONotOpenError):
            await adapter.read(1, 0)


async def test_initialized_adapter_remains_usable_after_pool_close(
    pool_factory,
) -> None:
    """The adapter owns no descriptor — `pool.close()` must not invalidate it."""
    pool = pool_factory()
    adapter = await pool.manage(io.BytesIO(b"abc"))

    await pool.close()

    assert await adapter.read(offset=0) == b"abc"
    assert await adapter.write(b"XY", offset=1) == 2
    assert await adapter.read(offset=0) == b"aXY"
