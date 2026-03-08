import pytest

from aiofilepool.errors import IOInitializedError, IONotOpenError, InvalidPositionError

from .conftest import FailingIO, RecordingChunker


pytestmark = pytest.mark.asyncio


async def test_async_io_initialization_detects_existing_size(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"hello") as case:
        assert await case.io.size() == 5
        assert await case.io.read() == b"hello"


async def test_async_io_context_manager_initializes_and_closes(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"context", initialize=False) as case:
        async with case.io as managed:
            assert await managed.read(3) == b"con"

        with pytest.raises(IONotOpenError):
            await case.io.read(1)


async def test_async_io_double_initialization_raises(async_io_case_factory) -> None:
    async with async_io_case_factory(data=b"abc", initialize=False) as case:
        await case.io

        with pytest.raises(IOInitializedError):
            await case.io


async def test_async_io_read_write_cursor_and_offsets(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"") as case:
        assert await case.io.write(b"abcdef") == 6
        assert await case.io.tell() == 6

        assert await case.io.seek(0) == 0
        assert await case.io.read(3) == b"abc"
        assert await case.io.tell() == 3

        assert await case.io.write(b"XY") == 2
        assert await case.io.read(offset=0) == b"abcXYf"


async def test_async_io_seek_whence_variants_and_invalid_positions(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abcdef") as case:
        assert await case.io.seek(2) == 2
        assert await case.io.seek(1, 1) == 3
        assert await case.io.seek(-1, 2) == 5

        with pytest.raises(InvalidPositionError):
            await case.io.seek(0, 999)
        with pytest.raises(InvalidPositionError):
            await case.io.seek(-1)
        with pytest.raises(InvalidPositionError):
            await case.io.seek(1, 2)


async def test_async_io_negative_offsets_and_sizes_raise_invalid_position(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abc") as case:
        with pytest.raises(InvalidPositionError):
            await case.io.read(1, -1)
        with pytest.raises(InvalidPositionError):
            await case.io.read(-1, 0)
        with pytest.raises(InvalidPositionError):
            await case.io.write(b"x", -1)


async def test_async_io_chunking_path_is_used_for_large_read_and_write(
    async_io_case_factory,
) -> None:
    payload = b"abcdefghijk"
    chunker = RecordingChunker([3, 3, 5])

    async with async_io_case_factory(
        chunking_threshold=4,
        chunker=chunker,
        data=b"",
    ) as case:
        assert await case.io.write(payload) == len(payload)
        assert await case.io.read(offset=0) == payload

    assert chunker.calls == [len(payload), len(payload)]


@pytest.mark.parametrize("thread_pool_size", [0, 2])
async def test_async_io_roundtrip_in_threaded_and_threadless_modes(
    thread_pool_size: int,
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(
        data=b"",
        thread_pool_size=thread_pool_size,
        chunking_threshold=8,
    ) as case:
        payload = b"thread-mode"

        assert await case.io.write(payload) == len(payload)
        assert await case.io.read(offset=0) == payload


async def test_async_io_chunks_small_read_leaves_remaining_bytes_available(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abcdef", chunking_threshold=10) as case:
        chunks = [chunk async for chunk in case.io.chunks(size=3, offset=1)]

        assert chunks == [b"bcd"]
        assert await case.io.read() == b"ef"


async def test_async_io_chunks_uses_custom_chunker_for_large_reads(
    async_io_case_factory,
) -> None:
    chunker = RecordingChunker([2, 2, 2])

    async with async_io_case_factory(
        data=b"abcdef",
        chunking_threshold=4,
        chunker=chunker,
    ) as case:
        chunks = [
            chunk
            async for chunk in case.io.chunks(size=6, offset=0, chunking_threshold=4)
        ]

        assert chunks == [b"ab", b"cd", b"ef"]
        assert chunker.calls == [6]
        assert await case.io.read() == b""


async def test_async_io_chunks_defaults_to_current_position_and_eof(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abcdef") as case:
        await case.io.seek(2)

        chunks = [chunk async for chunk in case.io.chunks()]

        assert chunks == [b"cdef"]
        assert await case.io.read() == b""


async def test_async_io_chunks_negative_offset_or_size_raises_invalid_position(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abcdef") as case:
        with pytest.raises(InvalidPositionError):
            [chunk async for chunk in case.io.chunks(size=1, offset=-1)]
        with pytest.raises(InvalidPositionError):
            [chunk async for chunk in case.io.chunks(size=-1, offset=0)]


async def test_async_io_chunks_closed_object_raises_not_open(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abcdef") as case:
        await case.io.close()

        with pytest.raises(IONotOpenError):
            [chunk async for chunk in case.io.chunks()]


async def test_uninitialized_async_io_rejects_operations_until_initialized(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abc", initialize=False) as case:
        with pytest.raises(IONotOpenError):
            await case.io.read(1)
        with pytest.raises(IONotOpenError):
            await case.io.write(b"x")
        with pytest.raises(IONotOpenError):
            [chunk async for chunk in case.io.chunks()]

        await case.io.close()
        await case.io

        assert await case.io.read() == b"abc"


async def test_async_io_read_past_eof_returns_short_data(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abc") as case:
        assert await case.io.read(size=5, offset=2) == b"c"


async def test_async_io_chunks_past_eof_return_only_consumed_data(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(data=b"abc") as case:
        chunks = [chunk async for chunk in case.io.chunks(size=5, offset=2)]

        assert chunks == [b"c"]
        assert await case.io.read() == b""


async def test_async_io_chunks_early_close_keeps_remaining_bytes_available(
    async_io_case_factory,
) -> None:
    chunker = RecordingChunker([2, 2, 2])

    async with async_io_case_factory(
        data=b"abcdef",
        chunking_threshold=4,
        chunker=chunker,
    ) as case:
        chunks = case.io.chunks(size=6, offset=0, chunking_threshold=4)

        assert await anext(chunks) == b"ab"
        await chunks.aclose()

        assert await case.io.read() == b"cdef"
        assert chunker.calls[0] == 6


async def test_async_io_chunks_accept_oversized_chunk_sizes(
    async_io_case_factory,
) -> None:
    chunker = RecordingChunker([4, 4])

    async with async_io_case_factory(
        data=b"abcdef",
        chunking_threshold=4,
        chunker=chunker,
    ) as case:
        chunks = [
            chunk
            async for chunk in case.io.chunks(size=6, offset=0, chunking_threshold=4)
        ]

        assert chunks == [b"abcd", b"ef"]
        assert await case.io.read() == b""
        assert chunker.calls == [6]


async def test_async_io_read_failure_leaves_object_usable(
    async_io_case_factory,
) -> None:
    async with async_io_case_factory(
        data=b"abc",
        thread_pool_size=0,
        io_obj=FailingIO(b"abc", fail_on={"read": 1}),
    ) as case:
        with pytest.raises(RuntimeError, match="read failed"):
            await case.io.read(1, 0)

        assert await case.io.read(1, 0) == b"a"


async def test_async_io_chunked_write_failure_preserves_committed_prefix(
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
        assert await case.io.read(offset=0) == b"ab"
