"""Concurrent disjoint-offset IO on a single handle — the load profile filepart
exercises in `download_manager.write_part` (scatter writes) and
`upload`/`fast_file_hash` (concurrent ranged reads).

Hypothesis-driven: each test generates a random set of disjoint intervals, drives
N concurrent tasks against one handle, and verifies the result via the deterministic
`assemble_intervals` oracle.

Because `_op_lock` serializes per-handle operations, the contract under test is:
*for any task completion ordering of disjoint-offset writes, the final on-disk
bytes equal the deterministic assembly* — i.e. no per-task interference,
no lost writes, no fd-eviction corruption.
"""

import asyncio

import pytest
from hypothesis import HealthCheck, given, settings

from aiofilepool import FilePool

from .conftest import assemble_intervals, count_fd, disjoint_intervals


pytestmark = [pytest.mark.asyncio, pytest.mark.slow]


_HYPOTHESIS_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)


# --- Concurrent disjoint writes (download fan-out pattern) --------------------


@_HYPOTHESIS_SETTINGS
@given(intervals=disjoint_intervals(max_total_size=4096, max_intervals=12))
async def test_concurrent_disjoint_writes_reproduce_assembled_buffer(
    tmp_path_factory, intervals: list[tuple[int, bytes]]
) -> None:
    """N concurrent `write(data, offset=)` tasks on one handle reproduce the
    deterministic assembled buffer, regardless of task completion order.

    This is the `download_manager.write_part` invariant: parts arrive in any
    order over the network, but the final file is byte-for-byte correct.
    """
    total_size = max(off + len(data) for off, data in intervals)
    base = tmp_path_factory.mktemp("scatter-write")
    path = base / "scatter.bin"
    path.touch()

    async with FilePool(
        descriptor_pool_size=2, thread_pool_size=2, chunking_threshold=1024
    ) as pool:
        handle = await pool.open(path, "r+b")
        await handle.allocate(total_size)

        await asyncio.gather(
            *(handle.write(data, offset=offset) for offset, data in intervals)
        )

        await handle.seek(0)
        actual = await handle.read()
        await handle.close()

    expected = assemble_intervals(intervals, total_size)
    assert actual == expected


# --- Concurrent disjoint reads (upload / fast_file_hash pattern) --------------


@_HYPOTHESIS_SETTINGS
@given(intervals=disjoint_intervals(max_total_size=4096, max_intervals=12))
async def test_concurrent_disjoint_reads_return_correct_slices(
    tmp_path_factory, intervals: list[tuple[int, bytes]]
) -> None:
    """N concurrent `read(size, offset=)` tasks on a pre-written handle return
    each interval's bytes regardless of completion order.

    This is the `upload` / `fast_file_hash` invariant: the part-upload pipeline
    issues many ranged reads in parallel, and each must see its own slice.
    """
    total_size = max(off + len(data) for off, data in intervals)
    base = tmp_path_factory.mktemp("scatter-read")
    path = base / "preloaded.bin"
    expected_file = assemble_intervals(intervals, total_size)
    path.write_bytes(expected_file)

    async with FilePool(
        descriptor_pool_size=2, thread_pool_size=2, chunking_threshold=1024
    ) as pool:
        handle = await pool.open(path, "rb")

        results = await asyncio.gather(
            *(handle.read(size=len(data), offset=offset) for offset, data in intervals)
        )

        await handle.close()

    for (offset, data), got in zip(intervals, results):
        assert got == data, f"slice at offset={offset} mismatched"


# --- Mixed read+write across disjoint intervals -------------------------------


@_HYPOTHESIS_SETTINGS
@given(intervals=disjoint_intervals(max_total_size=4096, max_intervals=10))
async def test_mixed_concurrent_reads_and_writes_on_disjoint_intervals(
    tmp_path_factory, intervals: list[tuple[int, bytes]]
) -> None:
    """Half the intervals are written, half are read (after the writes complete
    visibly). The read half must reflect the pre-written content; the written
    half must land at its offset."""
    if len(intervals) < 2:
        return  # Need at least two intervals to split read/write.

    total_size = max(off + len(data) for off, data in intervals)
    base = tmp_path_factory.mktemp("scatter-mixed")
    path = base / "mixed.bin"

    write_half = intervals[: len(intervals) // 2]
    read_half = intervals[len(intervals) // 2 :]

    expected_initial = assemble_intervals(read_half, total_size)
    path.write_bytes(expected_initial)

    async with FilePool(
        descriptor_pool_size=2, thread_pool_size=2, chunking_threshold=1024
    ) as pool:
        handle = await pool.open(path, "r+b")
        await handle.allocate(total_size)

        # Drive writes and reads concurrently.
        write_tasks = [handle.write(data, offset=offset) for offset, data in write_half]
        read_tasks = [
            handle.read(size=len(data), offset=offset) for offset, data in read_half
        ]

        results = await asyncio.gather(*write_tasks, *read_tasks)
        read_results = results[len(write_tasks) :]

        # Each read returns its pre-written content.
        for (offset, data), got in zip(read_half, read_results):
            assert got == data, f"read slice at offset={offset} mismatched"

        # Final file: union of pre-existing read-half bytes and newly-written write-half.
        await handle.seek(0)
        final = await handle.read()
        await handle.close()

    expected_final = assemble_intervals(intervals, total_size)
    assert final == expected_final


# --- Cross-handle scatter under fd-eviction pressure --------------------------


@_HYPOTHESIS_SETTINGS
@given(intervals=disjoint_intervals(max_total_size=2048, max_intervals=8))
async def test_two_handles_scatter_writes_under_cap_one_preserve_both_files(
    tmp_path_factory, intervals: list[tuple[int, bytes]]
) -> None:
    """Two handles on different paths share a `cap=1` pool. fd eviction must
    not corrupt either file's content under interleaved scatter writes.

    This catches eviction races that single-handle tests cannot reach.
    """
    total_size = max(off + len(data) for off, data in intervals)
    base = tmp_path_factory.mktemp("scatter-two-handles")
    path_a = base / "a.bin"
    path_b = base / "b.bin"
    path_a.touch()
    path_b.touch()

    async with FilePool(
        descriptor_pool_size=1, thread_pool_size=0, chunking_threshold=1024
    ) as pool:
        handle_a = await pool.open(path_a, "r+b")
        handle_b = await pool.open(path_b, "r+b")
        await handle_a.allocate(total_size)
        await handle_b.allocate(total_size)

        # Interleave: each interval is written to both handles concurrently.
        await asyncio.gather(
            *(handle_a.write(data, offset=offset) for offset, data in intervals),
            *(handle_b.write(data, offset=offset) for offset, data in intervals),
        )

        # Cap was respected throughout (best we can assert post-hoc).
        assert count_fd(pool) <= 1

        await handle_a.seek(0)
        await handle_b.seek(0)
        bytes_a = await handle_a.read()
        bytes_b = await handle_b.read()
        await handle_a.close()
        await handle_b.close()

    expected = assemble_intervals(intervals, total_size)
    assert bytes_a == expected
    assert bytes_b == expected


# --- Deterministic regression: maximum fan-out under cap=4 --------------------


async def test_download_fan_out_under_cap_pressure_preserves_every_file(
    tmp_path,
) -> None:
    """Mirrors filepart's worst-case load: many handles, many parts each, cap=4."""
    handle_count = 10
    parts_per_file = 8
    part_size = 8 * 1024
    total_size = parts_per_file * part_size

    paths = [tmp_path / f"f{i}.bin" for i in range(handle_count)]
    for p in paths:
        p.touch()

    payloads: list[list[bytes]] = []
    for i in range(handle_count):
        chunks = []
        for j in range(parts_per_file):
            chunks.append(
                bytes((i * parts_per_file + j) % 256 for _ in range(part_size))
            )
        payloads.append(chunks)

    async with FilePool(
        descriptor_pool_size=4, thread_pool_size=2, chunking_threshold=1 << 20
    ) as pool:
        handles = [await pool.open(p, "r+b") for p in paths]
        for h in handles:
            await h.allocate(total_size)

        max_observed = 0

        async def write_one(handle, parts):
            nonlocal max_observed
            for j, data in enumerate(parts):
                await handle.write(data, offset=j * part_size)
                max_observed = max(max_observed, count_fd(pool))

        await asyncio.gather(
            *(write_one(h, parts) for h, parts in zip(handles, payloads))
        )

        for h in handles:
            await h.close()

        assert max_observed <= 4

    for p, parts in zip(paths, payloads):
        assert p.read_bytes() == b"".join(parts)


async def test_starvation_guard_cap_one_with_many_handles_completes(
    tmp_path,
) -> None:
    """32 handles each do one write under cap=1 — every task must complete (no
    waiter starvation) within a generous wall-clock timeout."""
    handle_count = 32
    paths = [tmp_path / f"sg{i}.bin" for i in range(handle_count)]
    for p in paths:
        p.touch()

    async with FilePool(
        descriptor_pool_size=1, thread_pool_size=0, chunking_threshold=1024
    ) as pool:
        handles = [await pool.open(p, "r+b") for p in paths]

        async def write_one(idx: int):
            await handles[idx].allocate(8)
            await handles[idx].write(idx.to_bytes(8, "big"), offset=0)

        await asyncio.wait_for(
            asyncio.gather(*(write_one(i) for i in range(handle_count))),
            timeout=5.0,
        )

        for h in handles:
            await h.close()

        assert count_fd(pool) == 0

    for idx, p in enumerate(paths):
        assert p.read_bytes() == idx.to_bytes(8, "big")
