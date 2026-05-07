"""Hypothesis-driven capacity-invariant property test for `FilePool`.

The pool's defining invariant: at most `descriptor_pool_size` OS file
descriptors are open at any moment, even when the caller holds N >> cap
logical `FileHandle`s. This test drives the pool with random sequences of
operations across many handles and asserts:

* `len(_descriptors) <= max_descriptors` holds at every observation point;
* every file's final on-disk bytes match the deterministic concatenation of
  what was written;
* no asyncio tasks are left running.

See final-test-review.md §7 #6 / upstream §5 #1.
"""

import asyncio

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from aiofilepool import FilePool


pytestmark = [pytest.mark.asyncio, pytest.mark.slow]


@settings(
    max_examples=20,
    deadline=None,  # async + filesystem ops are slow under Hypothesis
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)
@given(
    cap=st.integers(min_value=1, max_value=4),
    handle_count=st.integers(min_value=4, max_value=12),
    op_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
async def test_capacity_invariant_holds_under_random_ops(
    tmp_path_factory, cap: int, handle_count: int, op_seed: int
) -> None:
    import random

    rng = random.Random(op_seed)
    base = tmp_path_factory.mktemp(f"pool-{cap}-{handle_count}-{op_seed}")
    paths = [base / f"f{i}.bin" for i in range(handle_count)]

    expected: list[bytearray] = [bytearray() for _ in range(handle_count)]

    async with FilePool(
        descriptor_pool_size=cap,
        thread_pool_size=0,
        chunking_threshold=1024,
    ) as pool:
        handles = [await pool.open(path, "w+b") for path in paths]
        max_observed = 0

        # Each handle writes 8 random ops worth of bytes.
        ops_per_handle = 8
        ops: list[tuple[int, bytes]] = []
        for i in range(handle_count):
            for _ in range(ops_per_handle):
                payload = bytes(rng.randint(0, 255) for _ in range(rng.randint(1, 16)))
                ops.append((i, payload))
        rng.shuffle(ops)

        for handle_index, payload in ops:
            await handles[handle_index].write(payload)
            expected[handle_index].extend(payload)
            current = len(pool._fd_manager._descriptors)  # noqa: SLF001
            assert current <= cap, f"FD pool exceeded cap: {current} > {cap}"
            max_observed = max(max_observed, current)

        # Read back. Every file must equal the concatenation of its writes.
        for i, handle in enumerate(handles):
            await handle.seek(0)
            data = await handle.read()
            assert bytes(data) == bytes(expected[i]), (
                f"file {i} content mismatch: got {data!r}, expected {bytes(expected[i])!r}"
            )

        for handle in handles:
            await handle.close()

        # Pool reaches at most `cap` open descriptors during the run.
        assert max_observed <= cap

    # After pool close, no leftover asyncio tasks owned by this test.
    leaked = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]
    assert not leaked, f"asyncio task leak after FilePool close: {leaked}"
