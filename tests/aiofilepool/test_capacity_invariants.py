"""Capacity-bound and slot-accounting invariants under random concurrent operations.

Mixed-op generation (write / read / close-and-reopen) with structural invariants
on the `FileDescriptorManager`'s internal state — not just the descriptor count,
but the relationships between `_descriptors`, `_cold_handles`, `_slots`, and each
handle's `_fd_materialized` flag.

The pool's defining contract: at most `cap` OS descriptors are open simultaneously,
even when the caller holds `>> cap` logical handles.
"""

import asyncio
import random

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from aiofilepool import FilePool

from .conftest import count_fd


pytestmark = [pytest.mark.asyncio, pytest.mark.slow]


def _assert_manager_invariants(pool: FilePool, cap: int) -> None:
    """Check every structural invariant the FileDescriptorManager promises."""
    mgr = pool._fd_manager  # noqa: SLF001
    descriptors_keys = set(mgr._descriptors.keys())  # noqa: SLF001
    cold = mgr._cold_handles  # noqa: SLF001
    slots = mgr._slots  # noqa: SLF001

    assert len(descriptors_keys) <= cap, (
        f"descriptor count {len(descriptors_keys)} > cap {cap}"
    )
    assert 0 <= slots <= cap, f"slots {slots} out of bounds [0, {cap}]"
    assert cold.issubset(descriptors_keys), (
        f"cold {cold} not subset of descriptors {descriptors_keys}"
    )
    assert all(h._fd_materialized for h in descriptors_keys), (  # noqa: SLF001
        f"open descriptor for never-materialized handle in {descriptors_keys}"
    )


@settings(
    max_examples=30,
    deadline=None,
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
async def test_capacity_and_structural_invariants_hold_under_random_ops(
    tmp_path_factory, cap: int, handle_count: int, op_seed: int
) -> None:
    """Random `{write, read, reopen}` op stream — every observation point must
    satisfy the manager's structural invariants, and every file's bytes match
    the deterministic concatenation of its writes."""
    rng = random.Random(op_seed)
    base = tmp_path_factory.mktemp(f"cap-{cap}-{handle_count}-{op_seed}")
    paths = [base / f"f{i}.bin" for i in range(handle_count)]
    for p in paths:
        p.touch()

    expected: list[bytearray] = [bytearray() for _ in range(handle_count)]
    pre_task_count = sum(1 for t in asyncio.all_tasks() if not t.done())

    async with FilePool(
        descriptor_pool_size=cap,
        thread_pool_size=0,
        chunking_threshold=1024,
    ) as pool:
        handles = [await pool.open(p, "w+b") for p in paths]

        ops_per_handle = 6
        op_kinds = ("write", "read", "reopen")
        op_log: list[tuple[str, int, object]] = []
        for i in range(handle_count):
            for _ in range(ops_per_handle):
                kind = rng.choice(op_kinds)
                if kind == "write":
                    chunk = bytes(
                        rng.randint(0, 255) for _ in range(rng.randint(1, 16))
                    )
                    op_log.append(("write", i, chunk))
                elif kind == "read":
                    op_log.append(("read", i, None))
                else:
                    op_log.append(("reopen", i, None))
        rng.shuffle(op_log)

        for kind, idx, payload in op_log:
            handle = handles[idx]
            if kind == "write":
                assert isinstance(payload, bytes)
                await handle.write(payload)
                expected[idx].extend(payload)
            elif kind == "read":
                # Read full file, then restore position to end-of-data so subsequent
                # writes continue to append correctly.
                _ = await handle.read(offset=0)
                await handle.seek(len(expected[idx]))
            else:
                # Reopen: close current handle and replace with a fresh r+b on same path.
                await handle.close()
                handles[idx] = await pool.open(paths[idx], "r+b")
                await handles[idx].seek(len(expected[idx]))

            _assert_manager_invariants(pool, cap)

        for i, handle in enumerate(handles):
            await handle.seek(0)
            got = await handle.read()
            assert bytes(got) == bytes(expected[i]), f"file {i} content mismatch"

        for handle in handles:
            await handle.close()

        _assert_manager_invariants(pool, cap)

    # No asyncio tasks owned by this test leaked.
    post_task_count = sum(1 for t in asyncio.all_tasks() if not t.done())
    assert post_task_count <= pre_task_count + 1  # at most the current test task


@settings(
    max_examples=20,
    deadline=None,
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
async def test_capacity_invariant_under_concurrent_writes(
    tmp_path_factory, cap: int, handle_count: int, op_seed: int
) -> None:
    """Concurrent writes (gathered via `asyncio.gather`) must respect the cap at
    every observation. The post-condition is the deterministic concatenation:
    each handle gets its own list of writes assigned, executed concurrently
    across all handles."""
    rng = random.Random(op_seed)
    base = tmp_path_factory.mktemp(f"cap-conc-{cap}-{handle_count}-{op_seed}")
    paths = [base / f"f{i}.bin" for i in range(handle_count)]
    for p in paths:
        p.touch()

    per_handle_payloads: list[list[bytes]] = []
    for _ in range(handle_count):
        n = rng.randint(1, 4)
        chunks = [
            bytes(rng.randint(0, 255) for _ in range(rng.randint(1, 12)))
            for _ in range(n)
        ]
        per_handle_payloads.append(chunks)

    async with FilePool(
        descriptor_pool_size=cap,
        thread_pool_size=0,
        chunking_threshold=1024,
    ) as pool:
        handles = [await pool.open(p, "w+b") for p in paths]
        max_observed = 0

        async def write_all(handle, chunks):
            nonlocal max_observed
            for chunk in chunks:
                await handle.write(chunk)
                max_observed = max(max_observed, count_fd(pool))

        await asyncio.gather(
            *(write_all(h, c) for h, c in zip(handles, per_handle_payloads))
        )

        assert max_observed <= cap
        _assert_manager_invariants(pool, cap)

        for i, handle in enumerate(handles):
            await handle.seek(0)
            data = await handle.read()
            assert bytes(data) == b"".join(per_handle_payloads[i])

        for handle in handles:
            await handle.close()

        assert count_fd(pool) == 0


async def test_descriptor_count_returns_to_zero_after_clean_close(
    pool_factory, file_writer
) -> None:
    """A clean run that closes every handle leaves the descriptor map empty."""
    paths = [file_writer(f"clean-{i}.bin", b"") for i in range(8)]

    async with pool_factory(descriptor_pool_size=2, thread_pool_size=0) as pool:
        handles = [await pool.open(p, "w+") for p in paths]
        for h in handles:
            await h.write(b"x")
        for h in handles:
            await h.close()

        assert count_fd(pool) == 0
        # No zombie cold-handle records remain.
        assert len(pool._fd_manager._cold_handles) == 0
