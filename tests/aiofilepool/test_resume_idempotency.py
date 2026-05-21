"""Resume idempotency — mirrors `filepart.storage.download_manager.py` byte-for-byte.

Real download flow:
1. New download: `open("wb")` → `allocate(N)` → scatter writes → close.
2. Resume: `open("r+b")` → `allocate(N)` (idempotent — file already sized) → write remaining
   parts → close.

The pool MUST treat `allocate(length)` as a no-op when `length <= current size`, with
no dispatched blocking call (verified via `count_blocking_calls` spy). Partial-then-
resumed flows must produce the same final file as a one-shot scatter.
"""

import os

import pytest

from aiofilepool.errors import (
    InvalidFileModeError,
    InvalidPositionError,
    IONotOpenError,
)

from .conftest import count_blocking_calls


pytestmark = pytest.mark.asyncio


# --- allocate() basics --------------------------------------------------------


async def test_allocate_grows_empty_file_to_size_with_zero_padding(
    pool_factory, file_writer
) -> None:
    path = file_writer("alloc-grow.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.allocate(1024)
        await handle.close()

    assert path.stat().st_size == 1024
    assert path.read_bytes() == b"\x00" * 1024


async def test_allocate_updates_handle_size_and_keeps_position(
    pool_factory, file_writer
) -> None:
    path = file_writer("alloc-size.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.allocate(64)

        assert await handle.size() == 64
        assert await handle.read(offset=0) == b"\x00" * 64


async def test_allocate_rejects_read_only_mode(pool_factory, file_writer) -> None:
    path = file_writer("alloc-readonly.bin", b"data")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r")
        with pytest.raises(InvalidFileModeError, match="file is not writable"):
            await handle.allocate(100)


async def test_allocate_rejects_negative_length(pool_factory, file_writer) -> None:
    path = file_writer("alloc-negative.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        with pytest.raises(InvalidPositionError, match="length must be >= 0"):
            await handle.allocate(-1)


async def test_allocate_rejects_closed_handle(pool_factory, file_writer) -> None:
    path = file_writer("alloc-closed.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.close()

        with pytest.raises(IONotOpenError):
            await handle.allocate(8)


async def test_allocate_rejects_uninitialized_handle(pool_factory, file_writer) -> None:
    path = file_writer("alloc-uninit.bin", b"")

    async with pool_factory() as pool:
        handle = pool.open(path, "w+")
        with pytest.raises(IONotOpenError):
            await handle.allocate(8)


# --- Idempotence (the load-bearing resume invariant) --------------------------


async def test_allocate_is_noop_when_length_below_current_size_no_blocking_dispatch(
    pool_factory, file_writer, monkeypatch
) -> None:
    """The "resume" allocate is the same call as the initial allocate. The pool MUST
    short-circuit when the file already has at least that many bytes — no
    `posix_fallocate` / `truncate` dispatch, no fd acquisition."""
    path = file_writer("alloc-noop.bin", b"0123456789")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r+")

        # Start counting AFTER initialization (which dispatches fs.stat).
        recorded = count_blocking_calls(pool, monkeypatch)
        await handle.allocate(3)
        await handle.allocate(10)  # exactly current size — still no-op
        await handle.close()

    assert recorded == []
    assert path.stat().st_size == 10


async def test_allocate_dispatches_when_length_exceeds_current_size(
    pool_factory, file_writer, monkeypatch
) -> None:
    """Counterpart to the no-op test: when growth is needed, at least one blocking
    call (the `posix_fallocate` or `truncate` invocation) is dispatched."""
    path = file_writer("alloc-grow-dispatch.bin", b"abcd")

    async with pool_factory() as pool:
        handle = await pool.open(path, "r+")

        recorded = count_blocking_calls(pool, monkeypatch)
        await handle.allocate(128)
        await handle.close()

    # At least one of {posix_fallocate, truncate} was dispatched.
    assert any(name in {"posix_fallocate", "truncate"} for name in recorded)
    assert path.stat().st_size == 128


async def test_resume_idempotent_allocate_after_reopen_does_not_redispatch(
    pool_factory, file_writer, monkeypatch
) -> None:
    """The exact resume flow: open `wb`, allocate, close; reopen `r+b`, allocate
    same size, write at offset. Second allocate must not dispatch any blocking
    growth call."""
    path = file_writer("resume-noop.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "wb")
        await handle.allocate(256)
        await handle.close()

    assert path.stat().st_size == 256

    async with pool_factory() as pool:
        handle = await pool.open(path, "r+b")

        recorded = count_blocking_calls(pool, monkeypatch)
        await handle.allocate(256)
        # Now perform a single offset write — recorded will include the write dispatches.
        await handle.write(b"PAYLOAD", offset=10)
        await handle.close()

    # Allocate dispatched nothing; only the write produced dispatches.
    assert "posix_fallocate" not in recorded
    assert "truncate" not in recorded
    assert path.stat().st_size == 256


# --- End-to-end resume flow ---------------------------------------------------


async def test_partial_resume_write_remaining_parts_yields_complete_file(
    pool_factory, file_writer
) -> None:
    """Write half the parts, close, reopen `r+b`, write the rest. Final bytes
    must equal a one-shot scatter."""
    part_size = 64
    part_count = 8
    total = part_size * part_count
    parts = [bytes([i] * part_size) for i in range(part_count)]

    path = file_writer("resume-partial.bin", b"")

    # Phase 1: new download — write parts [0..4).
    async with pool_factory() as pool:
        handle = await pool.open(path, "wb")
        await handle.allocate(total)
        for i in range(part_count // 2):
            await handle.write(parts[i], offset=i * part_size)
        await handle.close()

    # File now has parts [0..4) populated and parts [4..8) as zero padding.
    after_phase_1 = path.read_bytes()
    expected_phase_1 = b"".join(parts[: part_count // 2]) + b"\x00" * (
        (part_count - part_count // 2) * part_size
    )
    assert after_phase_1 == expected_phase_1

    # Phase 2: resume — reopen, idempotent allocate, write parts [4..8).
    async with pool_factory() as pool:
        handle = await pool.open(path, "r+b")
        await handle.allocate(total)
        for i in range(part_count // 2, part_count):
            await handle.write(parts[i], offset=i * part_size)
        await handle.close()

    expected_final = b"".join(parts)
    assert path.read_bytes() == expected_final


async def test_resume_after_unclean_shutdown_recovers_with_r_plus_b(
    pool_factory, file_writer
) -> None:
    """Simulate an unclean shutdown: write some parts to an allocated file, then
    leak the handle (no `close()`). Resume must still work via `r+b`."""
    part_size = 16
    parts = [bytes([42] * part_size) for _ in range(4)]
    total = part_size * len(parts)

    path = file_writer("resume-unclean.bin", b"")

    async with pool_factory() as pool:
        handle = await pool.open(path, "wb")
        await handle.allocate(total)
        await handle.write(parts[0], offset=0)
        # Leak: do not close. Pool's __aexit__ will drain.

    async with pool_factory() as pool:
        handle = await pool.open(path, "r+b")
        await handle.allocate(total)
        for i in range(1, len(parts)):
            await handle.write(parts[i], offset=i * part_size)
        await handle.close()

    assert path.read_bytes() == b"".join(parts)


# --- Platform-branch coverage --------------------------------------------------


def _ensure_posix_fallocate(monkeypatch) -> None:
    if not hasattr(os, "posix_fallocate"):

        def posix_fallocate(fd, offset, length):
            os.ftruncate(fd, length)

        monkeypatch.setattr(os, "posix_fallocate", posix_fallocate, raising=False)


async def test_allocate_grows_file_under_posix_branch(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("alloc-posix.bin", b"")
    monkeypatch.setattr("aiofilepool._handle._IS_POSIX", True)
    _ensure_posix_fallocate(monkeypatch)

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.allocate(1024)
        await handle.close()

    assert path.stat().st_size == 1024


async def test_allocate_grows_file_under_windows_branch(
    pool_factory, file_writer, monkeypatch
) -> None:
    path = file_writer("alloc-win.bin", b"")
    monkeypatch.setattr("aiofilepool._handle._IS_POSIX", False)

    async with pool_factory() as pool:
        handle = await pool.open(path, "w+")
        await handle.allocate(2048)
        await handle.close()

    assert path.stat().st_size == 2048
