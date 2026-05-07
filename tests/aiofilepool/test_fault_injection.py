"""Realistic OS-error fault-injection for `aiofilepool`.

The existing fault-injection tests in `test_fd_lifecycle.py` and
`test_async_binary_io_contract.py` use `RuntimeError` to drive failure paths.
Real production failure modes are `OSError(ENOSPC)`, `PermissionError`, and
`FileNotFoundError`. This file pins the pool's behavior under each.

See final-test-review.md §7 #11.
"""

import asyncio
import errno
import io
from collections import defaultdict
from collections.abc import Buffer

import pytest

from aiofilepool import FilePool


pytestmark = pytest.mark.asyncio


class _OSErrorIO(io.BytesIO):
    """BytesIO that raises a chosen OSError on a chosen operation/call."""

    def __init__(
        self,
        data: bytes = b"",
        *,
        fail_on: str = "write",
        on_call: int = 1,
        exc_factory=lambda: OSError(errno.ENOSPC, "No space left on device"),
    ) -> None:
        super().__init__(data)
        self.calls: dict[str, int] = defaultdict(int)
        self._fail_on = fail_on
        self._on_call = on_call
        self._exc_factory = exc_factory

    def _maybe_fail(self, op: str) -> None:
        self.calls[op] += 1
        if op == self._fail_on and self.calls[op] == self._on_call:
            raise self._exc_factory()

    def read(self, size: int | None = -1) -> bytes:
        self._maybe_fail("read")
        return super().read(size)

    def write(self, data: bytes | Buffer) -> int:
        self._maybe_fail("write")
        return super().write(data)

    def flush(self) -> None:
        self._maybe_fail("flush")

    def close(self) -> None:
        self._maybe_fail("close")
        super().close()


async def test_enospc_during_write_propagates_without_leaking_fd(tmp_path) -> None:
    """A write that fails with ENOSPC must surface the OSError to the caller
    and leave the pool's descriptor count consistent. The handle remains
    usable for close()."""
    failing = _OSErrorIO(
        b"",
        fail_on="write",
        on_call=1,
        exc_factory=lambda: OSError(errno.ENOSPC, "No space left on device"),
    )

    async with FilePool(
        descriptor_pool_size=2, thread_pool_size=0, chunking_threshold=1024
    ) as pool:
        adapter = await pool.manage(failing)
        with pytest.raises(OSError) as exc_info:
            await adapter.write(b"data")
        assert exc_info.value.errno == errno.ENOSPC
        # Pool should not have leaked a slot on this failure.
        assert len(pool._fd_manager._descriptors) <= 1  # noqa: SLF001
        await adapter.close()


async def test_permission_error_during_reopen_releases_slot(
    tmp_path, monkeypatch
) -> None:
    """If `open()` raises PermissionError while re-opening a cooled handle,
    the slot must be released so subsequent acquires can proceed.
    """
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    path_a.write_bytes(b"")
    path_b.write_bytes(b"")

    async with FilePool(
        descriptor_pool_size=1, thread_pool_size=0, chunking_threshold=1024
    ) as pool:
        handle_a = await pool.open(path_a, "r+b")
        handle_b = await pool.open(path_b, "r+b")
        # Use handle_a → it occupies the only slot, then is cooled.
        await handle_a.write(b"alpha")
        # Force handle_a to be evicted: poke handle_b which needs the slot.
        # Inject PermissionError on the next open() inside _fd_manager.
        original_open = open
        call_count = {"n": 0}

        def _failing_open(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise PermissionError(
                    errno.EACCES, "Permission denied (injected)", str(args[0])
                )
            return original_open(*args, **kwargs)

        monkeypatch.setattr(
            "aiofilepool._fd_manager.open", _failing_open, raising=False
        )

        with pytest.raises(PermissionError):
            await handle_b.write(b"beta")

        # After the permission failure, the pool's slot accounting must be
        # consistent — a subsequent operation on handle_a should succeed.
        # Disable the injection.
        monkeypatch.setattr(
            "aiofilepool._fd_manager.open", original_open, raising=False
        )
        await handle_a.write(b"second-write")
        await handle_a.close()
        await handle_b.close()


async def test_filenotfounderror_on_initial_open_propagates(tmp_path) -> None:
    """Opening a path that does not exist for read raises FileNotFoundError on
    handle initialization (the size stat fails first)."""
    missing = tmp_path / "does-not-exist.bin"

    async with FilePool(
        descriptor_pool_size=2, thread_pool_size=0, chunking_threshold=1024
    ) as pool:
        with pytest.raises(FileNotFoundError):
            await pool.open(missing, "rb")
        # Pool must remain usable after a failed open.
        existing = tmp_path / "existing.bin"
        existing.write_bytes(b"hello")
        handle = await pool.open(existing, "rb")
        data = await handle.read()
        assert data == b"hello"
        await handle.close()


async def test_concurrent_acquire_after_failed_open_does_not_deadlock(
    tmp_path, monkeypatch
) -> None:
    """Two concurrent tasks; the first triggers an open() failure that, if not
    cleaned up, would leave the slot count short and deadlock the second task.
    """
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    path_a.write_bytes(b"alpha")
    path_b.write_bytes(b"beta")

    async with FilePool(
        descriptor_pool_size=1, thread_pool_size=0, chunking_threshold=1024
    ) as pool:
        original_open = open
        call_count = {"n": 0}

        def _flaky_open(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError(errno.EIO, "I/O error (injected)", str(args[0]))
            return original_open(*args, **kwargs)

        monkeypatch.setattr("aiofilepool._fd_manager.open", _flaky_open, raising=False)

        async def _try_open_a():
            handle = await pool.open(path_a, "rb")
            await handle.read()
            await handle.close()

        async def _try_open_b():
            handle = await pool.open(path_b, "rb")
            data = await handle.read()
            await handle.close()
            return data

        # First attempt raises (open() injected to fail on call #1).
        with pytest.raises(OSError):
            await _try_open_a()

        # Second task: must complete within a small timeout (no deadlock).
        result = await asyncio.wait_for(_try_open_b(), timeout=2.0)
        assert result == b"beta"
