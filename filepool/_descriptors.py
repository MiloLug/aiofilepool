"""Descriptor bookkeeping and eviction for the async file pool."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from io import IOBase
from typing import Awaitable, Callable, TypeVar, cast

from ._errors import (
    DescriptorAcquireTimeoutError,
    HandleClosedError,
    PoolClosedError,
    PoolStateError,
)

T = TypeVar("T")
BlockingRunner = Callable[..., Awaitable[object]]


@dataclass(slots=True)
class DescriptorRecord:
    """Runtime metadata for one logical handle."""

    handle_id: int
    path: str
    mode: str
    binary: bool
    encoding: str | None
    errors: str | None
    newline: str | None
    position: int = 0
    fd: IOBase | None = None
    active: bool = False
    dirty: bool = False
    transitioning: bool = False
    last_used: float = field(default_factory=time.monotonic)


class DescriptorManager:
    """Manages a bounded set of real file descriptors with LRU eviction."""

    def __init__(
        self,
        max_descriptors: int,
        *,
        acquire_timeout: float | None = None,
        run_blocking: BlockingRunner | None = None,
    ) -> None:
        if max_descriptors < 1:
            raise ValueError("max_descriptors must be >= 1")
        if acquire_timeout is not None and acquire_timeout <= 0:
            raise ValueError("acquire_timeout must be > 0 or None")

        self._max_descriptors = max_descriptors
        self._acquire_timeout = acquire_timeout
        self._run_blocking = run_blocking

        self._records: dict[int, DescriptorRecord] = {}
        self._open_count = 0
        self._closed = False
        self._condition = asyncio.Condition()

    async def register(
        self,
        *,
        handle_id: int,
        path: str,
        mode: str,
        binary: bool,
        encoding: str | None,
        errors: str | None,
        newline: str | None,
        initial_position: int,
    ) -> None:
        """Register logical handle metadata (without opening the real fd yet)."""
        async with self._condition:
            if self._closed:
                raise PoolClosedError("descriptor manager is closed")
            if handle_id in self._records:
                raise PoolStateError(f"duplicate handle_id registration: {handle_id}")
            self._records[handle_id] = DescriptorRecord(
                handle_id=handle_id,
                path=path,
                mode=mode,
                binary=binary,
                encoding=encoding,
                errors=errors,
                newline=newline,
                position=initial_position,
            )
            self._condition.notify_all()

    async def acquire(self, handle_id: int) -> IOBase:
        """
        Acquire an active descriptor for the given handle.

        This method lazily opens and evicts descriptors while respecting the
        max descriptor budget. Slow filesystem work happens outside locks.
        """
        deadline = (
            time.monotonic() + self._acquire_timeout
            if self._acquire_timeout is not None
            else None
        )
        while True:
            action: tuple[str, DescriptorRecord] | None = None
            async with self._condition:
                if self._closed:
                    raise PoolClosedError("descriptor manager is closed")

                record = self._records.get(handle_id)
                if record is None:
                    raise HandleClosedError(f"handle {handle_id} is not registered")

                if (
                    record.fd is not None
                    and not record.active
                    and not record.transitioning
                ):
                    record.active = True
                    record.last_used = time.monotonic()
                    return record.fd

                if record.fd is None and not record.active and not record.transitioning:
                    if self._open_count < self._max_descriptors:
                        record.transitioning = True
                        self._open_count += 1
                        action = ("open", record)
                    else:
                        victim = self._select_eviction_candidate_locked()
                        if victim is not None:
                            victim.transitioning = True
                            action = ("evict", victim)

                if action is None:
                    await self._wait_for_capacity_locked(handle_id, deadline)
                    continue

            if action[0] == "open":
                return await self._open_reserved_record(handle_id, action[1])

            await self._evict_reserved_record(action[1])

    async def release(
        self,
        handle_id: int,
        *,
        dirty: bool | None = None,
        position: int | None = None,
    ) -> None:
        """Release a previously acquired descriptor and update metadata."""
        async with self._condition:
            record = self._records.get(handle_id)
            if record is None:
                return

            if position is not None:
                record.position = position
            if dirty is True:
                record.dirty = True
            elif dirty is False:
                record.dirty = False

            record.active = False
            record.last_used = time.monotonic()
            self._condition.notify(1)

    async def unregister(self, handle_id: int) -> None:
        """Unregister a handle and close its descriptor (if open)."""
        fd_to_close: IOBase | None = None
        async with self._condition:
            while True:
                record = self._records.get(handle_id)
                if record is None:
                    return

                if record.active or record.transitioning:
                    await self._condition.wait()
                    continue

                removed = self._records.pop(handle_id)
                fd_to_close = removed.fd
                removed.fd = None
                break

        close_error: BaseException | None = None
        if fd_to_close is not None:
            try:
                await self._close_fd(fd_to_close)
            except BaseException as exc:  # noqa: BLE001
                close_error = exc

        state_error: PoolStateError | None = None
        async with self._condition:
            if fd_to_close is not None:
                state_error = self._decrement_open_count_locked()
            self._condition.notify_all()

        if close_error is not None:
            raise close_error
        if state_error is not None:
            raise state_error

    async def close_all(self) -> None:
        """Close all open descriptors and clear all records."""
        to_close: list[IOBase] = []
        async with self._condition:
            self._closed = True
            while any(
                record.active or record.transitioning
                for record in self._records.values()
            ):
                await self._condition.wait()

            for record in self._records.values():
                if record.fd is not None:
                    to_close.append(record.fd)
                    record.fd = None
            self._records.clear()
            self._condition.notify_all()

        close_error: BaseException | None = None
        for fd in to_close:
            try:
                await self._close_fd(fd)
            except BaseException as exc:  # noqa: BLE001
                if close_error is None:
                    close_error = exc

        async with self._condition:
            self._open_count = 0
            self._condition.notify_all()

        if close_error is not None:
            raise close_error

    async def stats(self) -> dict[str, int]:
        """Return simple runtime counts useful for testing and observability."""
        async with self._condition:
            open_descriptors = sum(
                1 for record in self._records.values() if record.fd is not None
            )
            active_descriptors = sum(
                1 for record in self._records.values() if record.active
            )
            return {
                "registered_handles": len(self._records),
                "open_descriptors": open_descriptors,
                "active_descriptors": active_descriptors,
                "max_descriptors": self._max_descriptors,
            }

    async def _open_reserved_record(
        self,
        expected_handle_id: int,
        record: DescriptorRecord,
    ) -> IOBase:
        try:
            fd = await self._run_blocking_io(self._open_record_io, record)
        except BaseException:  # noqa: BLE001
            state_error: PoolStateError | None = None
            async with self._condition:
                record.transitioning = False
                state_error = self._decrement_open_count_locked()
                self._condition.notify_all()
            if state_error is not None:
                raise state_error
            raise

        fd_to_close: IOBase | None = None
        failure: BaseException | None = None
        state_error: PoolStateError | None = None

        async with self._condition:
            current = self._records.get(expected_handle_id)
            if self._closed:
                record.transitioning = False
                state_error = self._decrement_open_count_locked()
                self._condition.notify_all()
                fd_to_close = fd
                failure = PoolClosedError("descriptor manager is closed")
            elif current is not record:
                record.transitioning = False
                state_error = self._decrement_open_count_locked()
                self._condition.notify_all()
                fd_to_close = fd
                failure = HandleClosedError(
                    f"handle {expected_handle_id} is not registered"
                )
            else:
                record.fd = fd
                record.active = True
                record.transitioning = False
                record.last_used = time.monotonic()
                self._condition.notify(1)
                return fd

        close_error: BaseException | None = None
        if fd_to_close is not None:
            try:
                await self._close_fd(fd_to_close)
            except BaseException as exc:  # noqa: BLE001
                close_error = exc

        if failure is not None:
            if close_error is not None and hasattr(failure, "add_note"):
                failure.add_note(f"cleanup close failed: {close_error!r}")  # type: ignore[attr-defined]
            raise failure
        if close_error is not None:
            raise close_error
        if state_error is not None:
            raise state_error
        raise PoolStateError("unexpected descriptor open state")

    async def _evict_reserved_record(self, record: DescriptorRecord) -> None:
        fd = record.fd
        if fd is None:
            async with self._condition:
                record.transitioning = False
                self._condition.notify_all()
            return

        flush_error: BaseException | None = None
        close_error: BaseException | None = None
        if record.dirty:
            try:
                await self._flush_fd(fd)
            except BaseException as exc:  # noqa: BLE001
                flush_error = exc
        if flush_error is None:
            try:
                await self._close_fd(fd)
            except BaseException as exc:  # noqa: BLE001
                close_error = exc

        state_error: PoolStateError | None = None
        async with self._condition:
            record.transitioning = False
            if flush_error is None and close_error is None:
                record.fd = None
                record.active = False
                record.dirty = False
                state_error = self._decrement_open_count_locked()
            elif flush_error is not None:
                record.dirty = True
            else:
                record.fd = None
                record.active = False
                state_error = self._decrement_open_count_locked()
            self._condition.notify_all()

        if flush_error is not None:
            raise flush_error
        if close_error is not None:
            raise close_error
        if state_error is not None:
            raise state_error

    async def _wait_for_capacity_locked(
        self,
        handle_id: int,
        deadline: float | None,
    ) -> None:
        if deadline is None:
            await self._condition.wait()
            return

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DescriptorAcquireTimeoutError(
                f"timed out acquiring descriptor for handle {handle_id}"
            )

        try:
            await asyncio.wait_for(self._condition.wait(), timeout=remaining)
        except TimeoutError as exc:
            raise DescriptorAcquireTimeoutError(
                f"timed out acquiring descriptor for handle {handle_id}"
            ) from exc

    def _select_eviction_candidate_locked(self) -> DescriptorRecord | None:
        candidates = [
            record
            for record in self._records.values()
            if record.fd is not None and not record.active and not record.transitioning
        ]
        if not candidates:
            return None

        clean = [record for record in candidates if not record.dirty]
        if clean:
            return min(clean, key=lambda item: item.last_used)
        return min(candidates, key=lambda item: item.last_used)

    async def _run_blocking_io(self, func: Callable[..., T], *args: object) -> T:
        if self._run_blocking is None:
            return func(*args)
        result = await self._run_blocking(func, *args)
        return cast(T, result)

    async def _flush_fd(self, fd: IOBase) -> None:
        await self._run_blocking_io(self._flush_sync, fd)

    async def _close_fd(self, fd: IOBase) -> None:
        await self._run_blocking_io(self._close_sync, fd)

    def _open_record_io(self, record: DescriptorRecord) -> IOBase:
        if record.binary:
            fd = open(record.path, record.mode)
        else:
            fd = open(
                record.path,
                record.mode,
                encoding=record.encoding,
                errors=record.errors,
                newline=record.newline,
            )

        try:
            fd.seek(record.position)
        except Exception:
            fd.close()
            raise
        return fd

    @staticmethod
    def _flush_sync(fd: IOBase) -> None:
        fd.flush()

    @staticmethod
    def _close_sync(fd: IOBase) -> None:
        fd.close()

    def _decrement_open_count_locked(self) -> PoolStateError | None:
        self._open_count -= 1
        if self._open_count < 0:
            self._open_count = 0
            return PoolStateError("descriptor open_count became negative")
        return None
