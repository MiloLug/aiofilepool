"""Descriptor bookkeeping and eviction for the async file pool."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from io import IOBase

from ._errors import HandleClosedError, PoolClosedError, PoolStateError


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
    last_used: float = field(default_factory=time.monotonic)


class DescriptorManager:
    """Manages a bounded set of real file descriptors with LRU eviction."""

    def __init__(self, max_descriptors: int) -> None:
        if max_descriptors < 1:
            raise ValueError("max_descriptors must be >= 1")
        self._max_descriptors = max_descriptors
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

    async def acquire(self, handle_id: int) -> IOBase:
        """
        Acquire an active descriptor for the given handle.

        This method lazily opens and potentially evicts other descriptors while
        respecting the configured max descriptor limit.
        """
        async with self._condition:
            while True:
                if self._closed:
                    raise PoolClosedError("descriptor manager is closed")

                record = self._records.get(handle_id)
                if record is None:
                    raise HandleClosedError(f"handle {handle_id} is not registered")

                # Fast path: descriptor already open and idle.
                if record.fd is not None and not record.active:
                    record.active = True
                    record.last_used = time.monotonic()
                    return record.fd

                # Need to open/reopen descriptor.
                if record.fd is None:
                    if self._open_count < self._max_descriptors:
                        record.fd = self._open_record_locked(record)
                        self._open_count += 1
                        record.active = True
                        record.last_used = time.monotonic()
                        return record.fd

                    # Descriptor budget is full: evict one idle record if possible.
                    if self._evict_one_locked():
                        continue

                # All descriptors are active; wait until one is released.
                await self._condition.wait()

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
            self._condition.notify_all()

    async def unregister(self, handle_id: int) -> None:
        """Unregister a handle and close its descriptor (if open)."""
        async with self._condition:
            record = self._records.get(handle_id)
            if record is None:
                return

            while record.active:
                await self._condition.wait()
                record = self._records.get(handle_id)
                if record is None:
                    return

            removed = self._records.pop(handle_id, None)
            if removed is not None and removed.fd is not None:
                self._close_record_locked(removed)
            self._condition.notify_all()

    async def close_all(self) -> None:
        """Close all open descriptors and clear all records."""
        async with self._condition:
            self._closed = True
            while any(record.active for record in self._records.values()):
                await self._condition.wait()

            for record in list(self._records.values()):
                if record.fd is not None:
                    self._close_record_locked(record)
            self._records.clear()
            self._condition.notify_all()

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

    def _open_record_locked(self, record: DescriptorRecord) -> IOBase:
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

    def _evict_one_locked(self) -> bool:
        candidates = [
            record
            for record in self._records.values()
            if record.fd is not None and not record.active
        ]
        if not candidates:
            return False

        clean = [record for record in candidates if not record.dirty]
        if clean:
            victim = min(clean, key=lambda item: item.last_used)
        else:
            victim = min(candidates, key=lambda item: item.last_used)
            try:
                victim.fd.flush()  # type: ignore[union-attr]
            finally:
                victim.dirty = False

        self._close_record_locked(victim)
        victim.fd = None
        victim.active = False
        return True

    def _close_record_locked(self, record: DescriptorRecord) -> None:
        if record.fd is None:
            return
        try:
            record.fd.close()
        finally:
            record.fd = None
            self._open_count -= 1
            if self._open_count < 0:
                raise PoolStateError("descriptor open_count became negative")
