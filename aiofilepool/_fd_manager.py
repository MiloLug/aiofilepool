import asyncio
from io import IOBase
from aiofilepool._handle import FileHandle


class FileDescriptorManager:
    def __init__(self, max_descriptors: int):
        assert max_descriptors > 0, "max_descriptors must be >= 1"

        self._max_descriptors = max_descriptors
        self._descriptors: dict[FileHandle, IOBase] = {}
        self._cold_handles: set[FileHandle] = set()
        self._slots_cond = asyncio.Condition()
        self._slots = max_descriptors
        self._inactive_handles: set[FileHandle] = set()

    def _deactivate_handle(self, handle: FileHandle):
        self._cold_handles.discard(handle)
        fd = self._descriptors.pop(handle)
        fd.flush()
        fd.close()
        self._inactive_handles.add(handle)

    def _activate_handle(self, handle: FileHandle) -> IOBase:
        self._inactive_handles.discard(handle)
        self._descriptors[handle] = open(handle._path, handle._mode.renewal_mode)
        return self._descriptors[handle]

    async def _ensure_slot(self):
        async with self._slots_cond:
            if self._slots < 1:
                await self._slots_cond.wait_for(lambda: self._slots > 0)
            self._slots -= 1

        if len(self._descriptors) < self._max_descriptors:
            return

        assert len(self._cold_handles) > 0, (
            "Impossible state: no cold handles but there are slots and max descriptors"
        )
        handle = self._cold_handles.pop()
        self._deactivate_handle(handle)

    async def _restore_descriptor(self, handle: FileHandle) -> IOBase | None:
        if handle in self._descriptors:
            if handle in self._cold_handles:
                self._cold_handles.discard(handle)
                async with self._slots_cond:
                    self._slots -= 1
            return self._descriptors[handle]

        if handle not in self._inactive_handles:
            return None

        await self._ensure_slot()
        return self._activate_handle(handle)

    async def _open_descriptor(self, handle: FileHandle) -> IOBase:
        await self._ensure_slot()
        fd = open(handle._path, handle._mode.mode)
        self._descriptors[handle] = fd
        return fd

    async def acquire(self, handle: FileHandle) -> IOBase:
        return await self._restore_descriptor(handle) or await self._open_descriptor(
            handle
        )

    async def release(self, handle: FileHandle) -> None:
        self._cold_handles.add(handle)
        async with self._slots_cond:
            self._slots += 1
            self._slots_cond.notify()

    async def close(self, handle: FileHandle) -> None:
        if handle not in self._descriptors:
            self._inactive_handles.discard(handle)
            return

        fd = self._descriptors.pop(handle)
        fd.flush()
        fd.close()
        self._cold_handles.discard(handle)
        async with self._slots_cond:
            self._slots += 1
            self._slots_cond.notify()

    async def close_all(self) -> None:
        for fd in self._descriptors.values():
            fd.flush()
            fd.close()

        self._descriptors.clear()
        self._cold_handles.clear()
        self._inactive_handles.clear()
        async with self._slots_cond:
            self._slots = self._max_descriptors
            self._slots_cond.notify(self._slots)
