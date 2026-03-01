from io import IOBase
from aiofilepool._handle import FileHandle


class FileDescriptorManager:
    def __init__(self, max_descriptors: int):
        assert max_descriptors > 0, "max_descriptors must be >= 1"

        self._max_descriptors = max_descriptors
        self._descriptors: dict[FileHandle, IOBase] = {}
        self._cold_handles: set[FileHandle] = set()
        self._inactive_handles: set[FileHandle] = set()

    def _ensure_slot(self):
        if len(self._descriptors) < self._max_descriptors:
            return

        if len(self._cold_handles) > 0:
            handle = self._cold_handles.pop()
            fd = self._descriptors.pop(handle)
            fd.flush()
            fd.close()
            self._inactive_handles.add(handle)

    def _restore_descriptor(self, handle: FileHandle) -> IOBase | None:
        if handle in self._descriptors:
            self._cold_handles.discard(handle)
            return self._descriptors[handle]

        if handle not in self._inactive_handles:
            return None

        self._ensure_slot()
        self._inactive_handles.discard(handle)
        self._descriptors[handle] = open(handle._path, handle._mode.renewal_mode)
        return self._descriptors[handle]

    def _open_descriptor(self, handle: FileHandle) -> IOBase:
        self._ensure_slot()
        fd = open(handle._path, handle._mode.mode)
        self._descriptors[handle] = fd
        return fd

    def acquire(self, handle: FileHandle) -> IOBase:
        return self._restore_descriptor(handle) or self._open_descriptor(handle)

    def release(self, handle: FileHandle) -> None:
        self._cold_handles.add(handle)

    def close(self, handle: FileHandle) -> None:
        if handle not in self._descriptors:
            return

        fd = self._descriptors.pop(handle)
        fd.flush()
        fd.close()
        self._cold_handles.discard(handle)

    def close_all(self) -> None:
        for handle in self._descriptors:
            self.close(handle)
