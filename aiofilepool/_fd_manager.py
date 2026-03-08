import asyncio
from contextlib import asynccontextmanager
from typing import IO, AsyncGenerator
from aiofilepool._handle import FileHandle
from aiofilepool.errors import FilePoolNotOpenError


class FileDescriptorManager:
    def __init__(self, max_descriptors: int):
        if max_descriptors < 1:
            raise ValueError("max_descriptors must be >= 1")

        self._max_descriptors = max_descriptors
        self._descriptors: dict[FileHandle, IO] = {}
        self._cold_handles: set[FileHandle] = set()
        self._slots_cond = asyncio.Condition()
        self._slots = max_descriptors
        self._inactive_handles: set[FileHandle] = set()
        self._closed = False

    async def _release_slot(self):
        async with self._slots_cond:
            self._slots += 1
            assert self._slots <= self._max_descriptors, (
                "Impossible state: slots cannot exceed max descriptors"
            )
            self._slots_cond.notify()

    async def _ensure_slot(self):
        async with self._slots_cond:
            if self._slots < 1:
                await self._slots_cond.wait_for(lambda: self._closed or self._slots > 0)
            if self._closed:
                raise FilePoolNotOpenError()
            self._slots -= 1

        if len(self._descriptors) < self._max_descriptors:
            return

        assert len(self._cold_handles) > 0, (
            "Impossible state: no cold handles but there are slots and max descriptors"
        )

        handle = self._cold_handles.pop()
        fd = self._descriptors[handle]
        close_error: BaseException | None = None
        try:
            fd.flush()
        except BaseException as exc:
            close_error = exc

        try:
            fd.close()
        except BaseException as exc:
            if close_error is None:
                close_error = exc

        self._descriptors.pop(handle, None)
        self._inactive_handles.add(handle)
        if close_error is not None:
            await self._release_slot()
            raise close_error

    async def _activate_handle(self, handle: FileHandle) -> IO:
        await self._ensure_slot()
        try:
            self._descriptors[handle] = open(handle._path, handle._mode.renewal_mode)
            self._inactive_handles.discard(handle)
        except BaseException:
            await self._release_slot()
            raise
        return self._descriptors[handle]

    async def _discard_handle(self, handle: FileHandle) -> None:
        if handle not in self._descriptors:
            self._inactive_handles.discard(handle)
            return
        try:
            fd = self._descriptors.pop(handle)
            fd.flush()
            fd.close()
        finally:
            if handle in self._cold_handles:
                self._cold_handles.discard(handle)
            else:
                await self._release_slot()

    async def _cool_handle(self, handle: FileHandle) -> None:
        self._cold_handles.add(handle)
        await self._release_slot()

    async def _reheat_handle(self, handle: FileHandle) -> None:
        self._cold_handles.remove(handle)
        async with self._slots_cond:
            self._slots -= 1
            assert self._slots >= 0, "Impossible state: slots cannot be negative"

    async def _restore_descriptor(self, handle: FileHandle) -> IO | None:
        if handle in self._descriptors:
            if handle in self._cold_handles:
                await self._reheat_handle(handle)
            return self._descriptors[handle]

        if handle in self._inactive_handles:
            return await self._activate_handle(handle)

        return None

    async def _open_descriptor(self, handle: FileHandle) -> IO:
        await self._ensure_slot()
        try:
            fd = open(handle._path, handle._mode.mode)
            self._descriptors[handle] = fd
            return fd
        except BaseException:
            await self._release_slot()
            raise

    @asynccontextmanager
    async def acquire(self, handle: FileHandle) -> AsyncGenerator[IO]:
        if self._closed:
            raise FilePoolNotOpenError()

        fd = await self._restore_descriptor(handle) or await self._open_descriptor(
            handle
        )
        try:
            yield fd
        finally:
            await self._release(handle)

    async def _release(self, handle: FileHandle) -> None:
        if self._closed:
            await self._discard_handle(handle)
        else:
            await self._cool_handle(handle)

    async def discard(self, handle: FileHandle) -> None:
        await self._discard_handle(handle)

    async def close(self, timeout: float | None = None) -> None:
        if self._closed:
            return

        async with self._slots_cond:
            self._closed = True
            self._slots_cond.notify_all()

        close_error: BaseException | None = None
        for handle in self._cold_handles.copy():
            try:
                await self._discard_handle(handle)
            except BaseException as exc:  # noqa: BLE001
                if close_error is None:
                    close_error = exc

        self._inactive_handles.clear()
        async with self._slots_cond:
            await asyncio.wait_for(
                self._slots_cond.wait_for(lambda: len(self._descriptors) == 0),
                timeout=timeout,
            )

        if close_error is not None:
            raise close_error
