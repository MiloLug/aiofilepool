"""High-level FilePool API and lifecycle management."""

from __future__ import annotations

import asyncio
import functools
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from io import IOBase
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from ._descriptors import DescriptorManager
from ._errors import PoolClosedError, PoolStateError
from ._modes import ModeSpec, parse_mode, validate_open_kwargs
from ._open_request import OpenRequest

T = TypeVar("T")

if TYPE_CHECKING:
    from ._handle import FileHandle


class FilePool:
    """Async file pool with bounded real descriptors and optional thread workers."""

    def __init__(
        self,
        descriptor_pool_size: int = 256,
        thread_pool_size: int = 4,
        chunk_size: int = 64 * 1024,
        fsync_on_write: bool = False,
        descriptor_acquire_timeout: float | None = None,
    ) -> None:
        if descriptor_pool_size < 1:
            raise ValueError("descriptor_pool_size must be >= 1")
        if thread_pool_size < 0:
            raise ValueError("thread_pool_size must be >= 0")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if descriptor_acquire_timeout is not None and descriptor_acquire_timeout <= 0:
            raise ValueError("descriptor_acquire_timeout must be > 0 or None")

        self._descriptor_pool_size = descriptor_pool_size
        self._thread_pool_size = thread_pool_size
        self._chunk_size = chunk_size
        self._fsync_on_write = fsync_on_write
        self._descriptor_acquire_timeout = descriptor_acquire_timeout

        self._executor = (
            ThreadPoolExecutor(
                max_workers=thread_pool_size, thread_name_prefix="filepool-io"
            )
            if thread_pool_size > 0
            else None
        )
        self._manager = DescriptorManager(
            max_descriptors=descriptor_pool_size,
            acquire_timeout=descriptor_acquire_timeout,
            run_blocking=self._run_blocking,
        )

        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._next_handle_id = 1

        self._ops_condition = asyncio.Condition()
        self._active_operations = 0

        self._bg_tasks: set[asyncio.Task[None]] = set()
        self._live_handles: weakref.WeakSet[FileHandle] = weakref.WeakSet()

    def open(
        self,
        path: str | os.PathLike[str],
        mode: str = "rb",
        *,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> OpenRequest:
        """Create a one-shot request that can be awaited or used in async-with."""
        return OpenRequest(
            self,
            path=os.fspath(path),
            mode=mode,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    async def close(self) -> None:
        """Close pool, waiting for in-flight work and cleaning resources safely."""
        self._bind_loop()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_internal())

        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await self._close_task
            raise

    async def __aenter__(self) -> "FilePool":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _open_handle(
        self,
        path: str,
        mode: str,
        *,
        encoding: str | None,
        errors: str | None,
        newline: str | None,
    ):
        await self._begin_operation()
        handle_id: int | None = None
        registered = False
        try:
            mode_spec = parse_mode(mode)
            validate_open_kwargs(
                mode_spec, encoding=encoding, errors=errors, newline=newline
            )

            handle_id = self._allocate_handle_id()
            initial_position = self._compute_initial_position(path, mode_spec)

            await self._register_handle(
                handle_id=handle_id,
                path=path,
                mode_spec=mode_spec,
                encoding=encoding,
                errors=errors,
                newline=newline,
                initial_position=initial_position,
            )
            registered = True
            self._ensure_open()

            from ._handle import FileHandle  # local import to avoid circular dependency

            handle = FileHandle(
                pool=self,
                handle_id=handle_id,
                path=path,
                mode_spec=mode_spec,
                encoding=encoding,
                errors=errors,
                newline=newline,
                initial_position=initial_position,
            )
            self._track_handle(handle)
            return handle
        except BaseException:  # noqa: BLE001
            if registered and handle_id is not None:
                await self._unregister_handle(handle_id)
            raise
        finally:
            await self._end_operation()

    async def _begin_operation(self) -> None:
        self._bind_loop()
        async with self._ops_condition:
            if self._closed:
                raise PoolClosedError("file pool is closed")
            self._active_operations += 1

    async def _end_operation(self) -> None:
        async with self._ops_condition:
            self._active_operations -= 1
            if self._active_operations < 0:
                raise PoolStateError("active operation counter became negative")
            if self._active_operations == 0:
                self._ops_condition.notify_all()

    async def _run_blocking(self, func: Callable[..., T], *args: Any) -> T:
        """
        Execute callable in thread executor when configured.

        In no-thread mode, this runs synchronously in the event-loop thread.
        """
        if self._executor is None:
            return func(*args)
        loop = asyncio.get_running_loop()
        bound = functools.partial(func, *args)
        worker = loop.run_in_executor(self._executor, bound)
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Keep descriptor safety: do not unwind while worker may still be
            # using the file object in another thread.
            await worker
            raise

    async def stats(self) -> dict[str, int]:
        """Return lightweight pool/descriptor stats."""
        self._bind_loop()
        descriptor_stats = await self._manager.stats()
        return {
            **descriptor_stats,
            "thread_pool_size": self._thread_pool_size,
            "chunk_size": self._chunk_size,
            "descriptor_acquire_timeout_ms": (
                -1
                if self._descriptor_acquire_timeout is None
                else int(self._descriptor_acquire_timeout * 1000)
            ),
        }

    def _schedule_best_effort_cleanup(self, handle_id: int) -> None:
        """Schedule best-effort async handle cleanup from finalizers."""
        loop = self._loop
        if loop is None or loop.is_closed() or self._closed:
            return

        def _create_task() -> None:
            if self._closed:
                return
            task = loop.create_task(self._manager.unregister(handle_id))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

        loop.call_soon_threadsafe(_create_task)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def fsync_on_write(self) -> bool:
        return self._fsync_on_write

    @property
    def uses_threads(self) -> bool:
        return self._executor is not None

    @property
    def manager(self) -> DescriptorManager:
        """Internal descriptor manager. Prefer FilePool wrapper methods."""
        return self._manager

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
            return
        if self._loop is not loop:
            raise PoolStateError("FilePool can only be used from one event loop")

    def _ensure_open(self) -> None:
        if self._closed:
            raise PoolClosedError("file pool is closed")

    def _allocate_handle_id(self) -> int:
        handle_id = self._next_handle_id
        self._next_handle_id += 1
        return handle_id

    async def _register_handle(
        self,
        *,
        handle_id: int,
        path: str,
        mode_spec: ModeSpec,
        encoding: str | None,
        errors: str | None,
        newline: str | None,
        initial_position: int,
    ) -> None:
        await self._manager.register(
            handle_id=handle_id,
            path=path,
            mode=mode_spec.normalized,
            binary=mode_spec.binary,
            encoding=encoding,
            errors=errors,
            newline=newline,
            initial_position=initial_position,
        )

    async def _acquire_descriptor(self, handle_id: int) -> IOBase:
        return await self._manager.acquire(handle_id)

    async def _release_descriptor(
        self,
        handle_id: int,
        *,
        dirty: bool | None = None,
        position: int | None = None,
    ) -> None:
        await self._manager.release(handle_id, dirty=dirty, position=position)

    async def _unregister_handle(self, handle_id: int) -> None:
        await self._manager.unregister(handle_id)

    def _track_handle(self, handle: FileHandle) -> None:
        self._live_handles.add(handle)

    def _forget_handle(self, handle: FileHandle) -> None:
        self._live_handles.discard(handle)

    @staticmethod
    def _compute_initial_position(path: str, mode_spec: ModeSpec) -> int:
        if not mode_spec.append:
            return 0
        try:
            return os.path.getsize(path)
        except FileNotFoundError:
            return 0

    async def _close_internal(self) -> None:
        async with self._ops_condition:
            if self._closed:
                return
            self._closed = True
            while self._active_operations > 0:
                await self._ops_condition.wait()

        manager_error: BaseException | None = None
        try:
            await self._manager.close_all()
        except BaseException as exc:  # noqa: BLE001
            manager_error = exc

        for handle in list(self._live_handles):
            handle._mark_closed_by_pool()  # noqa: SLF001
        self._live_handles.clear()

        executor_error: BaseException | None = None
        if self._executor is not None:
            executor = self._executor
            self._executor = None
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    functools.partial(
                        executor.shutdown,
                        wait=True,
                        cancel_futures=False,
                    ),
                )
            except BaseException as exc:  # noqa: BLE001
                executor_error = exc

        pending_tasks = [task for task in self._bg_tasks if not task.done()]
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        if manager_error is not None:
            if executor_error is not None and hasattr(manager_error, "add_note"):
                manager_error.add_note(f"executor shutdown error: {executor_error!r}")  # type: ignore[attr-defined]
            raise manager_error
        if executor_error is not None:
            raise executor_error

    def __repr__(self) -> str:
        return (
            f"FilePool(descriptor_pool_size={self._descriptor_pool_size}, "
            f"thread_pool_size={self._thread_pool_size}, "
            f"chunk_size={self._chunk_size}, "
            f"fsync_on_write={self._fsync_on_write}, "
            f"closed={self._closed})"
        )
