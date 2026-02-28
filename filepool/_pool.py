"""High-level FilePool API and lifecycle management."""

from __future__ import annotations

import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from ._descriptors import DescriptorManager
from ._errors import PoolClosedError, PoolStateError
from ._modes import ModeSpec, parse_mode, validate_open_kwargs
from ._open_request import OpenRequest

T = TypeVar("T")


class FilePool:
    """Async file pool with bounded real descriptors and optional thread workers."""

    def __init__(
        self,
        descriptor_pool_size: int = 256,
        thread_pool_size: int = 4,
        chunk_size: int = 64 * 1024,
        fsync_on_write: bool = False,
    ) -> None:
        if descriptor_pool_size < 1:
            raise ValueError("descriptor_pool_size must be >= 1")
        if thread_pool_size < 0:
            raise ValueError("thread_pool_size must be >= 0")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        self._descriptor_pool_size = descriptor_pool_size
        self._thread_pool_size = thread_pool_size
        self._chunk_size = chunk_size
        self._fsync_on_write = fsync_on_write

        self._manager = DescriptorManager(max_descriptors=descriptor_pool_size)
        self._executor = (
            ThreadPoolExecutor(
                max_workers=thread_pool_size, thread_name_prefix="filepool-io"
            )
            if thread_pool_size > 0
            else None
        )

        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._next_handle_id = 1
        self._id_lock = asyncio.Lock()

        self._ops_condition = asyncio.Condition()
        self._active_operations = 0

        self._bg_tasks: set[asyncio.Task[None]] = set()

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
        """Close pool, wait for in-flight operations, then close descriptors."""
        async with self._ops_condition:
            if self._closed:
                return
            self._closed = True
            while self._active_operations > 0:
                await self._ops_condition.wait()

        await self._manager.close_all()

        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

        pending_tasks = [task for task in self._bg_tasks if not task.done()]
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

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
        self._bind_loop()
        self._ensure_open()

        mode_spec = parse_mode(mode)
        validate_open_kwargs(
            mode_spec, encoding=encoding, errors=errors, newline=newline
        )

        handle_id = await self._allocate_handle_id()
        initial_position = self._compute_initial_position(path, mode_spec)

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
        return handle

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
        descriptor_stats = await self._manager.stats()
        return {
            **descriptor_stats,
            "thread_pool_size": self._thread_pool_size,
            "chunk_size": self._chunk_size,
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

    async def _allocate_handle_id(self) -> int:
        async with self._id_lock:
            handle_id = self._next_handle_id
            self._next_handle_id += 1
            return handle_id

    @staticmethod
    def _compute_initial_position(path: str, mode_spec: ModeSpec) -> int:
        if not mode_spec.append:
            return 0
        try:
            return os.path.getsize(path)
        except FileNotFoundError:
            return 0
