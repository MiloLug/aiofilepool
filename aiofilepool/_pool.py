import asyncio
from enum import IntEnum
import functools
from concurrent.futures.thread import ThreadPoolExecutor
from collections.abc import Callable
from typing import Any, BinaryIO, Self

from aiofilepool._binary_io import BinaryIOAdapter
from aiofilepool._chunking import BalancedChunker, Chunker
from aiofilepool._fd_manager import FileDescriptorManager
from aiofilepool._fs import AsyncFileSystem
from aiofilepool._handle import FileHandle
from aiofilepool._types import StrOrBytesPath
from aiofilepool.errors import FilePoolNotOpenError

from ._modes import ModeSpec


class FilePoolState(IntEnum):
    OPEN = 0
    CLOSING = 1
    CLOSED = 2


class FilePool:
    def __init__(
        self,
        descriptor_pool_size: int = 128,
        thread_pool_size: int = 4,
        loop: asyncio.AbstractEventLoop | None = None,
        chunker: Chunker = BalancedChunker(),
        chunking_threshold: int = 128 * 1024 * 1024,
        fs: AsyncFileSystem | None = None,
    ):
        """
        Args:
            descriptor_pool_size: The maximum number of file descriptors to use.
            thread_pool_size: The maximum number of threads to use.
                If 0, no threads will be used.
            loop: The event loop to use.
            chunker: The chunker to use.
            chunking_threshold: The size threshold at which to use chunking.
            fs: The file system tools to use. If None, a default implementation will be used.
        """
        self._thread_pool_size = thread_pool_size
        self._loop = loop or asyncio.get_running_loop()
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=thread_pool_size)
            if thread_pool_size > 0
            else None
        )
        self._fd_manager = FileDescriptorManager(max_descriptors=descriptor_pool_size)
        self._state = FilePoolState.OPEN
        self._close_task: asyncio.Task[None] | None = None
        self._chunker = chunker
        self._chunking_threshold = chunking_threshold
        self.fs = fs or AsyncFileSystem(self)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self, timeout: float | None = None) -> None:
        if self._close_task is None:
            self._state = FilePoolState.CLOSING
            self._close_task = asyncio.create_task(self._close_internal())
        try:
            await asyncio.wait_for(asyncio.shield(self._close_task), timeout=timeout)
        except asyncio.CancelledError:
            await self._close_task
            raise
        else:
            self._state = FilePoolState.CLOSED

    async def _run_blocking[T](self, func: Callable[..., T], *args: Any) -> T:
        if self._executor is None:
            result = func(*args)
            await asyncio.sleep(0)
            return result

        coro_result = self._loop.run_in_executor(self._executor, func, *args)
        try:
            return await asyncio.shield(coro_result)
        except asyncio.CancelledError:
            await coro_result
            raise

    def _open_guard(self) -> None:
        if self._state != FilePoolState.OPEN:
            raise FilePoolNotOpenError()

    def open(
        self,
        path: StrOrBytesPath,
        mode: str = "r",
    ) -> FileHandle:
        self._open_guard()
        mode_spec = ModeSpec.from_str(mode)
        return FileHandle(
            pool=self,
            path=path,
            mode=mode_spec,
        )

    def manage(self, io: BinaryIO) -> BinaryIOAdapter:
        return BinaryIOAdapter(self, io)

    async def _close_internal(self) -> None:
        close_error: BaseException | None = None
        try:
            await self._fd_manager.close()
        except BaseException as e:
            if close_error is None:
                close_error = e

        if self._executor is not None:
            executor = self._executor
            self._executor = None
            await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(
                    executor.shutdown,
                    wait=True,
                    cancel_futures=False,
                ),
            )

        if close_error is not None:
            raise close_error
