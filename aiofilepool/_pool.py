import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
import os
from typing import Any, Callable, Self

from aiofilepool._chunking import balanced_chunks
from aiofilepool._fd_manager import FileDescriptorManager
from aiofilepool._handle import FileHandle
from aiofilepool.errors import FilePoolClosedError

from ._modes import ModeSpec


class FilePool:
    def __init__(
        self,
        descriptor_pool_size: int = 128,
        thread_pool_size: int = 4,
        max_chunk_size: int = 128 * 1024 * 1024,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self._thread_pool_size = thread_pool_size
        self._chunk_size = max_chunk_size
        self._loop = loop or asyncio.get_running_loop()
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=thread_pool_size)
            if thread_pool_size > 0
            else None
        )
        self._fd_manager = FileDescriptorManager(max_descriptors=descriptor_pool_size)
        self._closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        await self._fd_manager.close_all()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        self._closed = True

    async def _close_handle(self, handle: FileHandle) -> None:
        await self._fd_manager.close(handle)

    async def _get_stats(self, path: str) -> os.stat_result:
        stat = os.stat(path)
        return stat

    async def _run_blocking[T](self, func: Callable[..., T], *args: Any) -> T:
        if self._executor is None:
            result = func(*args)
            await asyncio.sleep(0)
            return result

        result = self._loop.run_in_executor(self._executor, func, *args)
        try:
            return await asyncio.shield(result)
        except asyncio.CancelledError:
            await result
            raise

    async def _read(self, handle: FileHandle, size: int, offset: int) -> bytes:
        fd = await self._fd_manager.acquire(handle)
        try:
            fd.seek(offset)
            if size < self._chunk_size:
                data = await self._run_blocking(fd.read, size)
            else:
                data = bytearray()
                for chunk_size in balanced_chunks(size, self._chunk_size):
                    data.extend(await self._run_blocking(fd.read, chunk_size))
        finally:
            await self._fd_manager.release(handle)

        return bytes(data)

    async def _write(
        self, handle: FileHandle, data: bytes, offset: int
    ) -> tuple[int, Exception | None]:
        written = 0
        error = None
        fd = await self._fd_manager.acquire(handle)
        try:
            fd.seek(offset)
            if len(data) < self._chunk_size:
                written = await self._run_blocking(fd.write, data)
            else:
                for chunk_size in balanced_chunks(len(data), self._chunk_size):
                    written += await self._run_blocking(fd.write, data[:chunk_size])
                    data = data[chunk_size:]
        except BaseException as e:
            error = e
        finally:
            await self._fd_manager.release(handle)

        return written, error

    async def _truncate(self, handle: FileHandle, size: int) -> None:
        fd = await self._fd_manager.acquire(handle)
        try:
            await self._run_blocking(fd.truncate, size)
        finally:
            await self._fd_manager.release(handle)

    def open(
        self,
        path: str | os.PathLike[str],
        mode: str,
    ) -> FileHandle:
        if self._closed:
            raise FilePoolClosedError()
        mode_spec = ModeSpec.from_str(mode)
        return FileHandle(
            pool=self,
            path=os.fspath(path),
            mode=mode_spec,
        )
