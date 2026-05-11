import os
from typing import TYPE_CHECKING

from aiofilepool._types import FileDescriptorOrPath, StrOrBytesPath

if TYPE_CHECKING:
    from aiofilepool._pool import FilePool


class AsyncFileSystem:
    def __init__(self, pool: "FilePool"):
        self._pool = pool

    async def stat(self, path: FileDescriptorOrPath) -> os.stat_result:
        return await self._pool._run_blocking(os.stat, path)

    async def rename(self, old: StrOrBytesPath, new: StrOrBytesPath) -> None:
        await self._pool._run_blocking(os.rename, old, new)

    async def exists(self, path: FileDescriptorOrPath) -> bool:
        return await self._pool._run_blocking(os.path.exists, path)

    async def is_file(self, path: FileDescriptorOrPath) -> bool:
        return await self._pool._run_blocking(os.path.isfile, path)

    async def is_dir(self, path: FileDescriptorOrPath) -> bool:
        return await self._pool._run_blocking(os.path.isdir, path)
