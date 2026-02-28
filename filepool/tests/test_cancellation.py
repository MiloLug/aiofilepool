from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from filepool import FilePool


class FilePoolCancellationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_cancelled_read_does_not_leak_active_descriptor(self) -> None:
        path = self.tmp_path / "huge.bin"
        path.write_bytes(b"x" * (4 * 1024 * 1024))

        async with FilePool(
            descriptor_pool_size=1, thread_pool_size=0, chunk_size=4096
        ) as pool:
            handle = await pool.open(path, "rb")
            try:
                task = asyncio.create_task(handle.read())
                await asyncio.sleep(0)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                stats = await pool.stats()
                self.assertEqual(stats["active_descriptors"], 0)
                self.assertLessEqual(stats["open_descriptors"], 1)

                await handle.seek(0)
                self.assertEqual(await handle.read(8), b"xxxxxxxx")
            finally:
                await handle.close()

    async def test_cancelled_threaded_operation_keeps_pool_usable(self) -> None:
        path = self.tmp_path / "threaded.bin"
        path.write_bytes(b"y" * (2 * 1024 * 1024))

        async with FilePool(
            descriptor_pool_size=1, thread_pool_size=1, chunk_size=4096
        ) as pool:
            handle = await pool.open(path, "rb")
            try:
                task = asyncio.create_task(handle.read())
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

                stats = await pool.stats()
                self.assertEqual(stats["active_descriptors"], 0)

                await handle.seek(0)
                self.assertEqual(await handle.read(4), b"yyyy")
            finally:
                await handle.close()
