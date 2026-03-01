from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from filepool import FilePool, PoolClosedError


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

    async def test_cancelled_write_marks_release_as_dirty(self) -> None:
        path = self.tmp_path / "cancel-write.bin"
        path.write_bytes(b"x" * (2 * 1024 * 1024))

        async with FilePool(
            descriptor_pool_size=1,
            thread_pool_size=0,
            chunk_size=1024,
        ) as pool:
            handle = await pool.open(path, "rb+")
            release_dirty_values: list[bool | None] = []
            original_release = pool._release_descriptor

            async def tracking_release(handle_id, *, dirty=None, position=None):
                release_dirty_values.append(dirty)
                await original_release(handle_id, dirty=dirty, position=position)

            pool._release_descriptor = tracking_release  # type: ignore[method-assign]

            try:
                task = asyncio.create_task(handle.write(b"z" * (8 * 1024 * 1024)))
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                await handle.close()

            self.assertIn(True, release_dirty_values)

    async def test_cancelled_handle_close_still_unregisters(self) -> None:
        path = self.tmp_path / "close-handle.bin"
        path.write_bytes(b"abc")

        pool = FilePool(descriptor_pool_size=1, thread_pool_size=1)
        handle = await pool.open(path, "rb")
        started = asyncio.Event()
        proceed = asyncio.Event()
        original_unregister = pool._unregister_handle

        async def slow_unregister(handle_id: int) -> None:
            started.set()
            await proceed.wait()
            await original_unregister(handle_id)

        pool._unregister_handle = slow_unregister  # type: ignore[method-assign]

        try:
            close_task = asyncio.create_task(handle.close())
            await started.wait()
            close_task.cancel()
            proceed.set()

            with self.assertRaises(asyncio.CancelledError):
                await close_task

            self.assertTrue(handle.closed)
            stats = await pool.stats()
            self.assertEqual(stats["registered_handles"], 0)
        finally:
            await pool.close()

    async def test_cancelled_pool_close_completes_cleanup(self) -> None:
        path = self.tmp_path / "close-pool.bin"
        path.write_bytes(b"abc")

        pool = FilePool(descriptor_pool_size=1, thread_pool_size=1)
        handle = await pool.open(path, "rb")
        started = asyncio.Event()
        proceed = asyncio.Event()
        original_close_all = pool.manager.close_all

        async def slow_close_all() -> None:
            started.set()
            await proceed.wait()
            await original_close_all()

        pool.manager.close_all = slow_close_all  # type: ignore[method-assign]

        close_task = asyncio.create_task(pool.close())
        await started.wait()
        close_task.cancel()
        proceed.set()

        with self.assertRaises(asyncio.CancelledError):
            await close_task

        await pool.close()
        self.assertTrue(handle.closed)
        stats = await pool.stats()
        self.assertEqual(stats["registered_handles"], 0)

    async def test_open_in_progress_fails_when_pool_is_closing(self) -> None:
        path = self.tmp_path / "race.bin"
        path.write_bytes(b"abc")

        pool = FilePool(descriptor_pool_size=1, thread_pool_size=0)
        started = asyncio.Event()
        proceed = asyncio.Event()
        original_register = pool._register_handle

        async def slow_register(**kwargs) -> None:
            started.set()
            await proceed.wait()
            await original_register(**kwargs)

        pool._register_handle = slow_register  # type: ignore[method-assign]

        async def open_handle():
            return await pool.open(path, "rb")

        open_task = asyncio.create_task(open_handle())
        await started.wait()
        close_task = asyncio.create_task(pool.close())
        await asyncio.sleep(0)
        proceed.set()

        with self.assertRaises(PoolClosedError):
            await open_task
        await close_task
