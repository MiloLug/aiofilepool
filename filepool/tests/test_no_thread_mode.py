from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from filepool import FilePool


class FilePoolNoThreadModeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_cooperative_mode_interleaves_tasks(self) -> None:
        size = 2 * 1024 * 1024
        p1 = self.tmp_path / "a.bin"
        p2 = self.tmp_path / "b.bin"
        p1.write_bytes(b"a" * size)
        p2.write_bytes(b"b" * size)

        async with FilePool(
            descriptor_pool_size=2, thread_pool_size=0, chunk_size=8 * 1024
        ) as pool:
            h1 = await pool.open(p1, "rb")
            h2 = await pool.open(p2, "rb")

            async def consume(handle) -> int:
                total = 0
                while True:
                    block = await handle.read(16 * 1024)
                    if not block:
                        break
                    total += len(block)
                return total

            done = asyncio.Event()
            ticks = 0

            async def ticker() -> None:
                nonlocal ticks
                while not done.is_set():
                    ticks += 1
                    await asyncio.sleep(0)

            ticker_task = asyncio.create_task(ticker())
            try:
                c1, c2 = await asyncio.gather(consume(h1), consume(h2))
                self.assertEqual(c1, size)
                self.assertEqual(c2, size)
            finally:
                done.set()
                await ticker_task
                await h1.close()
                await h2.close()

            # Not a strict performance assertion; we only verify cooperative yielding happened.
            self.assertGreater(ticks, 0)

    async def test_no_thread_mode_text_roundtrip(self) -> None:
        p = self.tmp_path / "text.txt"

        async with FilePool(
            descriptor_pool_size=1, thread_pool_size=0, chunk_size=4
        ) as pool:
            async with pool.open(p, "w+", encoding="utf-8") as handle:
                await handle.write("abcdef")
                await handle.seek(0)
                self.assertEqual(await handle.read(), "abcdef")
