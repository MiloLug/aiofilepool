from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from filepool import FilePool


class FilePoolEvictionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_descriptor_count_is_bounded_and_positions_survive_eviction(
        self,
    ) -> None:
        file_count = 24
        paths: list[Path] = []
        for i in range(file_count):
            path = self.tmp_path / f"f{i:03d}.bin"
            path.write_bytes(bytes([i]) * 8)
            paths.append(path)

        async with FilePool(descriptor_pool_size=3, thread_pool_size=2) as pool:
            handles = [await pool.open(path, "rb") for path in paths]

            try:
                first_round = []
                for handle in handles:
                    first_round.append(await handle.read(1))
                    stats = await pool.stats()
                    self.assertLessEqual(stats["open_descriptors"], 3)

                second_round = []
                for handle in handles:
                    second_round.append(await handle.read(1))
                    stats = await pool.stats()
                    self.assertLessEqual(stats["open_descriptors"], 3)

                self.assertEqual(first_round, [bytes([i]) for i in range(file_count)])
                self.assertEqual(second_round, [bytes([i]) for i in range(file_count)])
            finally:
                for handle in handles:
                    await handle.close()

    async def test_reopen_preserves_virtual_cursor(self) -> None:
        p1 = self.tmp_path / "a.bin"
        p2 = self.tmp_path / "b.bin"
        p1.write_bytes(b"abcdef")
        p2.write_bytes(b"123456")

        async with FilePool(descriptor_pool_size=1, thread_pool_size=1) as pool:
            h1 = await pool.open(p1, "rb")
            h2 = await pool.open(p2, "rb")
            try:
                self.assertEqual(await h1.read(2), b"ab")
                self.assertEqual(await h2.read(1), b"1")  # evicts h1 descriptor
                self.assertEqual(
                    await h1.read(2), b"cd"
                )  # must continue from previous cursor
            finally:
                await h1.close()
                await h2.close()
