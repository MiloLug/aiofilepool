from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from filepool import FilePool, InvalidModeError, TextModePositionalError


class FilePoolTextModeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_text_mode_sequential_seek_tell(self) -> None:
        path = self.tmp_path / "note.txt"
        async with FilePool(descriptor_pool_size=2, thread_pool_size=2) as pool:
            async with pool.open(path, "w+", encoding="utf-8") as handle:
                self.assertEqual(await handle.tell(), 0)
                await handle.write("hello world")
                self.assertEqual(await handle.tell(), 11)
                await handle.seek(6)
                self.assertEqual(await handle.tell(), 6)
                self.assertEqual(await handle.read(), "world")

    async def test_text_mode_rejects_offset_for_read_and_write(self) -> None:
        path = self.tmp_path / "data.txt"
        path.write_text("abc", encoding="utf-8")

        async with FilePool(descriptor_pool_size=2, thread_pool_size=1) as pool:
            async with pool.open(path, "r+", encoding="utf-8") as handle:
                with self.assertRaises(TextModePositionalError):
                    await handle.read(1, offset=0)
                with self.assertRaises(TextModePositionalError):
                    await handle.write("x", offset=0)

    async def test_binary_mode_rejects_text_kwargs(self) -> None:
        path = self.tmp_path / "blob.bin"
        path.write_bytes(b"abc")

        async with FilePool(descriptor_pool_size=2, thread_pool_size=1) as pool:
            with self.assertRaises(InvalidModeError):
                await pool.open(path, "rb", encoding="utf-8")
