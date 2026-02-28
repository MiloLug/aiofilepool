from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from filepool import (
    AppendModePositionalError,
    DataTypeMismatchError,
    FilePool,
    TextModePositionalError,
)


class FilePoolBasicTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_open_request_supports_await_and_async_with(self) -> None:
        path = self.tmp_path / "sample.bin"
        path.write_bytes(b"abcdef")

        async with FilePool(descriptor_pool_size=4, thread_pool_size=2) as pool:
            request = pool.open(path, "rb")
            handle = await request
            self.assertEqual(await handle.read(), b"abcdef")
            await handle.close()

            async with pool.open(path, "rb") as h2:
                self.assertEqual(await h2.read(), b"abcdef")

    async def test_binary_sequential_positional_seek_and_tell(self) -> None:
        path = self.tmp_path / "bytes.bin"
        path.write_bytes(b"0123456789")

        async with FilePool(descriptor_pool_size=2, thread_pool_size=2) as pool:
            async with pool.open(path, "rb+") as handle:
                self.assertEqual(await handle.read(3), b"012")
                self.assertEqual(await handle.tell(), 3)

                # Positional read must not move the virtual cursor.
                self.assertEqual(await handle.read(2, offset=6), b"67")
                self.assertEqual(await handle.tell(), 3)

                await handle.seek(3)
                self.assertEqual(await handle.write(b"XX"), 2)
                self.assertEqual(await handle.tell(), 5)

                # Positional write must not move the virtual cursor.
                self.assertEqual(await handle.write(b"Q", offset=0), 1)
                self.assertEqual(await handle.tell(), 5)

        self.assertEqual(path.read_bytes(), b"Q12XX56789")

    async def test_text_mode_is_sequential_and_validates_types(self) -> None:
        path = self.tmp_path / "text.txt"

        async with FilePool(descriptor_pool_size=3, thread_pool_size=1) as pool:
            async with pool.open(path, "w+", encoding="utf-8") as handle:
                self.assertEqual(await handle.write("hello"), 5)
                await handle.seek(0)
                self.assertEqual(await handle.read(), "hello")

                with self.assertRaises(TextModePositionalError):
                    await handle.read(1, offset=0)

                with self.assertRaises(TextModePositionalError):
                    await handle.write("x", offset=0)

                with self.assertRaises(DataTypeMismatchError):
                    await handle.write(b"x")  # type: ignore[arg-type]

    async def test_append_mode_rejects_positional_write(self) -> None:
        path = self.tmp_path / "append.bin"
        path.write_bytes(b"ab")

        async with FilePool(descriptor_pool_size=2, thread_pool_size=1) as pool:
            async with pool.open(path, "ab+") as handle:
                with self.assertRaises(AppendModePositionalError):
                    await handle.write(b"z", offset=0)
