from aiofilepool import AsyncBinaryIO, BinaryIOAdapter, FileHandle, FilePool, ModeSpec


def test_public_api_exports_are_importable() -> None:
    assert FilePool is not None
    assert FileHandle is not None
    assert ModeSpec is not None
    assert BinaryIOAdapter is not None
    assert AsyncBinaryIO is not None


def test_io_implementations_subclass_async_binary_io() -> None:
    assert issubclass(FileHandle, AsyncBinaryIO)
    assert issubclass(BinaryIOAdapter, AsyncBinaryIO)
