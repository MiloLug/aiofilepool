import pytest

from aiofilepool.errors import (
    FileHandleInitializedError,
    FileHandleNotOpenError,
    FilePoolNotOpenError,
    InvalidFileModeError,
    InvalidFilePositionError,
)


@pytest.mark.parametrize(
    ("error_type", "default_message"),
    [
        (InvalidFileModeError, "invalid file mode"),
        (FileHandleNotOpenError, "file handle is not open"),
        (FileHandleInitializedError, "file handle is already initialized"),
        (InvalidFilePositionError, "invalid file position"),
        (FilePoolNotOpenError, "file pool is not open"),
    ],
)
def test_error_classes_expose_default_messages(
    error_type, default_message: str
) -> None:
    assert str(error_type()) == default_message


def test_error_message_can_be_overridden() -> None:
    assert str(FilePoolNotOpenError("custom message")) == "custom message"
