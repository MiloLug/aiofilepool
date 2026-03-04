import pytest

from aiofilepool.errors import (
    IOInitializedError,
    IONotOpenError,
    FilePoolNotOpenError,
    InvalidFileModeError,
    InvalidPositionError,
)


@pytest.mark.parametrize(
    ("error_type", "default_message"),
    [
        (InvalidFileModeError, "invalid file mode"),
        (IONotOpenError, "file handle is not open"),
        (IOInitializedError, "file handle is already initialized"),
        (InvalidPositionError, "invalid file position"),
        (FilePoolNotOpenError, "file pool is not open"),
    ],
)
def test_error_classes_expose_default_messages(
    error_type, default_message: str
) -> None:
    assert str(error_type()) == default_message


def test_error_message_can_be_overridden() -> None:
    assert str(FilePoolNotOpenError("custom message")) == "custom message"
