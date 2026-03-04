import pytest

from aiofilepool._modes import ModeSpec
from aiofilepool.errors import InvalidFileModeError


@pytest.mark.parametrize(
    ("mode", "read", "write", "truncate", "flags", "renewal_mode"),
    [
        ("r", True, False, False, {"r"}, "rb"),
        ("r+", True, True, False, {"r", "+"}, "rb"),
        ("w", False, True, True, {"w"}, "r+b"),
        ("w+", True, True, True, {"w", "+"}, "r+b"),
    ],
)
def test_mode_spec_from_str_valid_modes(
    mode: str,
    read: bool,
    write: bool,
    truncate: bool,
    flags: set[str],
    renewal_mode: str,
) -> None:
    spec = ModeSpec.from_str(mode)

    assert spec.read is read
    assert spec.write is write
    assert spec.truncate is truncate
    assert spec.mode.endswith("b")
    assert set(spec.mode[:-1]) == flags
    assert spec.renewal_mode == renewal_mode


@pytest.mark.parametrize("mode", ["rw", "wr", "r+w", "w+r"])
def test_mode_spec_rejects_modes_containing_both_read_and_write(mode: str) -> None:
    with pytest.raises(InvalidFileModeError, match="both 'r' and 'w'"):
        ModeSpec.from_str(mode)


@pytest.mark.parametrize("mode", ["", "+", "x", "b"])
def test_mode_spec_rejects_modes_without_read_or_write(mode: str) -> None:
    with pytest.raises(InvalidFileModeError, match="at least read or write"):
        ModeSpec.from_str(mode)
