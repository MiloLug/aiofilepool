import pytest

from aiofilepool._modes import ModeSpec
from aiofilepool.errors import InvalidFileModeError


@pytest.mark.parametrize(
    ("mode", "read", "write", "truncate", "canonical_mode", "renewal_mode"),
    [
        ("r", True, False, False, "rb", "rb"),
        ("rb", True, False, False, "rb", "rb"),
        ("r+", True, True, False, "r+b", "r+b"),
        ("r+b", True, True, False, "r+b", "r+b"),
        ("rb+", True, True, False, "r+b", "r+b"),
        ("w", False, True, True, "wb", "r+b"),
        ("wb", False, True, True, "wb", "r+b"),
        ("w+", True, True, True, "w+b", "r+b"),
        ("w+b", True, True, True, "w+b", "r+b"),
        ("wb+", True, True, True, "w+b", "r+b"),
    ],
)
def test_mode_spec_accepts_valid_modes(
    mode: str,
    read: bool,
    write: bool,
    truncate: bool,
    canonical_mode: str,
    renewal_mode: str,
) -> None:
    spec = ModeSpec.from_str(mode)

    assert spec.read is read
    assert spec.write is write
    assert spec.truncate is truncate
    assert spec.mode == canonical_mode
    assert spec.renewal_mode == renewal_mode


@pytest.mark.parametrize("mode", ["rw", "wr", "r+w", "w+r", "rbw", "wr+b"])
def test_mode_spec_rejects_modes_containing_both_read_and_write(mode: str) -> None:
    with pytest.raises(InvalidFileModeError, match="both 'r' and 'w'"):
        ModeSpec.from_str(mode)


@pytest.mark.parametrize("mode", ["rr", "r++", "ww", "rbb", "wb++"])
def test_mode_spec_rejects_duplicate_flags(mode: str) -> None:
    with pytest.raises(InvalidFileModeError, match="duplicate"):
        ModeSpec.from_str(mode)


@pytest.mark.parametrize("mode", ["", "+", "b"])
def test_mode_spec_rejects_modes_without_read_or_write(mode: str) -> None:
    with pytest.raises(InvalidFileModeError, match="at least read or write"):
        ModeSpec.from_str(mode)


@pytest.mark.parametrize("mode", ["x", "ra", "rbt", "wx"])
def test_mode_spec_rejects_modes_with_unknown_flags(mode: str) -> None:
    with pytest.raises(InvalidFileModeError, match="unknown"):
        ModeSpec.from_str(mode)
