import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

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


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(mode=st.text(alphabet="rwb+", min_size=0, max_size=6))
def test_mode_spec_fuzz_accepts_or_rejects_consistently(mode: str) -> None:
    """For every string of `r/w/b/+` flags, `from_str` either raises a clean
    `InvalidFileModeError` OR returns a spec whose `(read, write, truncate)`
    triple is consistent with the source flags and whose `mode`/`renewal_mode`
    are valid Python `open()` strings (no duplicate flags, all known)."""
    has_r = "r" in mode
    has_w = "w" in mode
    has_plus = "+" in mode
    has_b = "b" in mode
    has_dup = any(mode.count(c) > 1 for c in "rwb+")
    valid_chars_only = set(mode) <= {"r", "w", "+", "b"}
    has_r_xor_w = has_r ^ has_w

    expect_accept = (
        not has_dup and valid_chars_only and has_r_xor_w and (has_r or has_w)
    )

    if not expect_accept:
        with pytest.raises(InvalidFileModeError):
            ModeSpec.from_str(mode)
        return

    spec = ModeSpec.from_str(mode)

    # Truth table the parser MUST satisfy.
    writable = has_w or has_plus
    assert spec.read is (has_r or has_plus)
    assert spec.write is writable
    assert spec.truncate is has_w

    # Canonical mode is `r/w` + optional `+` + always `b`, no duplicates.
    assert spec.mode in {"rb", "r+b", "wb", "w+b"}
    assert all(spec.mode.count(c) <= 1 for c in "rw+b")

    # Renewal mode reuses the readable variant: `rb` if read-only, `r+b` if writable.
    assert spec.renewal_mode in {"rb", "r+b"}
    if writable:
        assert spec.renewal_mode == "r+b"

    # Static use: the canonical mode is a valid open() string. (No assertion call —
    # implicit: `open(path, spec.mode)` must work; covered by lifecycle/io tests.)
    _ = has_b  # `b` is implicit in the canonical mode; presence in input is fine.
