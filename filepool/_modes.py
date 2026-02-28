"""Mode parsing and validation helpers for the async file pool."""

from __future__ import annotations

from dataclasses import dataclass
from io import UnsupportedOperation
from typing import Any

from ._errors import (
    AppendModePositionalError,
    DataTypeMismatchError,
    InvalidModeError,
    TextModePositionalError,
)


@dataclass(frozen=True, slots=True)
class ModeSpec:
    """Normalized mode information used by the pool."""

    raw: str
    normalized: str
    binary: bool
    readable: bool
    writable: bool
    append: bool
    updating: bool
    creating: bool
    truncating: bool

    @property
    def text(self) -> bool:
        return not self.binary


def parse_mode(mode: str) -> ModeSpec:
    """Parse and normalize a Python file mode string."""
    if not isinstance(mode, str) or not mode:
        raise InvalidModeError("mode must be a non-empty string")

    valid = set("rwaxtb+")
    invalid = sorted({char for char in mode if char not in valid})
    if invalid:
        raise InvalidModeError(f"mode contains invalid characters: {invalid}")

    bases = [char for char in mode if char in "rwax"]
    if len(bases) != 1:
        raise InvalidModeError("mode must contain exactly one of 'r', 'w', 'a', or 'x'")
    base = bases[0]

    if mode.count("+") > 1:
        raise InvalidModeError("mode can contain '+' at most once")
    updating = "+" in mode

    if "b" in mode and "t" in mode:
        raise InvalidModeError("mode cannot contain both 'b' and 't'")
    if mode.count("b") > 1 or mode.count("t") > 1:
        raise InvalidModeError("mode can contain 'b' or 't' at most once")
    binary = "b" in mode

    # Disallow repeated base markers and accidental duplicated flags.
    for flag in "rwaxt":
        if mode.count(flag) > 1:
            raise InvalidModeError(f"mode flag '{flag}' is repeated")

    readable = base == "r" or updating
    writable = base in {"w", "a", "x"} or updating

    normalized = base
    if binary:
        normalized += "b"
    if updating:
        normalized += "+"

    return ModeSpec(
        raw=mode,
        normalized=normalized,
        binary=binary,
        readable=readable,
        writable=writable,
        append=(base == "a"),
        updating=updating,
        creating=(base == "x"),
        truncating=(base == "w"),
    )


def validate_open_kwargs(
    spec: ModeSpec,
    *,
    encoding: str | None,
    errors: str | None,
    newline: str | None,
) -> None:
    """Validate text/binary kwargs against the parsed mode."""
    if spec.binary and (
        encoding is not None or errors is not None or newline is not None
    ):
        raise InvalidModeError(
            "encoding/errors/newline are only valid for text modes; "
            f"received them for mode '{spec.raw}'"
        )


def validate_read_args(spec: ModeSpec, *, offset: int | None) -> None:
    """Validate read arguments for mode rules."""
    if not spec.readable:
        raise UnsupportedOperation(f"file mode '{spec.raw}' does not allow reads")
    if offset is not None:
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if spec.text:
            raise TextModePositionalError(
                "text mode does not support positional read offset"
            )


def validate_write_args(spec: ModeSpec, *, offset: int | None, data: Any) -> None:
    """Validate write arguments for mode and datatype rules."""
    if not spec.writable:
        raise UnsupportedOperation(f"file mode '{spec.raw}' does not allow writes")

    if spec.binary and not isinstance(data, (bytes, bytearray, memoryview)):
        raise DataTypeMismatchError(
            f"binary mode '{spec.raw}' expects bytes-like data, got {type(data).__name__}"
        )
    if spec.text and not isinstance(data, str):
        raise DataTypeMismatchError(
            f"text mode '{spec.raw}' expects str data, got {type(data).__name__}"
        )

    if offset is not None:
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if spec.text:
            raise TextModePositionalError(
                "text mode does not support positional write offset"
            )
        if spec.append:
            raise AppendModePositionalError(
                "append mode does not support positional write offset"
            )
