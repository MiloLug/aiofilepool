from typing import Self

from aiofilepool.errors import InvalidFileModeError


class ModeSpec:
    __slots__ = ("read", "write", "truncate", "mode", "renewal_mode")

    def __init__(
        self, read: bool, write: bool, truncate: bool, mode: str, renewal_mode: str
    ):
        self.read = read
        self.write = write
        self.truncate = truncate
        self.mode = mode
        self.renewal_mode = renewal_mode

    @classmethod
    def from_str(cls, mode: str) -> Self:
        has_r = "r" in mode
        has_w = "w" in mode
        has_plus = "+" in mode
        writable = has_w or has_plus

        if has_r and has_w:
            raise InvalidFileModeError("mode cannot contain both 'r' and 'w'")
        if not (has_r or has_w):
            raise InvalidFileModeError("mode must be at least read or write")

        return cls(
            read=has_r or has_plus,
            write=writable,
            truncate=has_w,
            mode="".join(set(mode) & set("rwx+")) + "b",
            renewal_mode=f"r{'+' if writable else ''}b",
        )
