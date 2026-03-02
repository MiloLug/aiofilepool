from typing import Self

from aiofilepool.errors import InvalidFileModeError


class ModeSpec:
    __slots__ = ("read", "write", "truncate", "mode", "renewal_mode")

    def __init__(self, read: bool, write: bool, truncate: bool):
        if not read and not write:
            raise InvalidFileModeError("mode must be at least read or write")

        self.read = read
        self.write = write
        self.truncate = truncate
        self.mode = f"{'r' if read and not truncate else ''}{'w' if write and truncate else ''}{'+' if write and read else ''}b"
        self.renewal_mode = f"r{'+' if write else ''}b"

    @classmethod
    def from_str(cls, mode: str) -> Self:
        has_r = "r" in mode
        has_w = "w" in mode
        has_plus = "+" in mode

        return cls(
            read=has_r or has_plus,
            write=has_w or has_plus,
            truncate=has_w,
        )
