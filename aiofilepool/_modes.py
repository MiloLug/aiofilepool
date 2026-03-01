from typing import Self


class ModeSpec:
    __slots__ = ("read", "write", "truncate")

    def __init__(self, read: bool, write: bool, truncate: bool):
        self.read = read
        self.write = write
        self.truncate = truncate

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
