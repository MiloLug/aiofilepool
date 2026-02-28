"""Standalone async file pool with bounded real descriptors.

This package exposes:
- `FilePool`: top-level async pool API
- `FileHandle`: virtual async file handle
- mode and lifecycle exceptions from `_errors`
"""

from ._errors import (
    AppendModePositionalError,
    DataTypeMismatchError,
    DescriptorAcquireTimeoutError,
    HandleClosedError,
    InvalidModeError,
    PoolClosedError,
    PoolError,
    PoolStateError,
    TextModePositionalError,
)
from ._handle import FileHandle
from ._pool import FilePool

__all__ = [
    "AppendModePositionalError",
    "DataTypeMismatchError",
    "DescriptorAcquireTimeoutError",
    "FileHandle",
    "FilePool",
    "HandleClosedError",
    "InvalidModeError",
    "PoolClosedError",
    "PoolError",
    "PoolStateError",
    "TextModePositionalError",
]
