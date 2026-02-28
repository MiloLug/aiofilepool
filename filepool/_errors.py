"""Custom exception hierarchy for the async file pool module."""


class PoolError(Exception):
    """Base exception for all file pool errors."""


class PoolClosedError(PoolError):
    """Raised when an operation is attempted on a closed pool."""


class HandleClosedError(PoolError):
    """Raised when an operation is attempted on a closed handle."""


class InvalidModeError(PoolError, ValueError):
    """Raised when an invalid file mode or option combination is used."""


class TextModePositionalError(PoolError, ValueError):
    """Raised when positional I/O is requested in text mode."""


class AppendModePositionalError(PoolError, ValueError):
    """Raised when positional write is requested in append mode."""


class DataTypeMismatchError(PoolError, TypeError):
    """Raised when written data type does not match file mode."""


class DescriptorAcquireTimeoutError(PoolError, TimeoutError):
    """Raised when descriptor acquisition times out."""


class PoolStateError(PoolError, RuntimeError):
    """Raised when internal state invariants are violated."""
