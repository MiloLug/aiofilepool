class FilePoolError(Exception):
    message: str

    def __init__(self, message: str = ""):
        self.message = message or self.message
        super().__init__(self.message)


class InvalidFileModeError(FilePoolError):
    message = "invalid file mode"


class IONotOpenError(FilePoolError):
    message = "file handle is not open"


class IOInitializedError(FilePoolError):
    message = "file handle is already initialized"


class InvalidPositionError(FilePoolError):
    message = "invalid file position"


class FilePoolNotOpenError(FilePoolError):
    message = "file pool is not open"
