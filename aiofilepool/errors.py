class FilePoolError(Exception):
    message: str

    def __init__(self, message: str = ""):
        self.message = message or self.message
        super().__init__(self.message)


class InvalidFileModeError(FilePoolError):
    message = "invalid file mode"


class FileHandleNotOpenError(FilePoolError):
    message = "file handle is not open"


class FileHandleInitializedError(FilePoolError):
    message = "file handle is already initialized"


class InvalidFilePositionError(FilePoolError):
    message = "invalid file position"


class FilePoolClosedError(FilePoolError):
    message = "file pool is closed"
