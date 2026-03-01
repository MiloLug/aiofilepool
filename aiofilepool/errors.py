class FilePoolError(Exception):
    pass


class InvalidFileModeError(FilePoolError):
    pass


class HandleClosedError(FilePoolError):
    pass


class InvalidFilePositionError(FilePoolError):
    pass


class InvalidFileSizeError(FilePoolError):
    pass
