from os import PathLike
from typing import Literal

type StrOrBytesPath = str | bytes | PathLike[str] | PathLike[bytes]
type FileDescriptorOrPath = int | StrOrBytesPath

# [os.SEEK_SET, os.SEEK_CUR, os.SEEK_END]
type Whence = Literal[0, 1, 2]
