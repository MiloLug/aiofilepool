from os import PathLike


type StrOrBytesPath = str | bytes | PathLike[str] | PathLike[bytes]
type FileDescriptorOrPath = int | StrOrBytesPath
