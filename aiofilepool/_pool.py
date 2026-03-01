import os

from ._modes import ModeSpec
from ._open_request import OpenRequest


class FilePool:
    def __init__(
        self,
        descriptor_pool_size: int = 256,
        thread_pool_size: int = 4,
        chunk_size: int = 64 * 1024 * 1024,
        fsync_on_write: bool = False,
    ):
        self._descriptor_pool_size = descriptor_pool_size
        self._thread_pool_size = thread_pool_size
        self._chunk_size = chunk_size
        self._fsync_on_write = fsync_on_write

        self._executor = ThreadPoolExecutor(
            max_workers=thread_pool_size,
            thread_name_prefix="aiofilepool-io",
        )
        self._manager = DescriptorManager(
            max_descriptors=descriptor_pool_size,
        )

    def open(
        self,
        path: str | os.PathLike[str],
        mode: str,
    ) -> OpenRequest:
        mode_spec = ModeSpec.from_str(mode)
        return OpenRequest(
            self,
            path=os.fspath(path),
            mode_spec=mode_spec,
        )
