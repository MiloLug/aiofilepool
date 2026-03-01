from __future__ import annotations

from typing import TYPE_CHECKING

from ._errors import PoolStateError

if TYPE_CHECKING:
    from ._handle import FileHandle
    from ._pool import FilePool


class OpenRequest:
    """One-shot wrapper that can be awaited or used in `async with`."""

    def __init__(
        self,
        pool: "FilePool",
        path: str,
        read: bool = True,
        write: bool = False,
    ) -> None:
        self._pool = pool
        self._path = path
        self._used = False
        self._entered_handle: FileHandle | None = None

    def __await__(self):
        return self._consume().__await__()

    async def __aenter__(self) -> "FileHandle":
        handle = await self._consume()
        self._entered_handle = handle
        return await handle.__aenter__()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        handle = self._entered_handle
        self._entered_handle = None
        if handle is not None:
            await handle.__aexit__(exc_type, exc, tb)

    async def _consume(self) -> "FileHandle":
        if self._used:
            raise PoolStateError(
                "OpenRequest is one-shot; create a new pool.open(...) request"
            )
        self._used = True
        return await self._pool._open_handle(  # noqa: SLF001
            self._path,
            self._mode,
            encoding=self._encoding,
            errors=self._errors,
            newline=self._newline,
        )

    def __repr__(self) -> str:
        return (
            f"OpenRequest(path={self._path!r}, mode={self._mode!r}, used={self._used})"
        )
