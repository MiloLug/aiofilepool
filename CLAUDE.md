# CLAUDE.md

Guide Claude Code (claude.ai/code) in this repo.

## Commands

Python 3.12+ via `uv`. Pre-commit runs `uv-lock`, `ruff-check --fix`, `ruff-format`, `mypy .`.

```bash
uv sync                                   # install deps (incl. dev group)
uv run --group dev mypy .                 # type-check
uv run ruff check --fix && uv run ruff format
uv run pytest                             # full suite (pytest-asyncio auto mode)
uv run pytest -m "not slow"               # skip slow property/stress tests
uv run pytest tests/aiofilepool/test_fd_pool.py::test_name   # single test
uv run python main.py                     # micro-benchmark vs plain open()
uv build                                  # build sdist + wheel
```

## Architecture

`aiofilepool` — async file-descriptor pool. `FilePool` caps the number of
concurrently-open OS file descriptors while callers hold thousands of logical
`FileHandle` objects; a handle acquires a real fd only while doing I/O and
re-opens on demand (`_fd_manager.py`), so chunked reads/writes over many files
never exhaust the OS fd limit.

Single flat package, **stdlib-only** (no third-party runtime deps):

- `_pool.py` — `FilePool`: the pool + `open()` / `manage()` / `close()`; owns the fd manager, the thread-pool executor, and the default chunker/threshold.
- `_handle.py` — `FileHandle` (implements `AsyncBinaryIO`): `read` / `write` / `seek` / `tell` / `size` / `truncate` / `allocate` / `chunks` / `fsync` / `close`.
- `_fd_manager.py` — `FileDescriptorManager`: caps live fds, re-opens on demand, evicts.
- `_base_io.py` — `AsyncBinaryIO` ABC + IO-state enum. `_binary_io.py` — `BinaryIOAdapter` wraps a sync `BinaryIO`.
- `_chunking.py` — `Chunker` / `FixedChunker` / `BalancedChunker` split large I/O. `_modes.py` — `ModeSpec` parses open-mode strings. `_fs.py` — `AsyncFileSystem` fs helpers. `_types.py` — path/type aliases. `errors.py` — exception hierarchy.

Public API is re-exported from `aiofilepool/__init__.py`.

## Conventions

- `pytest-asyncio` auto mode (no `@pytest.mark.asyncio` needed). The `slow` marker tags `hypothesis` property/stress tests — deselect with `-m "not slow"`.
- Fully type-annotated; ships `py.typed` (PEP 561). Run `uv run --group dev mypy .` before committing (pre-commit gate).
- Both absolute (`from aiofilepool._x import ...`) and relative (`from ._x import ...`) intra-package imports appear; keep the import package name `aiofilepool`.
