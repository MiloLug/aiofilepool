# filepool

`filepool` is a standalone stdlib-only async module for virtual file handles with a bounded set of real OS descriptors.

## What it guarantees

- Supports many logical handles while capping real open descriptors via `descriptor_pool_size`.
- `pool.open(...)` supports both:
  - `handle = await pool.open(...)`
  - `async with pool.open(...) as handle: ...`
- Sequential I/O for both binary and text modes using a virtual handle position.
- Binary mode supports positional I/O (`offset=`) with pread/pwrite semantics (virtual position is unchanged).
- Text mode is sequential-only (no positional `offset=`).
- `await write(...)` always flushes before returning.

## What it does not guarantee

- It does not bypass OS hard limits at one instant. Instead, it stays robust by bounding real open fds and reopening on demand.
- In `thread_pool_size=0` mode, syscalls are still synchronous. The module provides cooperative interleaving by chunking and yielding between chunks.
- It does not deduplicate handles: each `open()` call creates a distinct logical handle.

## Core API

```python
from filepool import FilePool

async def demo() -> None:
    async with FilePool(descriptor_pool_size=64, thread_pool_size=4) as pool:
        async with pool.open("data.bin", "rb") as f:
            blob = await f.read(4096)

        handle = await pool.open("events.log", "a", encoding="utf-8")
        await handle.write("event\n")
        await handle.close()
```

## Mode rules

- Binary mode (`b` in mode):
  - `read(..., offset=...)` and `write(..., offset=...)` are allowed.
  - Data type must be bytes-like for writes.
- Text mode (no `b`):
  - `offset=` in read/write raises `TextModePositionalError`.
  - Data type must be `str` for writes.
- Append modes (`a`, `ab`, `a+`, `ab+`):
  - `write(..., offset=...)` raises `AppendModePositionalError`.

## Durability tuning

- `fsync_on_write=False` (default): flush only, faster.
- `fsync_on_write=True`: flush + fsync on each write/truncate, safer but slower.

## Performance tuning

- `descriptor_pool_size`:
  - Increase if many handles are active simultaneously and churn is high.
  - Decrease to tighten fd usage.
- `thread_pool_size`:
  - `> 0`: better event-loop responsiveness under heavy I/O.
  - `0`: no threads, cooperative only.
- `chunk_size` (used in no-thread mode):
  - Larger: better throughput, less interleaving.
  - Smaller: more interleaving, more overhead.

## Lifecycle recommendations

- Prefer `async with` for both pool and handles.
- Explicitly close handles when not using context managers.
- `__del__` emits a `ResourceWarning` and attempts best-effort cleanup, but this is not a correctness mechanism.

