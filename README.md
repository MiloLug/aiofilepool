# aiofilepool

Async file-descriptor pool for Python.
`FilePool` caps how many OS file descriptors are open at once,
while your code can freely hold **thousands** of
logical async file handles.

- **Async-native** - built on `asyncio`; blocking I/O is offloaded to a thread pool.
- **Chunked I/O** - large reads/writes split via pluggable chunkers; you can open streams with `chunks()`.
- **Zero dependencies** - pure standard library.

## Install

```bash
pip install aiofilepool
# or
uv add aiofilepool
```

Requires Python 3.12+.

## Usage

```python
import asyncio
from aiofilepool import FilePool


async def main():
    # at most 128 real OS fds open at once
    async with FilePool(descriptor_pool_size=128) as pool:
        handle = await pool.open("data.bin", "w+")
        await handle.write(b"hello world")
        await handle.seek(0)
        print(await handle.read())  # b"hello world"
        await handle.fsync()

        # handles auto-close on exit from pool. Also you can do this manually:
        await handle.close()

        # or with a manager:
        async with pool.open("data.bin", "w+") as fd:
            ...


asyncio.run(main())
```

It bounds only the real fds and manages temporary closing and re-opening them, so this is completely fine:

```python
async with FilePool(descriptor_pool_size=64) as pool:
    handles = [await pool.open(f"part-{i}.bin", "w+") for i in range(1000)]
    await asyncio.gather(*(h.write(payload) for h in handles))
```

Also you can read a large file in chunks:

```python
handle = await pool.open("big.bin", "r")
async for chunk in handle.chunks():
    process(chunk)
```

> Although, the handle.read uses chunking too, so chunks are there just for convenience

## API

- **`FilePool`** - the pool: `open(path, mode)`, `manage(binary_io)`, `close()`, and an async context manager.
- **`FileHandle`** - an async file handle: `read`, `write`, `seek`, `tell`, `size` etc. A handler you use to talk to a file.
- **`AsyncBinaryIO`** - the async binary-IO protocol `FileHandle` implements.
- **`BinaryIOAdapter`** - allows to wrap an existing synchronous `BinaryIO` as an `AsyncBinaryIO`.
- **`ModeSpec`** - a parsed file open-mode.
- **`Chunker`, `FixedChunker`, `BalancedChunker`** - chunking strategies. You can use them with `AsyncBinaryIO.chunks` or as a default for the `FilePool`.
- **`aiofilepool.errors`** - the exception hierarchy.

## License

MIT — see [LICENSE](LICENSE).
