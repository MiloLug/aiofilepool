from pathlib import Path
from aiofilepool import FileHandle, FilePool
import asyncio
import time

data = b"a" * 1024 * 1024 * 1024


async def normal():
    base = Path(".test-files")
    for i in range(10):
        with open(base / f"file{i}.bin", "w+b") as f:
            f.write(data)
    print("done")


async def with_pool():
    base = Path(".test-files")
    async with FilePool() as pool:
        fds: list[FileHandle] = []
        for i in range(10):
            fds.append(await pool.open(base / f"file{i}.bin", "w+"))
        await asyncio.gather(*[fd.write(data) for fd in fds])
        print("written")
    print("done")


async def main():
    start = time.time()
    await with_pool()
    end = time.time()
    print("Pool Time taken: ", end - start)
    start = time.time()
    await normal()
    end = time.time()
    print("Normal Time taken: ", end - start)


if __name__ == "__main__":
    asyncio.run(main())
