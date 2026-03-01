from aiofilepool import FilePool
import asyncio
import time


async def normal():
    with (
        open("D:\\Downloads\\smallD.mkv", "rb") as f,
        open("D:\\Downloads\\smallD2.mkv", "rb") as f2,
    ):
        f.read()
        f2.read()
    print("done")


async def with_pool():
    async with FilePool() as pool:
        async with (
            pool.open("D:\\Downloads\\smallD.mkv", "r") as f,
            pool.open("D:\\Downloads\\smallD2.mkv", "r") as f2,
        ):
            await asyncio.gather(f.read(), f2.read())
    print("done")


async def main():
    start = time.time()
    await with_pool()
    end = time.time()
    print("Time taken: ", end - start)


if __name__ == "__main__":
    asyncio.run(main())
