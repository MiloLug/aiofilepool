import pytest

from aiofilepool._chunking import BalancedChunker, FixedChunker


@pytest.mark.parametrize("chunk_size", [0, -1, -32])
def test_fixed_chunker_rejects_non_positive_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        FixedChunker(chunk_size)


def test_fixed_chunker_splits_into_equal_chunks_with_remainder() -> None:
    chunker = FixedChunker(4)
    assert list(chunker(10)) == [4, 4, 2]


def test_fixed_chunker_returns_no_chunks_for_empty_data() -> None:
    chunker = FixedChunker(8)
    assert list(chunker(0)) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scale_up": 1}, "scale_up must be > 1"),
        ({"scale_down": 0}, "scale_down must be between 0 and 1"),
        ({"scale_down": 1}, "scale_down must be between 0 and 1"),
        ({"min_chunk_size": 0}, "min_chunk_size must be > 0"),
        ({"max_chunk_size": 0}, "max_chunk_size must be > 0"),
        (
            {"min_chunk_size": 32, "max_chunk_size": 16},
            "min_chunk_size must be <= max_chunk_size",
        ),
        (
            {"scale_up": 1.01, "min_chunk_size": 1, "max_chunk_size": 16},
            "min_chunk_size is too small to scale up",
        ),
    ],
)
def test_balanced_chunker_validates_constructor_inputs(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BalancedChunker(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("data_size", [1, 7, 64, 129])
def test_balanced_chunker_output_is_positive_and_sums_to_size(data_size: int) -> None:
    chunker = BalancedChunker(
        scale_up=1.5,
        scale_down=0.75,
        min_chunk_size=8,
        max_chunk_size=64,
    )

    chunks = list(chunker(data_size))

    assert sum(chunks) == data_size
    assert all(chunk > 0 for chunk in chunks)


def test_balanced_chunker_scaling_path_respects_invariants(
    monkeypatch: pytest.MonkeyPatch,
):
    times = iter([100, 120, 140, 160, 180, 200])
    monkeypatch.setattr(
        "aiofilepool._chunking.time.perf_counter_ns",
        lambda: next(times),
    )

    chunker = BalancedChunker(
        scale_up=2.0,
        scale_down=0.5,
        min_chunk_size=10,
        max_chunk_size=40,
    )

    chunks = list(chunker(70))

    assert sum(chunks) == 70
    assert all(10 <= chunk <= 40 for chunk in chunks)
    assert len(chunks) > 1


def test_balanced_chunker_never_emits_a_chunk_larger_than_max() -> None:
    chunker = BalancedChunker(
        scale_up=2.0,
        scale_down=0.5,
        min_chunk_size=8,
        max_chunk_size=32,
    )

    chunks = list(chunker(512))

    assert max(chunks) <= 32
    assert sum(chunks) == 512


def test_balanced_chunker_exact_threshold_boundary_stays_stable() -> None:
    chunker = BalancedChunker(
        scale_up=2.0,
        scale_down=0.5,
        min_chunk_size=8,
        max_chunk_size=32,
    )

    assert list(chunker(16)) == [16]
