"""Property-based tests for `FixedChunker` and `BalancedChunker`.

Hypothesis property tests verify the load-bearing chunker invariants:

* `sum(chunks) == data_size`
* every chunk is `> 0`
* every chunk is `<= max_chunk_size` (BalancedChunker)
* min/max bounds are respected when `data_size` allows

Plus a small set of constructor-validation example tests retained verbatim.
"""

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from aiofilepool._chunking import BalancedChunker, FixedChunker


# --- FixedChunker -------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [0, -1, -32])
def test_fixed_chunker_rejects_non_positive_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        FixedChunker(chunk_size)


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    chunk_size=st.integers(min_value=1, max_value=4096),
    data_size=st.integers(min_value=0, max_value=4096 * 8 + 17),
)
def test_fixed_chunker_invariants(chunk_size: int, data_size: int) -> None:
    chunks = list(FixedChunker(chunk_size)(data_size))
    assert sum(chunks) == data_size
    if chunks:
        assert all(0 < c <= chunk_size for c in chunks)
        assert all(c == chunk_size for c in chunks[:-1])


# --- BalancedChunker ----------------------------------------------------------


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


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    data_size=st.integers(min_value=1, max_value=8192),
    min_chunk_size=st.integers(min_value=8, max_value=64),
    max_chunk_size=st.integers(min_value=64, max_value=512),
    scale_up=st.floats(min_value=1.5, max_value=4.0, allow_nan=False),
    scale_down=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
)
def test_balanced_chunker_invariants(
    data_size: int,
    min_chunk_size: int,
    max_chunk_size: int,
    scale_up: float,
    scale_down: float,
) -> None:
    if min_chunk_size > max_chunk_size:
        return
    chunker = BalancedChunker(
        scale_up=scale_up,
        scale_down=scale_down,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
    )
    chunks = list(chunker(data_size))
    assert sum(chunks) == data_size
    assert all(c > 0 for c in chunks)
    assert all(c <= max_chunk_size for c in chunks)


def test_balanced_chunker_zero_data_size_emits_no_chunks() -> None:
    chunker = BalancedChunker(
        scale_up=2.0, scale_down=0.5, min_chunk_size=8, max_chunk_size=32
    )
    assert list(chunker(0)) == []


def test_balanced_chunker_exact_min_chunk_size_emits_single_chunk() -> None:
    chunker = BalancedChunker(
        scale_up=2.0, scale_down=0.5, min_chunk_size=8, max_chunk_size=32
    )
    assert list(chunker(8)) == [8]
