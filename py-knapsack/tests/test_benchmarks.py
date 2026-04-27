"""
Benchmark tests for algorithm performance comparison.

Run with: pytest tests/test_benchmarks.py --benchmark-only
"""

import pytest

from knapsack import Item
from knapsack.algorithms import (
    branch_and_bound,
    dynamic_programming,
    dynamic_programming_optimized,
    greedy_approximation,
)


@pytest.fixture
def small_dataset():
    """10 items, small capacity."""
    items = [Item(weight=i * 3 + 1, value=i * 10 + 5) for i in range(10)]
    return items, 50


@pytest.fixture
def medium_dataset():
    """50 items, medium capacity."""
    items = [Item(weight=(i % 20) + 1, value=i * 5 + 10) for i in range(50)]
    return items, 200


@pytest.fixture
def large_dataset():
    """200 items, large capacity."""
    items = [Item(weight=(i % 50) + 1, value=i * 3 + 7) for i in range(200)]
    return items, 1000


class TestBenchmarkSmall:
    """Benchmarks on small dataset (all algorithms)."""

    def test_dp_small(self, benchmark, small_dataset):
        items, capacity = small_dataset
        result = benchmark(dynamic_programming, items, capacity)
        assert result.is_optimal

    def test_dp_optimized_small(self, benchmark, small_dataset):
        items, capacity = small_dataset
        result = benchmark(dynamic_programming_optimized, items, capacity)
        assert result.is_optimal

    def test_branch_and_bound_small(self, benchmark, small_dataset):
        items, capacity = small_dataset
        result = benchmark(branch_and_bound, items, capacity)
        assert result.is_optimal

    def test_greedy_small(self, benchmark, small_dataset):
        items, capacity = small_dataset
        result = benchmark(greedy_approximation, items, capacity)
        assert result.total_value > 0


class TestBenchmarkMedium:
    """Benchmarks on medium dataset."""

    def test_dp_medium(self, benchmark, medium_dataset):
        items, capacity = medium_dataset
        result = benchmark(dynamic_programming, items, capacity)
        assert result.is_optimal

    def test_dp_optimized_medium(self, benchmark, medium_dataset):
        items, capacity = medium_dataset
        result = benchmark(dynamic_programming_optimized, items, capacity)
        assert result.is_optimal

    def test_greedy_medium(self, benchmark, medium_dataset):
        items, capacity = medium_dataset
        result = benchmark(greedy_approximation, items, capacity)
        assert result.total_value > 0


class TestBenchmarkLarge:
    """Benchmarks on large dataset."""

    def test_dp_large(self, benchmark, large_dataset):
        items, capacity = large_dataset
        result = benchmark(dynamic_programming, items, capacity)
        assert result.is_optimal

    def test_dp_optimized_large(self, benchmark, large_dataset):
        items, capacity = large_dataset
        result = benchmark(dynamic_programming_optimized, items, capacity)
        assert result.is_optimal

    def test_greedy_large(self, benchmark, large_dataset):
        items, capacity = large_dataset
        result = benchmark(greedy_approximation, items, capacity)
        assert result.total_value > 0
