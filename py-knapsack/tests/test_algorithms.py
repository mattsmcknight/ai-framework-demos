"""
Tests for knapsack algorithms.

Each algorithm should produce correct (or approximately correct for greedy)
results for the 0/1 knapsack problem.
"""

import pytest

from knapsack.algorithms import (
    Algorithm,
    branch_and_bound,
    dynamic_programming,
    dynamic_programming_optimized,
    greedy_approximation,
)
from knapsack.models import Item


class AlgorithmTestBase:
    """Base class for algorithm tests with common fixtures."""

    @pytest.fixture
    def simple_items(self) -> list[Item]:
        """Simple test case with known optimal solution."""
        return [
            Item(weight=10, value=60, name="A"),
            Item(weight=20, value=100, name="B"),
            Item(weight=30, value=120, name="C"),
        ]

    @pytest.fixture
    def single_item(self) -> list[Item]:
        """Single item test case."""
        return [Item(weight=10, value=50)]

    @pytest.fixture
    def all_fit_items(self) -> list[Item]:
        """All items fit in knapsack."""
        return [
            Item(weight=5, value=30),
            Item(weight=10, value=60),
            Item(weight=5, value=40),
        ]

    @pytest.fixture
    def none_fit_items(self) -> list[Item]:
        """No items fit in knapsack."""
        return [
            Item(weight=100, value=500),
            Item(weight=200, value=800),
        ]

    @pytest.fixture
    def edge_case_items(self) -> list[Item]:
        """Edge case with very different value densities."""
        return [
            Item(weight=1, value=1, name="low_density"),
            Item(weight=2, value=100, name="high_density"),
            Item(weight=3, value=2, name="medium_density"),
        ]


class TestDynamicProgramming(AlgorithmTestBase):
    """Tests for the standard DP algorithm."""

    def test_simple_case(self, simple_items):
        """Should solve simple knapsack optimally.

        With capacity 50: items B(20,100) + C(30,120) = 220 value
        """
        result = dynamic_programming(simple_items, capacity=50)

        assert result.total_value == 220
        assert result.total_weight == 50
        assert result.is_optimal is True
        assert result.algorithm == "dynamic_programming"

    def test_single_item_fits(self, single_item):
        """Should select single item when it fits."""
        result = dynamic_programming(single_item, capacity=10)

        assert result.total_value == 50
        assert result.total_weight == 10
        assert len(result.selected_items) == 1

    def test_single_item_doesnt_fit(self, single_item):
        """Should return empty when single item doesn't fit."""
        result = dynamic_programming(single_item, capacity=5)

        assert result.total_value == 0
        assert result.total_weight == 0
        assert len(result.selected_items) == 0

    def test_all_items_fit(self, all_fit_items):
        """Should take all items when they all fit."""
        result = dynamic_programming(all_fit_items, capacity=100)

        assert result.total_value == 130  # 30 + 60 + 40
        assert result.total_weight == 20  # 5 + 10 + 5
        assert len(result.selected_items) == 3

    def test_no_items_fit(self, none_fit_items):
        """Should return empty when no items fit."""
        result = dynamic_programming(none_fit_items, capacity=50)

        assert result.total_value == 0
        assert len(result.selected_items) == 0

    def test_empty_items_list(self):
        """Should handle empty items list."""
        result = dynamic_programming([], capacity=50)

        assert result.total_value == 0
        assert result.total_weight == 0
        assert len(result.selected_items) == 0

    def test_zero_capacity(self, simple_items):
        """Should return empty for zero capacity."""
        result = dynamic_programming(simple_items, capacity=0)

        assert result.total_value == 0
        assert len(result.selected_items) == 0

    def test_exact_fit(self):
        """Should find solution that exactly fills capacity."""
        items = [
            Item(weight=3, value=30),
            Item(weight=4, value=40),
            Item(weight=5, value=50),
        ]
        result = dynamic_programming(items, capacity=7)

        # Optimal: 3 + 4 = 7 weight, 70 value
        assert result.total_value == 70
        assert result.total_weight == 7


class TestDynamicProgrammingOptimized(AlgorithmTestBase):
    """Tests for the space-optimized DP algorithm."""

    def test_simple_case(self, simple_items):
        """Should produce same result as standard DP."""
        result = dynamic_programming_optimized(simple_items, capacity=50)

        assert result.total_value == 220
        assert result.is_optimal is True
        assert result.algorithm == "dynamic_programming_optimized"

    def test_matches_standard_dp(self, simple_items):
        """Results should match standard DP algorithm."""
        dp_result = dynamic_programming(simple_items, capacity=50)
        opt_result = dynamic_programming_optimized(simple_items, capacity=50)

        assert dp_result.total_value == opt_result.total_value
        assert dp_result.total_weight == opt_result.total_weight

    def test_large_capacity(self):
        """Should handle large capacities efficiently."""
        items = [Item(weight=i, value=i * 10) for i in range(1, 20)]
        result = dynamic_programming_optimized(items, capacity=1000)

        assert result.total_value > 0
        assert result.is_optimal is True


class TestBranchAndBound(AlgorithmTestBase):
    """Tests for the branch and bound algorithm."""

    def test_simple_case(self, simple_items):
        """Should solve simple knapsack optimally."""
        result = branch_and_bound(simple_items, capacity=50)

        assert result.total_value == 220
        assert result.is_optimal is True
        assert result.algorithm == "branch_and_bound"

    def test_matches_dp(self, simple_items):
        """Results should match DP algorithm."""
        dp_result = dynamic_programming(simple_items, capacity=50)
        bb_result = branch_and_bound(simple_items, capacity=50)

        assert dp_result.total_value == bb_result.total_value

    def test_prunes_search_space(self, simple_items):
        """Should evaluate fewer items than brute force."""
        result = branch_and_bound(simple_items, capacity=50)

        # Brute force would evaluate 2^3 = 8 combinations
        # B&B should prune some branches
        assert result.items_evaluated < 8

    def test_empty_items(self):
        """Should handle empty items list."""
        result = branch_and_bound([], capacity=50)

        assert result.total_value == 0


class TestGreedyApproximation(AlgorithmTestBase):
    """Tests for the greedy approximation algorithm."""

    def test_simple_case(self, simple_items):
        """Should return valid (possibly non-optimal) solution."""
        result = greedy_approximation(simple_items, capacity=50)

        assert result.total_weight <= 50
        assert result.total_value > 0
        assert result.is_optimal is False
        assert result.algorithm == "greedy_approximation"

    def test_selects_by_value_density(self, edge_case_items):
        """Should prioritize high value density items."""
        result = greedy_approximation(edge_case_items, capacity=5)

        # High density item (w=2, v=100) should be selected
        selected_names = [item.name for item in result.selected_items]
        assert "high_density" in selected_names

    def test_fast_on_large_inputs(self):
        """Greedy should be fast on large inputs."""
        items = [Item(weight=i % 100 + 1, value=i * 5) for i in range(1000)]
        result = greedy_approximation(items, capacity=10000)

        assert result.total_value > 0

    def test_empty_items(self):
        """Should handle empty items list."""
        result = greedy_approximation([], capacity=50)

        assert result.total_value == 0


class TestAlgorithmEnum:
    """Tests for the Algorithm enum."""

    def test_all_algorithms_exist(self):
        """Should have all expected algorithm variants."""
        assert Algorithm.DYNAMIC_PROGRAMMING
        assert Algorithm.DYNAMIC_PROGRAMMING_OPTIMIZED
        assert Algorithm.BRANCH_AND_BOUND
        assert Algorithm.GREEDY

    def test_algorithm_value_is_string(self):
        """Algorithm values should be string identifiers."""
        assert isinstance(Algorithm.DYNAMIC_PROGRAMMING.value, str)

    def test_get_solver_function(self):
        """Should return correct solver function for each algorithm."""
        assert Algorithm.DYNAMIC_PROGRAMMING.get_solver() == dynamic_programming
        assert Algorithm.BRANCH_AND_BOUND.get_solver() == branch_and_bound
        assert Algorithm.GREEDY.get_solver() == greedy_approximation


class TestAlgorithmCorrectnessMatrix:
    """
    Cross-validation tests ensuring all optimal algorithms agree.
    """

    @pytest.fixture
    def optimal_algorithms(self):
        """Algorithms that guarantee optimal solutions."""
        return [
            dynamic_programming,
            dynamic_programming_optimized,
            branch_and_bound,
        ]

    @pytest.mark.parametrize(
        "items,capacity,expected_value",
        [
            # Classic textbook example
            (
                [Item(10, 60), Item(20, 100), Item(30, 120)],
                50,
                220,
            ),
            # All items fit
            (
                [Item(5, 50), Item(5, 50)],
                100,
                100,
            ),
            # Single item
            (
                [Item(1, 1)],
                1,
                1,
            ),
            # No items fit
            (
                [Item(100, 100)],
                50,
                0,
            ),
        ],
    )
    def test_all_optimal_algorithms_agree(
        self, optimal_algorithms, items, capacity, expected_value
    ):
        """All optimal algorithms should produce the same value."""
        for algorithm in optimal_algorithms:
            result = algorithm(items, capacity)
            assert result.total_value == expected_value, (
                f"{algorithm.__name__} returned {result.total_value}, "
                f"expected {expected_value}"
            )
