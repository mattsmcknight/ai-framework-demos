"""
Tests for the high-level KnapsackSolver API.
"""

import pytest

from knapsack import Algorithm, Item, KnapsackResult, KnapsackSolver


class TestKnapsackSolver:
    """Tests for the KnapsackSolver class."""

    @pytest.fixture
    def solver(self):
        """Default solver instance."""
        return KnapsackSolver()

    @pytest.fixture
    def standard_items(self):
        """Standard test items."""
        return [
            Item(weight=10, value=60, name="A"),
            Item(weight=20, value=100, name="B"),
            Item(weight=30, value=120, name="C"),
        ]

    def test_default_algorithm(self, solver, standard_items):
        """Should use DP by default."""
        result = solver.solve(standard_items, capacity=50)

        assert result.algorithm == "dynamic_programming"
        assert result.total_value == 220

    def test_specify_algorithm(self, solver, standard_items):
        """Should allow algorithm selection."""
        result = solver.solve(
            standard_items,
            capacity=50,
            algorithm=Algorithm.BRANCH_AND_BOUND,
        )

        assert result.algorithm == "branch_and_bound"
        assert result.total_value == 220

    def test_solve_returns_result_object(self, solver, standard_items):
        """Should return KnapsackResult instance."""
        result = solver.solve(standard_items, capacity=50)

        assert isinstance(result, KnapsackResult)
        assert hasattr(result, "selected_items")
        assert hasattr(result, "total_value")
        assert hasattr(result, "total_weight")

    def test_result_contains_selected_items(self, solver, standard_items):
        """Selected items should be Item instances."""
        result = solver.solve(standard_items, capacity=50)

        assert all(isinstance(item, Item) for item in result.selected_items)

    def test_invalid_capacity_raises(self, solver, standard_items):
        """Should raise error for invalid capacity."""
        with pytest.raises(ValueError):
            solver.solve(standard_items, capacity=-10)

    def test_invalid_items_raises(self, solver):
        """Should raise error for invalid items."""
        with pytest.raises(TypeError):
            solver.solve(["not", "items"], capacity=50)  # type: ignore

    def test_empty_items_returns_empty_result(self, solver):
        """Should handle empty items list gracefully."""
        result = solver.solve([], capacity=50)

        assert result.total_value == 0
        assert result.total_weight == 0
        assert len(result.selected_items) == 0

    def test_fluent_api(self, standard_items):
        """Should support fluent/builder pattern."""
        result = (
            KnapsackSolver()
            .with_items(standard_items)
            .with_capacity(50)
            .using(Algorithm.DYNAMIC_PROGRAMMING)
            .solve()
        )

        assert result.total_value == 220


class TestSolverConfiguration:
    """Tests for solver configuration options."""

    def test_create_with_default_algorithm(self):
        """Should accept default algorithm in constructor."""
        solver = KnapsackSolver(default_algorithm=Algorithm.GREEDY)

        items = [Item(10, 60), Item(20, 100)]
        result = solver.solve(items, capacity=30)

        assert result.algorithm == "greedy_approximation"

    def test_per_call_algorithm_overrides_default(self):
        """Per-call algorithm should override constructor default."""
        solver = KnapsackSolver(default_algorithm=Algorithm.GREEDY)

        items = [Item(10, 60), Item(20, 100)]
        result = solver.solve(items, capacity=30, algorithm=Algorithm.DYNAMIC_PROGRAMMING)

        assert result.algorithm == "dynamic_programming"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def solver(self):
        return KnapsackSolver()

    def test_single_item_exactly_fits(self, solver):
        """Item weight equals capacity."""
        items = [Item(weight=50, value=100)]
        result = solver.solve(items, capacity=50)

        assert result.total_value == 100
        assert result.total_weight == 50

    def test_single_item_too_heavy(self, solver):
        """Item weight exceeds capacity."""
        items = [Item(weight=51, value=100)]
        result = solver.solve(items, capacity=50)

        assert result.total_value == 0

    def test_zero_capacity(self, solver):
        """Zero capacity should return empty result."""
        items = [Item(1, 100)]
        result = solver.solve(items, capacity=0)

        assert result.total_value == 0
        assert len(result.selected_items) == 0

    def test_duplicate_items(self, solver):
        """Should handle items with identical properties."""
        items = [
            Item(weight=10, value=50),
            Item(weight=10, value=50),
            Item(weight=10, value=50),
        ]
        result = solver.solve(items, capacity=25)

        # Should select 2 items (weight=20, value=100)
        assert result.total_value == 100
        assert result.total_weight == 20

    def test_many_small_items(self, solver):
        """Should handle many items efficiently."""
        items = [Item(weight=1, value=i) for i in range(100)]
        result = solver.solve(items, capacity=50)

        # Should take the 50 highest value items (indices 50-99)
        # Values: 50+51+...+99 = sum(50..99)
        expected = sum(range(50, 100))
        assert result.total_value == expected

    def test_large_values_and_weights(self, solver):
        """Should handle large numeric values."""
        items = [
            Item(weight=1000000, value=5000000),
            Item(weight=2000000, value=8000000),
        ]
        result = solver.solve(items, capacity=3000000)

        assert result.total_value == 13000000
