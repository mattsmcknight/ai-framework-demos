"""
Property-based tests using Hypothesis.

These tests verify invariants that should hold for any valid input.
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from knapsack import Algorithm, Item, KnapsackSolver
from knapsack.algorithms import dynamic_programming, greedy_approximation

# Custom strategies for generating valid items
item_strategy = st.builds(
    Item,
    weight=st.integers(min_value=1, max_value=1000),
    value=st.integers(min_value=0, max_value=10000),
    name=st.none(),
)

items_strategy = st.lists(item_strategy, min_size=0, max_size=50)
capacity_strategy = st.integers(min_value=0, max_value=10000)


class TestInvariants:
    """Property-based tests for solver invariants."""

    @given(items=items_strategy, capacity=capacity_strategy)
    @settings(max_examples=100)
    def test_result_weight_never_exceeds_capacity(self, items, capacity):
        """The total weight of selected items must never exceed capacity."""
        solver = KnapsackSolver()
        result = solver.solve(items, capacity=capacity)

        assert result.total_weight <= capacity

    @given(items=items_strategy, capacity=capacity_strategy)
    @settings(max_examples=100)
    def test_total_weight_matches_sum_of_selected(self, items, capacity):
        """Total weight must equal sum of selected item weights."""
        solver = KnapsackSolver()
        result = solver.solve(items, capacity=capacity)

        computed_weight = sum(item.weight for item in result.selected_items)
        assert result.total_weight == computed_weight

    @given(items=items_strategy, capacity=capacity_strategy)
    @settings(max_examples=100)
    def test_total_value_matches_sum_of_selected(self, items, capacity):
        """Total value must equal sum of selected item values."""
        solver = KnapsackSolver()
        result = solver.solve(items, capacity=capacity)

        computed_value = sum(item.value for item in result.selected_items)
        assert result.total_value == computed_value

    @given(items=items_strategy, capacity=capacity_strategy)
    @settings(max_examples=100)
    def test_selected_items_are_subset_of_input(self, items, capacity):
        """Selected items must be a subset of input items."""
        solver = KnapsackSolver()
        result = solver.solve(items, capacity=capacity)

        input_set = set(items)
        for selected in result.selected_items:
            assert selected in input_set

    @given(items=items_strategy, capacity=capacity_strategy)
    @settings(max_examples=100)
    def test_greedy_never_worse_than_empty(self, items, capacity):
        """Greedy solution should never return negative value."""
        result = greedy_approximation(items, capacity)

        assert result.total_value >= 0

    @given(items=items_strategy, capacity=capacity_strategy)
    @settings(max_examples=50)
    def test_dp_result_is_optimal_upper_bound(self, items, capacity):
        """DP (optimal) should always >= greedy result."""
        dp_result = dynamic_programming(items, capacity)
        greedy_result = greedy_approximation(items, capacity)

        assert dp_result.total_value >= greedy_result.total_value


class TestMonotonicity:
    """Tests for monotonic properties."""

    @given(items=items_strategy, capacity=capacity_strategy)
    @settings(max_examples=50)
    def test_larger_capacity_yields_geq_value(self, items, capacity):
        """Larger capacity should never decrease optimal value."""
        assume(capacity > 0)

        solver = KnapsackSolver()
        result_small = solver.solve(items, capacity=capacity)
        result_large = solver.solve(items, capacity=capacity + 100)

        assert result_large.total_value >= result_small.total_value

    @given(capacity=capacity_strategy)
    @settings(max_examples=50)
    def test_more_items_yields_geq_value(self, capacity):
        """Adding items should never decrease optimal value."""
        assume(capacity > 0)

        items_small = [Item(weight=5, value=10)]
        items_large = items_small + [Item(weight=3, value=8)]

        solver = KnapsackSolver()
        result_small = solver.solve(items_small, capacity=capacity)
        result_large = solver.solve(items_large, capacity=capacity)

        assert result_large.total_value >= result_small.total_value


class TestAlgorithmConsistency:
    """Tests that optimal algorithms always agree."""

    @given(items=items_strategy, capacity=capacity_strategy)
    @settings(max_examples=30)
    def test_all_optimal_algorithms_match(self, items, capacity):
        """All optimal algorithms should return the same value."""
        solver = KnapsackSolver()

        results = [
            solver.solve(items, capacity=capacity, algorithm=Algorithm.DYNAMIC_PROGRAMMING),
            solver.solve(
                items, capacity=capacity, algorithm=Algorithm.DYNAMIC_PROGRAMMING_OPTIMIZED
            ),
            solver.solve(items, capacity=capacity, algorithm=Algorithm.BRANCH_AND_BOUND),
        ]

        values = [r.total_value for r in results]
        assert len(set(values)) == 1, f"Algorithms disagree: {values}"
