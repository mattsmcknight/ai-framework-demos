"""
Knapsack algorithm implementations.

This module provides multiple algorithms for solving the 0/1 knapsack problem:
- Dynamic Programming (standard and space-optimized)
- Branch and Bound
- Greedy Approximation

All algorithms share a common interface and return KnapsackResult objects.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from knapsack.models import Item, KnapsackResult

# Type alias for solver functions
SolverFunction = Callable[[Sequence[Item], int], KnapsackResult]


class Algorithm(Enum):
    """
    Available knapsack algorithms.

    Attributes:
        DYNAMIC_PROGRAMMING: Standard DP with O(n*W) time and space.
        DYNAMIC_PROGRAMMING_OPTIMIZED: Space-optimized DP with O(W) space.
        BRANCH_AND_BOUND: Branch and bound with pruning.
        GREEDY: Fast approximation using value density.
    """

    DYNAMIC_PROGRAMMING = "dynamic_programming"
    DYNAMIC_PROGRAMMING_OPTIMIZED = "dynamic_programming_optimized"
    BRANCH_AND_BOUND = "branch_and_bound"
    GREEDY = "greedy_approximation"

    def get_solver(self) -> SolverFunction:
        """Return the solver function for this algorithm."""
        mapping: dict[Algorithm, SolverFunction] = {
            Algorithm.DYNAMIC_PROGRAMMING: dynamic_programming,
            Algorithm.DYNAMIC_PROGRAMMING_OPTIMIZED: dynamic_programming_optimized,
            Algorithm.BRANCH_AND_BOUND: branch_and_bound,
            Algorithm.GREEDY: greedy_approximation,
        }
        return mapping[self]


def dynamic_programming(items: Sequence[Item], capacity: int) -> KnapsackResult:
    """
    Solve 0/1 knapsack using standard dynamic programming.

    Time complexity: O(n * W) where n = items, W = capacity
    Space complexity: O(n * W)

    Args:
        items: Sequence of items to consider.
        capacity: Maximum weight capacity.

    Returns:
        KnapsackResult with optimal solution.

    Examples:
        >>> items = [Item(10, 60), Item(20, 100), Item(30, 120)]
        >>> result = dynamic_programming(items, 50)
        >>> result.total_value
        220
    """
    n = len(items)

    if n == 0 or capacity == 0:
        return KnapsackResult(
            selected_items=(),
            total_value=0,
            total_weight=0,
            capacity=capacity,
            algorithm="dynamic_programming",
            is_optimal=True,
        )

    # Build DP table
    # dp[i][w] = max value achievable using items[0..i-1] with capacity w
    dp: list[list[int]] = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        item = items[i - 1]
        for w in range(capacity + 1):
            # Don't take item i
            dp[i][w] = dp[i - 1][w]

            # Take item i if it fits and improves value
            if item.weight <= w:
                value_with_item = dp[i - 1][w - item.weight] + item.value
                dp[i][w] = max(dp[i][w], value_with_item)

    # Backtrack to find selected items
    selected: list[Item] = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            # Item i-1 was selected
            item = items[i - 1]
            selected.append(item)
            w -= item.weight

    selected.reverse()
    total_value = dp[n][capacity]
    total_weight = sum(item.weight for item in selected)

    return KnapsackResult(
        selected_items=tuple(selected),
        total_value=total_value,
        total_weight=total_weight,
        capacity=capacity,
        algorithm="dynamic_programming",
        is_optimal=True,
    )


def dynamic_programming_optimized(items: Sequence[Item], capacity: int) -> KnapsackResult:
    """
    Solve 0/1 knapsack using space-optimized dynamic programming.

    Uses only O(W) space by keeping only the previous row.
    Reconstructs the solution using a separate backtracking pass.

    Time complexity: O(n * W)
    Space complexity: O(W)

    Args:
        items: Sequence of items to consider.
        capacity: Maximum weight capacity.

    Returns:
        KnapsackResult with optimal solution.
    """
    n = len(items)

    if n == 0 or capacity == 0:
        return KnapsackResult(
            selected_items=(),
            total_value=0,
            total_weight=0,
            capacity=capacity,
            algorithm="dynamic_programming_optimized",
            is_optimal=True,
        )

    # Use 1D array, process in reverse to avoid overwriting needed values
    dp: list[int] = [0] * (capacity + 1)

    # Track which items were selected at each state
    # This is a bit more complex with 1D, so we'll track decision points
    selected_at: list[list[bool]] = [[False] * (capacity + 1) for _ in range(n)]

    for i, item in enumerate(items):
        # Process in reverse to ensure we use previous row's values
        for w in range(capacity, item.weight - 1, -1):
            value_with_item = dp[w - item.weight] + item.value
            if value_with_item > dp[w]:
                dp[w] = value_with_item
                selected_at[i][w] = True

    # Backtrack to find selected items
    selected: list[Item] = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if selected_at[i][w]:
            item = items[i]
            selected.append(item)
            w -= item.weight

    selected.reverse()
    total_value = dp[capacity]
    total_weight = sum(item.weight for item in selected)

    return KnapsackResult(
        selected_items=tuple(selected),
        total_value=total_value,
        total_weight=total_weight,
        capacity=capacity,
        algorithm="dynamic_programming_optimized",
        is_optimal=True,
    )


@dataclass
class _BBNode:
    """Node in the branch and bound search tree."""

    level: int  # Current item index
    value: int  # Value accumulated so far
    weight: int  # Weight accumulated so far
    bound: float  # Upper bound on maximum value achievable
    selected: tuple[int, ...]  # Indices of selected items

    def __lt__(self, other: _BBNode) -> bool:
        # Max-heap: higher bound = higher priority
        return self.bound > other.bound


def _compute_bound(
    node: _BBNode, items: Sequence[Item], capacity: int, items_by_density: list[int]
) -> float:
    """
    Compute upper bound using fractional knapsack relaxation.

    This bound is tight and allows effective pruning.
    """
    if node.weight >= capacity:
        return 0.0

    bound = float(node.value)
    remaining_capacity = capacity - node.weight

    # Greedily add items by value density (fractional relaxation)
    for idx in items_by_density:
        if idx <= node.level:
            continue

        item = items[idx]
        if item.weight <= remaining_capacity:
            bound += item.value
            remaining_capacity -= item.weight
        else:
            # Take fraction of this item
            bound += item.value * (remaining_capacity / item.weight)
            break

    return bound


def branch_and_bound(items: Sequence[Item], capacity: int) -> KnapsackResult:
    """
    Solve 0/1 knapsack using branch and bound with best-first search.

    Uses fractional knapsack relaxation for upper bound computation.
    More efficient than brute force due to pruning.

    Time complexity: O(2^n) worst case, but typically much better due to pruning.
    Space complexity: O(2^n) worst case for priority queue.

    Args:
        items: Sequence of items to consider.
        capacity: Maximum weight capacity.

    Returns:
        KnapsackResult with optimal solution and items_evaluated count.
    """
    n = len(items)

    if n == 0 or capacity == 0:
        return KnapsackResult(
            selected_items=(),
            total_value=0,
            total_weight=0,
            capacity=capacity,
            algorithm="branch_and_bound",
            is_optimal=True,
            items_evaluated=0,
        )

    # Sort items by value density for bound computation
    items_by_density = sorted(range(n), key=lambda i: items[i].value_density(), reverse=True)

    # Create root node
    root = _BBNode(level=-1, value=0, weight=0, bound=0.0, selected=())
    root.bound = _compute_bound(root, items, capacity, items_by_density)

    best_value = 0
    best_selection: tuple[int, ...] = ()
    nodes_evaluated = 0

    # Priority queue (max-heap via negative comparison)
    pq: list[_BBNode] = [root]

    while pq:
        node = heapq.heappop(pq)
        nodes_evaluated += 1

        # Pruning: skip if bound can't beat current best
        if node.bound <= best_value:
            continue

        # Can't go deeper
        if node.level >= n - 1:
            continue

        next_level = node.level + 1
        item = items[next_level]

        # Branch 1: Include item (if it fits)
        if node.weight + item.weight <= capacity:
            new_value = node.value + item.value
            new_weight = node.weight + item.weight
            new_selected = node.selected + (next_level,)

            if new_value > best_value:
                best_value = new_value
                best_selection = new_selected

            include_node = _BBNode(
                level=next_level,
                value=new_value,
                weight=new_weight,
                bound=0.0,
                selected=new_selected,
            )
            include_node.bound = _compute_bound(include_node, items, capacity, items_by_density)

            if include_node.bound > best_value:
                heapq.heappush(pq, include_node)

        # Branch 2: Exclude item
        exclude_node = _BBNode(
            level=next_level,
            value=node.value,
            weight=node.weight,
            bound=0.0,
            selected=node.selected,
        )
        exclude_node.bound = _compute_bound(exclude_node, items, capacity, items_by_density)

        if exclude_node.bound > best_value:
            heapq.heappush(pq, exclude_node)

    # Build result from best selection
    selected = tuple(items[i] for i in best_selection)
    total_weight = sum(item.weight for item in selected)

    return KnapsackResult(
        selected_items=selected,
        total_value=best_value,
        total_weight=total_weight,
        capacity=capacity,
        algorithm="branch_and_bound",
        is_optimal=True,
        items_evaluated=nodes_evaluated,
    )


def greedy_approximation(items: Sequence[Item], capacity: int) -> KnapsackResult:
    """
    Solve 0/1 knapsack using greedy approximation by value density.

    This is a fast O(n log n) approximation that may not find the optimal solution.
    Useful for large instances where optimal algorithms are too slow.

    The approximation ratio is not bounded for 0/1 knapsack (unlike fractional).

    Time complexity: O(n log n) for sorting
    Space complexity: O(n)

    Args:
        items: Sequence of items to consider.
        capacity: Maximum weight capacity.

    Returns:
        KnapsackResult with approximate solution (is_optimal=False).

    Examples:
        >>> items = [Item(10, 60), Item(20, 100), Item(30, 120)]
        >>> result = greedy_approximation(items, 50)
        >>> result.is_optimal
        False
    """
    if not items or capacity == 0:
        return KnapsackResult(
            selected_items=(),
            total_value=0,
            total_weight=0,
            capacity=capacity,
            algorithm="greedy_approximation",
            is_optimal=False,
        )

    # Sort by value density (descending)
    indexed_items = list(enumerate(items))
    indexed_items.sort(key=lambda x: x[1].value_density(), reverse=True)

    selected: list[Item] = []
    total_weight = 0
    total_value = 0

    for _, item in indexed_items:
        if total_weight + item.weight <= capacity:
            selected.append(item)
            total_weight += item.weight
            total_value += item.value

    return KnapsackResult(
        selected_items=tuple(selected),
        total_value=total_value,
        total_weight=total_weight,
        capacity=capacity,
        algorithm="greedy_approximation",
        is_optimal=False,
    )
