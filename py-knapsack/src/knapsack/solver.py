"""
High-level KnapsackSolver API.

This module provides a user-friendly interface for solving knapsack problems.
"""

from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import Self

from knapsack.algorithms import Algorithm
from knapsack.models import Item, KnapsackResult, validate_capacity, validate_items


class KnapsackSolver:
    """
    High-level solver for the 0/1 knapsack problem.

    Supports multiple algorithms and provides both direct and fluent APIs.

    Examples:
        Direct API:
        >>> solver = KnapsackSolver()
        >>> items = [Item(10, 60), Item(20, 100), Item(30, 120)]
        >>> result = solver.solve(items, capacity=50)
        >>> result.total_value
        220

        Fluent API:
        >>> result = (
        ...     KnapsackSolver()
        ...     .with_items(items)
        ...     .with_capacity(50)
        ...     .using(Algorithm.BRANCH_AND_BOUND)
        ...     .solve()
        ... )
    """

    def __init__(self, default_algorithm: Algorithm = Algorithm.DYNAMIC_PROGRAMMING) -> None:
        """
        Initialize the solver.

        Args:
            default_algorithm: Algorithm to use when not specified per-call.
        """
        self._default_algorithm = default_algorithm
        self._items: tuple[Item, ...] | None = None
        self._capacity: int | None = None
        self._algorithm: Algorithm | None = None

    def solve(
        self,
        items: Sequence[Item] | None = None,
        *,
        capacity: int | None = None,
        algorithm: Algorithm | None = None,
    ) -> KnapsackResult:
        """
        Solve the knapsack problem.

        Args:
            items: Items to consider (or use items set via with_items).
            capacity: Knapsack capacity (or use capacity set via with_capacity).
            algorithm: Algorithm to use (or use default/fluent-configured algorithm).

        Returns:
            KnapsackResult with the solution.

        Raises:
            ValueError: If items or capacity not provided.
            TypeError: If items contains non-Item objects.
        """
        # Resolve parameters from direct args or fluent state
        resolved_items = items if items is not None else self._items
        resolved_capacity = capacity if capacity is not None else self._capacity
        resolved_algorithm = algorithm or self._algorithm or self._default_algorithm

        if resolved_items is None:
            raise ValueError("Items must be provided either to solve() or via with_items()")
        if resolved_capacity is None:
            raise ValueError("Capacity must be provided either to solve() or via with_capacity()")

        # Validate inputs
        validated_items = validate_items(resolved_items)
        validated_capacity = validate_capacity(resolved_capacity)

        # Get solver function and execute
        solver_fn = resolved_algorithm.get_solver()
        return solver_fn(validated_items, validated_capacity)

    def with_items(self, items: Sequence[Item]) -> Self:
        """
        Set items for fluent API.

        Args:
            items: Items to consider.

        Returns:
            Self for chaining.
        """
        self._items = validate_items(items)
        return self

    def with_capacity(self, capacity: int) -> Self:
        """
        Set capacity for fluent API.

        Args:
            capacity: Knapsack capacity.

        Returns:
            Self for chaining.
        """
        self._capacity = validate_capacity(capacity)
        return self

    def using(self, algorithm: Algorithm) -> Self:
        """
        Set algorithm for fluent API.

        Args:
            algorithm: Algorithm to use.

        Returns:
            Self for chaining.
        """
        self._algorithm = algorithm
        return self
