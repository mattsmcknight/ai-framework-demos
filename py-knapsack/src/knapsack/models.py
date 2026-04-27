"""
Data models for the knapsack solver.

This module defines the core data structures used throughout the library.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Item:
    """
    Represents an item that can be placed in the knapsack.

    Attributes:
        weight: The weight of the item (must be positive).
        value: The value/profit of the item (must be non-negative).
        name: Optional identifier for the item.

    Examples:
        >>> item = Item(weight=10, value=60, name="laptop")
        >>> item.weight
        10
        >>> item.value
        60

    Raises:
        ValueError: If weight is not positive or value is negative.
    """

    weight: int
    value: int
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate item constraints after initialization."""
        if self.weight <= 0:
            raise ValueError(f"Item weight must be positive, got {self.weight}")
        if self.value < 0:
            raise ValueError(f"Item value must be non-negative, got {self.value}")

    def value_density(self) -> float:
        """
        Calculate the value-to-weight ratio.

        Returns:
            The value per unit weight.
        """
        return self.value / self.weight

    def __repr__(self) -> str:
        name_part = f", name={self.name!r}" if self.name else ""
        return f"Item(weight={self.weight}, value={self.value}{name_part})"


@dataclass(frozen=True, slots=True)
class KnapsackResult:
    """
    The result of solving a knapsack problem.

    Attributes:
        selected_items: The items selected for the optimal solution.
        total_value: The sum of values of selected items.
        total_weight: The sum of weights of selected items.
        capacity: The knapsack capacity used for this solution.
        algorithm: The name of the algorithm used.
        is_optimal: Whether this is guaranteed to be the optimal solution.

    Examples:
        >>> result = KnapsackResult(
        ...     selected_items=(Item(10, 60), Item(20, 100)),
        ...     total_value=160,
        ...     total_weight=30,
        ...     capacity=50,
        ...     algorithm="dynamic_programming"
        ... )
        >>> result.utilization
        0.6
    """

    selected_items: tuple[Item, ...]
    total_value: int
    total_weight: int
    capacity: int
    algorithm: str
    is_optimal: bool = True
    items_evaluated: int = 0

    @property
    def utilization(self) -> float:
        """
        Calculate what fraction of capacity is used.

        Returns:
            A value between 0.0 and 1.0 representing capacity utilization.
        """
        if self.capacity == 0:
            return 0.0
        return self.total_weight / self.capacity

    @property
    def item_count(self) -> int:
        """Return the number of selected items."""
        return len(self.selected_items)

    def __repr__(self) -> str:
        return (
            f"KnapsackResult(value={self.total_value}, weight={self.total_weight}, "
            f"items={self.item_count}, utilization={self.utilization:.1%})"
        )


def validate_items(items: Sequence[Item]) -> tuple[Item, ...]:
    """
    Validate and normalize a sequence of items.

    Args:
        items: A sequence of Item objects.

    Returns:
        A tuple of validated items.

    Raises:
        TypeError: If items is not iterable or contains non-Item objects.
        ValueError: If items sequence is empty.
    """
    if not hasattr(items, "__iter__"):
        raise TypeError(f"Expected iterable of items, got {type(items).__name__}")

    validated: list[Item] = []
    for i, item in enumerate(items):
        if not isinstance(item, Item):
            raise TypeError(
                f"Expected Item at index {i}, got {type(item).__name__}"
            )
        validated.append(item)

    return tuple(validated)


def validate_capacity(capacity: int) -> int:
    """
    Validate the knapsack capacity.

    Args:
        capacity: The maximum weight the knapsack can hold.

    Returns:
        The validated capacity.

    Raises:
        TypeError: If capacity is not an integer.
        ValueError: If capacity is negative.
    """
    if not isinstance(capacity, int):
        raise TypeError(f"Capacity must be an integer, got {type(capacity).__name__}")
    if capacity < 0:
        raise ValueError(f"Capacity must be non-negative, got {capacity}")
    return capacity
