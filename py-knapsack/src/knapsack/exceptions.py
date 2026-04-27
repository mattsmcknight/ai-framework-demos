"""
Custom exceptions for the knapsack solver.
"""


class KnapsackError(Exception):
    """Base exception for all knapsack-related errors."""

    pass


class InvalidItemError(KnapsackError, ValueError):
    """Raised when an item has invalid properties."""

    pass


class InvalidCapacityError(KnapsackError, ValueError):
    """Raised when capacity is invalid."""

    pass


class AlgorithmError(KnapsackError):
    """Raised when an algorithm encounters an error."""

    pass


class CapacityExceededError(KnapsackError):
    """Raised when the minimum item weight exceeds capacity."""

    def __init__(self, min_weight: int, capacity: int) -> None:
        self.min_weight = min_weight
        self.capacity = capacity
        super().__init__(
            f"No item can fit: minimum item weight ({min_weight}) "
            f"exceeds capacity ({capacity})"
        )
