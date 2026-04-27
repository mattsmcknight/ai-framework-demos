"""
Knapsack Problem Solver

A production-ready library for solving various knapsack problem variants
with multiple algorithm implementations.
"""

from knapsack.algorithms import (
    Algorithm,
    branch_and_bound,
    dynamic_programming,
    dynamic_programming_optimized,
    greedy_approximation,
)
from knapsack.models import Item, KnapsackResult
from knapsack.solver import KnapsackSolver

__version__ = "1.0.0"

__all__ = [
    # Core API
    "KnapsackSolver",
    "Item",
    "KnapsackResult",
    # Algorithms
    "Algorithm",
    "dynamic_programming",
    "dynamic_programming_optimized",
    "branch_and_bound",
    "greedy_approximation",
]
