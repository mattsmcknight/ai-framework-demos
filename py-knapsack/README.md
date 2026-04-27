# py-knapsack

A production-ready Python library for solving the 0/1 knapsack problem with multiple algorithm implementations.

## Features

- **Multiple Algorithms**: Dynamic programming (standard & space-optimized), branch and bound, greedy approximation
- **Type-Safe**: Full type hints with strict mypy compliance
- **Well-Tested**: Comprehensive test suite with property-based testing (Hypothesis)
- **Production Ready**: Proper error handling, documentation, and benchmarks

## Installation

```bash
pip install py-knapsack
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from knapsack import KnapsackSolver, Item, Algorithm

# Define items (weight, value, optional name)
items = [
    Item(weight=10, value=60, name="laptop"),
    Item(weight=20, value=100, name="camera"),
    Item(weight=30, value=120, name="jewelry"),
]

# Solve using default algorithm (dynamic programming)
solver = KnapsackSolver()
result = solver.solve(items, capacity=50)

print(f"Total value: {result.total_value}")      # 220
print(f"Total weight: {result.total_weight}")    # 50
print(f"Items: {[item.name for item in result.selected_items]}")
```

## Fluent API

```python
from knapsack import KnapsackSolver, Item, Algorithm

result = (
    KnapsackSolver()
    .with_items([Item(10, 60), Item(20, 100), Item(30, 120)])
    .with_capacity(50)
    .using(Algorithm.BRANCH_AND_BOUND)
    .solve()
)
```

## Algorithms

| Algorithm | Time Complexity | Space Complexity | Optimal? | Best For |
|-----------|----------------|------------------|----------|----------|
| `DYNAMIC_PROGRAMMING` | O(n·W) | O(n·W) | ✅ | Default choice |
| `DYNAMIC_PROGRAMMING_OPTIMIZED` | O(n·W) | O(W) | ✅ | Memory-constrained |
| `BRANCH_AND_BOUND` | O(2ⁿ) worst | O(2ⁿ) worst | ✅ | Sparse solutions |
| `GREEDY` | O(n log n) | O(n) | ❌ | Very large instances |

### Algorithm Selection Guide

- **Default**: Use `DYNAMIC_PROGRAMMING` - reliable O(n·W) performance
- **Large capacity**: Use `DYNAMIC_PROGRAMMING_OPTIMIZED` to reduce memory
- **Need speed over optimality**: Use `GREEDY` for fast approximation
- **Debugging/analysis**: Use `BRANCH_AND_BOUND` - provides `items_evaluated` metric

## API Reference

### Item

```python
@dataclass(frozen=True)
class Item:
    weight: int      # Must be positive
    value: int       # Must be non-negative  
    name: str | None # Optional identifier
```

### KnapsackResult

```python
@dataclass(frozen=True)
class KnapsackResult:
    selected_items: tuple[Item, ...]
    total_value: int
    total_weight: int
    capacity: int
    algorithm: str
    is_optimal: bool
    items_evaluated: int  # For branch_and_bound
    
    @property
    def utilization(self) -> float: ...
    
    @property
    def item_count(self) -> int: ...
```

### Direct Functions

```python
from knapsack.algorithms import (
    dynamic_programming,
    dynamic_programming_optimized,
    branch_and_bound,
    greedy_approximation,
)

result = dynamic_programming(items, capacity=50)
```

## Development

### Setup

```bash
git clone https://github.com/example/py-knapsack.git
cd py-knapsack
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=knapsack --cov-report=html

# Property-based tests only
pytest tests/test_properties.py

# Benchmarks
pytest tests/test_benchmarks.py --benchmark-only
```

### Code Quality

```bash
# Type checking
mypy src/

# Linting
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## Problem Definition

The **0/1 Knapsack Problem**: Given a set of items, each with a weight and value, determine which items to include in a collection so that the total weight does not exceed a given capacity and the total value is maximized.

**Constraints**:
- Each item can be included at most once (0/1 constraint)
- Item weights must be positive integers
- Item values must be non-negative integers
- Capacity must be a non-negative integer

## License

MIT
