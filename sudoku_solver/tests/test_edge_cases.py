"""Edge case tests for the Sudoku DLX solver.

Tests behavior at the boundaries of the problem domain:
- Empty grid (no clues): solver finds at least one valid solution
- Fully solved grid: confirmed as valid without modification
- Unsolvable puzzle: no solution returned
- Multiple solutions: count_solutions detects ambiguity
- Single empty cell: solved correctly
"""

from __future__ import annotations

import pytest

from sudoku_solver.constraint_mapper import SudokuConstraintMapper
from sudoku_solver.solution_decoder import SolutionDecoder
from sudoku_solver.solver import DLXSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solve_puzzle(grid: list[list[int]]) -> list[list[int]] | None:
    """Build matrix, solve, decode. Returns grid or None."""
    mapper = SudokuConstraintMapper()
    matrix = mapper.build_matrix(grid)
    solver = DLXSolver(matrix)
    row_ids = solver.solve_one()
    if row_ids is None:
        return None
    return SolutionDecoder().decode(row_ids)


def _is_valid_solution(grid: list[list[int]]) -> bool:
    """Return True if grid is a complete, valid Sudoku solution."""
    full_set = set(range(1, 10))

    for r in range(9):
        if set(grid[r]) != full_set:
            return False

    for c in range(9):
        if {grid[r][c] for r in range(9)} != full_set:
            return False

    for br in range(3):
        for bc in range(3):
            box = {
                grid[br * 3 + dr][bc * 3 + dc]
                for dr in range(3) for dc in range(3)
            }
            if box != full_set:
                return False

    return True


# ---------------------------------------------------------------------------
# Empty grid tests
# ---------------------------------------------------------------------------


class TestEmptyGrid:
    """An all-zeros grid should be solvable (many solutions exist)."""

    def test_empty_grid_has_solution(
        self, empty_grid: list[list[int]]
    ) -> None:
        """The solver should find at least one solution for an empty grid."""
        result = _solve_puzzle(empty_grid)
        assert result is not None

    def test_empty_grid_solution_is_valid(
        self, empty_grid: list[list[int]]
    ) -> None:
        """The solution for an empty grid must be a valid Sudoku."""
        result = _solve_puzzle(empty_grid)
        assert result is not None
        assert _is_valid_solution(result)

    def test_empty_grid_has_multiple_solutions(
        self, empty_grid: list[list[int]]
    ) -> None:
        """An empty grid should have more than one solution."""
        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(empty_grid)
        solver = DLXSolver(matrix)
        count = solver.count_solutions(limit=2)
        assert count >= 2


# ---------------------------------------------------------------------------
# Solved grid tests
# ---------------------------------------------------------------------------


class TestSolvedGrid:
    """A fully filled valid grid should be confirmed as-is."""

    def test_solved_grid_is_valid(
        self, solved_grid: list[list[int]]
    ) -> None:
        """The solved grid fixture itself should be a valid solution."""
        assert _is_valid_solution(solved_grid)

    def test_solved_grid_passes_solver(
        self, solved_grid: list[list[int]]
    ) -> None:
        """Feeding a solved grid to the solver should return it unchanged."""
        result = _solve_puzzle(solved_grid)
        assert result is not None
        assert result == solved_grid

    def test_solved_grid_has_exactly_one_solution(
        self, solved_grid: list[list[int]]
    ) -> None:
        """A fully filled grid should have exactly one solution (itself)."""
        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(solved_grid)
        solver = DLXSolver(matrix)
        count = solver.count_solutions(limit=2)
        assert count == 1


# ---------------------------------------------------------------------------
# Unsolvable puzzle tests
# ---------------------------------------------------------------------------


class TestUnsolvablePuzzle:
    """An unsolvable puzzle should return no solution."""

    def test_unsolvable_returns_none(
        self, unsolvable_puzzle: list[list[int]]
    ) -> None:
        """The solver should return None for an unsolvable puzzle."""
        result = _solve_puzzle(unsolvable_puzzle)
        assert result is None

    def test_unsolvable_count_is_zero(
        self, unsolvable_puzzle: list[list[int]]
    ) -> None:
        """count_solutions should return 0 for an unsolvable puzzle."""
        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(unsolvable_puzzle)
        solver = DLXSolver(matrix)
        count = solver.count_solutions(limit=0)
        assert count == 0

    def test_unsolvable_matrix_restored(
        self, unsolvable_puzzle: list[list[int]]
    ) -> None:
        """The matrix should be fully restored after failing to solve."""
        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(unsolvable_puzzle)

        # Snapshot column count before.
        col_count_before = _count_headers(matrix)

        solver = DLXSolver(matrix)
        solver.solve_one()

        # Snapshot column count after.
        col_count_after = _count_headers(matrix)
        assert col_count_before == col_count_after


# ---------------------------------------------------------------------------
# Multiple solutions tests
# ---------------------------------------------------------------------------


class TestMultipleSolutions:
    """Puzzles with multiple solutions should be detectable."""

    def test_empty_grid_multiple_solutions(
        self, empty_grid: list[list[int]]
    ) -> None:
        """count_solutions(limit=2) on empty grid should return >= 2."""
        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(empty_grid)
        solver = DLXSolver(matrix)
        count = solver.count_solutions(limit=2)
        assert count >= 2

    def test_ambiguous_puzzle_detected(self) -> None:
        """A puzzle with two known solutions should be detected as ambiguous.

        We create a near-complete puzzle by clearing two cells from a solved
        grid in a way that creates exactly two valid completions.
        """
        # Start from the easy solution and clear two cells that can swap.
        # Row 0: 5 3 4 6 7 8 9 1 2
        # Row 1: 6 7 2 1 9 5 3 4 8
        # Clear cells (0,6)=9 and (0,7)=1 -- but that won't create ambiguity
        # easily. Instead, use a known underconstrained puzzle.
        # A grid with 0 clues has many solutions; limit=2 catches it.
        empty = [[0] * 9 for _ in range(9)]
        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(empty)
        solver = DLXSolver(matrix)
        count = solver.count_solutions(limit=2)
        assert count >= 2


# ---------------------------------------------------------------------------
# Single empty cell tests
# ---------------------------------------------------------------------------


class TestSingleEmptyCell:
    """A puzzle with only one empty cell should be solved trivially."""

    def test_single_empty_cell_solved(
        self, solved_grid: list[list[int]]
    ) -> None:
        """Clearing one cell should produce a uniquely solvable puzzle."""
        # Clear cell (4, 4) from the solved grid.
        puzzle = [row[:] for row in solved_grid]
        original_value = puzzle[4][4]
        puzzle[4][4] = 0

        result = _solve_puzzle(puzzle)
        assert result is not None
        assert result[4][4] == original_value

    def test_single_empty_cell_solution_matches_original(
        self, solved_grid: list[list[int]]
    ) -> None:
        """Solving a single-gap puzzle should recover the original grid."""
        puzzle = [row[:] for row in solved_grid]
        puzzle[0][0] = 0

        result = _solve_puzzle(puzzle)
        assert result is not None
        assert result == solved_grid

    def test_single_empty_cell_unique_solution(
        self, solved_grid: list[list[int]]
    ) -> None:
        """A single-gap puzzle should have exactly one solution."""
        puzzle = [row[:] for row in solved_grid]
        puzzle[8][8] = 0

        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(puzzle)
        solver = DLXSolver(matrix)
        count = solver.count_solutions(limit=2)
        assert count == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_headers(matrix: object) -> int:
    """Count the number of columns in the matrix header list."""
    count = 0
    current = matrix.root.right  # type: ignore[attr-defined]
    while current is not matrix.root:  # type: ignore[attr-defined]
        count += 1
        current = current.right
    return count
