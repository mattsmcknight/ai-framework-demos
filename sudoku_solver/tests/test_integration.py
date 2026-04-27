"""Integration tests: full pipeline from puzzle to verified solution.

These tests exercise the complete Sudoku solver pipeline end-to-end:
parse grid -> validate -> build constraint matrix -> solve -> decode -> verify.

Each test solves a known puzzle, confirms the solution matches the expected
answer, and independently verifies that the solution satisfies all Sudoku
constraints (no duplicate digits in any row, column, or 3x3 box).
"""

from __future__ import annotations

import pytest

from sudoku_solver.constraint_mapper import SudokuConstraintMapper
from sudoku_solver.solution_decoder import SolutionDecoder
from sudoku_solver.solver import DLXSolver
from sudoku_solver.validator import SudokuValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solve_puzzle(grid: list[list[int]]) -> list[list[int]] | None:
    """Run the full pipeline: build matrix -> solve -> decode.

    Args:
        grid: A 9x9 Sudoku grid (0 = empty).

    Returns:
        A 9x9 solved grid, or None if no solution exists.
    """
    mapper = SudokuConstraintMapper()
    matrix = mapper.build_matrix(grid)
    solver = DLXSolver(matrix)
    row_ids = solver.solve_one()
    if row_ids is None:
        return None
    decoder = SolutionDecoder()
    return decoder.decode(row_ids)


def _verify_solution(grid: list[list[int]]) -> None:
    """Assert that the grid is a valid, complete Sudoku solution.

    Checks:
    - All cells contain digits 1-9 (no zeros).
    - No duplicate digits in any row.
    - No duplicate digits in any column.
    - No duplicate digits in any 3x3 box.
    """
    full_set = set(range(1, 10))

    # Every cell must be 1-9.
    for r in range(9):
        for c in range(9):
            assert 1 <= grid[r][c] <= 9, (
                f"Cell ({r},{c}) has invalid value {grid[r][c]}"
            )

    # No duplicates in rows.
    for r in range(9):
        row_vals = set(grid[r])
        assert row_vals == full_set, (
            f"Row {r} is incomplete or has duplicates: {grid[r]}"
        )

    # No duplicates in columns.
    for c in range(9):
        col_vals = {grid[r][c] for r in range(9)}
        assert col_vals == full_set, (
            f"Column {c} is incomplete or has duplicates"
        )

    # No duplicates in 3x3 boxes.
    for box_row in range(3):
        for box_col in range(3):
            box_vals = {
                grid[box_row * 3 + dr][box_col * 3 + dc]
                for dr in range(3)
                for dc in range(3)
            }
            assert box_vals == full_set, (
                f"Box ({box_row},{box_col}) is incomplete or has duplicates"
            )


def _verify_givens_preserved(
    puzzle: list[list[int]], solution: list[list[int]]
) -> None:
    """Assert that all pre-filled cells in the puzzle appear unchanged."""
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != 0:
                assert solution[r][c] == puzzle[r][c], (
                    f"Given cell ({r},{c})={puzzle[r][c]} changed to "
                    f"{solution[r][c]} in solution"
                )


# ---------------------------------------------------------------------------
# Tests: solve known puzzles
# ---------------------------------------------------------------------------


class TestSolveKnownPuzzles:
    """Solve puzzles of varying difficulty and verify against known answers."""

    def test_solve_easy_puzzle(
        self, easy_puzzle: list[list[int]], easy_solution: list[list[int]]
    ) -> None:
        """Easy puzzle (30 clues) should produce the known solution."""
        result = _solve_puzzle(easy_puzzle)
        assert result is not None
        assert result == easy_solution

    def test_solve_medium_puzzle(
        self, medium_puzzle: list[list[int]], medium_solution: list[list[int]]
    ) -> None:
        """Medium puzzle (25 clues) should produce the known solution."""
        result = _solve_puzzle(medium_puzzle)
        assert result is not None
        assert result == medium_solution

    def test_solve_hard_puzzle(
        self, hard_puzzle: list[list[int]], hard_solution: list[list[int]]
    ) -> None:
        """Hard puzzle (21 clues) should produce the known solution."""
        result = _solve_puzzle(hard_puzzle)
        assert result is not None
        assert result == hard_solution

    def test_solve_minimal_puzzle(
        self,
        minimal_puzzle: list[list[int]],
        minimal_solution: list[list[int]],
    ) -> None:
        """Minimal 17-clue puzzle should produce the known solution."""
        result = _solve_puzzle(minimal_puzzle)
        assert result is not None
        assert result == minimal_solution


# ---------------------------------------------------------------------------
# Tests: verify solution correctness independently
# ---------------------------------------------------------------------------


class TestVerifySolutions:
    """Verify each solution satisfies all Sudoku constraints independently."""

    def test_easy_solution_valid(self, easy_puzzle: list[list[int]]) -> None:
        """Easy puzzle solution has no duplicates in rows/cols/boxes."""
        result = _solve_puzzle(easy_puzzle)
        assert result is not None
        _verify_solution(result)

    def test_medium_solution_valid(
        self, medium_puzzle: list[list[int]]
    ) -> None:
        """Medium puzzle solution has no duplicates in rows/cols/boxes."""
        result = _solve_puzzle(medium_puzzle)
        assert result is not None
        _verify_solution(result)

    def test_hard_solution_valid(self, hard_puzzle: list[list[int]]) -> None:
        """Hard puzzle solution has no duplicates in rows/cols/boxes."""
        result = _solve_puzzle(hard_puzzle)
        assert result is not None
        _verify_solution(result)

    def test_minimal_solution_valid(
        self, minimal_puzzle: list[list[int]]
    ) -> None:
        """Minimal puzzle solution has no duplicates in rows/cols/boxes."""
        result = _solve_puzzle(minimal_puzzle)
        assert result is not None
        _verify_solution(result)


# ---------------------------------------------------------------------------
# Tests: verify givens are preserved in solutions
# ---------------------------------------------------------------------------


class TestGivensPreserved:
    """Verify that pre-filled cells are not changed by the solver."""

    def test_easy_givens_preserved(
        self, easy_puzzle: list[list[int]]
    ) -> None:
        """All given cells in the easy puzzle appear in the solution."""
        result = _solve_puzzle(easy_puzzle)
        assert result is not None
        _verify_givens_preserved(easy_puzzle, result)

    def test_medium_givens_preserved(
        self, medium_puzzle: list[list[int]]
    ) -> None:
        """All given cells in the medium puzzle appear in the solution."""
        result = _solve_puzzle(medium_puzzle)
        assert result is not None
        _verify_givens_preserved(medium_puzzle, result)

    def test_hard_givens_preserved(
        self, hard_puzzle: list[list[int]]
    ) -> None:
        """All given cells in the hard puzzle appear in the solution."""
        result = _solve_puzzle(hard_puzzle)
        assert result is not None
        _verify_givens_preserved(hard_puzzle, result)

    def test_minimal_givens_preserved(
        self, minimal_puzzle: list[list[int]]
    ) -> None:
        """All given cells in the minimal puzzle appear in the solution."""
        result = _solve_puzzle(minimal_puzzle)
        assert result is not None
        _verify_givens_preserved(minimal_puzzle, result)


# ---------------------------------------------------------------------------
# Tests: full pipeline with formatting
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Test the complete pipeline including validation and formatting."""

    def test_full_pipeline_parse_validate_solve_format(
        self, easy_puzzle: list[list[int]]
    ) -> None:
        """Exercise every stage: validate -> build matrix -> solve -> decode -> format."""
        # Stage 1: validate
        validator = SudokuValidator()
        errors = validator.validate_structure(easy_puzzle)
        assert errors == []
        errors = validator.validate_constraints(easy_puzzle)
        assert errors == []

        # Stage 2: build matrix
        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(easy_puzzle)
        assert not matrix.is_empty()

        # Stage 3: solve
        solver = DLXSolver(matrix)
        row_ids = solver.solve_one()
        assert row_ids is not None
        assert len(row_ids) == 81

        # Stage 4: decode
        decoder = SolutionDecoder()
        grid = decoder.decode(row_ids)
        assert len(grid) == 9
        assert all(len(row) == 9 for row in grid)

        # Stage 5: format
        formatted = decoder.format_grid(grid)
        assert "+" in formatted  # has separators
        assert "|" in formatted  # has box dividers

        compact = decoder.format_grid_compact(grid)
        assert len(compact) == 81
        assert compact.isdigit()
        assert "0" not in compact

    def test_full_pipeline_solution_uniqueness(
        self, medium_puzzle: list[list[int]]
    ) -> None:
        """The medium puzzle should have exactly one solution."""
        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(medium_puzzle)
        solver = DLXSolver(matrix)
        count = solver.count_solutions(limit=2)
        assert count == 1

    def test_full_pipeline_matrix_restored_after_solve(
        self, easy_puzzle: list[list[int]]
    ) -> None:
        """The DLX matrix should be fully restored after solving."""
        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(easy_puzzle)

        # Count columns before solve.
        col_count_before = sum(
            1 for _ in _iter_headers(matrix)
        )

        solver = DLXSolver(matrix)
        solver.solve_one()

        # Count columns after solve.
        col_count_after = sum(
            1 for _ in _iter_headers(matrix)
        )
        assert col_count_before == col_count_after


def _iter_headers(matrix: object) -> list[str]:
    """Walk the header list and return column names."""
    names: list[str] = []
    current = matrix.root.right  # type: ignore[attr-defined]
    while current is not matrix.root:  # type: ignore[attr-defined]
        names.append(current.name)
        current = current.right
    return names
