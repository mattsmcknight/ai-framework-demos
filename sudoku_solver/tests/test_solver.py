"""Unit tests for the DLXSolver (Algorithm X on Dancing Links)."""

from __future__ import annotations

from typing import Any

import pytest

from sudoku_solver.dlx_matrix import DLXMatrix
from sudoku_solver.dlx_node import ColumnHeader
from sudoku_solver.solver import DLXSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_column_row_ids(column: ColumnHeader) -> list[Any]:
    """Walk down a column and return the row_ids of all data nodes."""
    row_ids: list[Any] = []
    node = column.down
    while node is not column:
        row_ids.append(node.row_id)
        node = node.down
    return row_ids


def _collect_headers(matrix: DLXMatrix) -> list[str]:
    """Traverse the header list left-to-right and return column names."""
    names: list[str] = []
    current = matrix.root.right
    while current is not matrix.root:
        assert isinstance(current, ColumnHeader)
        names.append(current.name)
        current = current.right
    return names


def _snapshot_matrix(matrix: DLXMatrix) -> dict[str, Any]:
    """Capture full matrix state for comparison."""
    header_names = _collect_headers(matrix)
    columns: dict[str, dict[str, Any]] = {}
    for name in header_names:
        col = matrix.columns[name]
        columns[name] = {
            "size": col.size,
            "row_ids": _collect_column_row_ids(col),
        }
    return {"header_names": header_names, "columns": columns}


def _knuth_matrix() -> DLXMatrix:
    """Build Knuth's exact cover example matrix.

    Universe: {1, 2, 3, 4, 5, 6, 7}
    Row A: {1, 4, 7}
    Row B: {1, 4}
    Row C: {4, 5, 7}
    Row D: {3, 5, 6}
    Row E: {2, 3, 6, 7}
    Row F: {2, 7}

    Unique exact cover: {B, D, F}.
    """
    cols = ["1", "2", "3", "4", "5", "6", "7"]
    matrix = DLXMatrix(cols)
    matrix.add_row("A", ["1", "4", "7"])
    matrix.add_row("B", ["1", "4"])
    matrix.add_row("C", ["4", "5", "7"])
    matrix.add_row("D", ["3", "5", "6"])
    matrix.add_row("E", ["2", "3", "6", "7"])
    matrix.add_row("F", ["2", "7"])
    return matrix


# ---------------------------------------------------------------------------
# A known Sudoku puzzle for end-to-end testing
# ---------------------------------------------------------------------------

EASY_PUZZLE: list[list[int]] = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

EASY_SOLUTION: list[list[int]] = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


# ---------------------------------------------------------------------------
# Generic exact cover tests
# ---------------------------------------------------------------------------


class TestGenericExactCover:
    """Test the solver on generic exact cover instances (not Sudoku-specific)."""

    def test_knuths_example_solution_is_bdf(self) -> None:
        """Knuth's paper example has the unique solution {B, D, F}."""
        matrix = _knuth_matrix()
        solver = DLXSolver(matrix, max_solutions=0)
        solutions = list(solver.solve())

        assert len(solutions) == 1
        assert sorted(solutions[0]) == ["B", "D", "F"]

    def test_no_solution_contradictory_constraints(self) -> None:
        """A matrix with contradictory constraints should yield no solutions."""
        # Two columns, two rows that both cover column A.
        # No row covers column B alone, so no exact cover is possible.
        matrix = DLXMatrix(["A", "B"])
        matrix.add_row("r1", ["A"])
        matrix.add_row("r2", ["A"])
        # Column B is never covered -> no solution.
        solver = DLXSolver(matrix, max_solutions=0)
        solutions = list(solver.solve())

        assert len(solutions) == 0

    def test_multiple_solutions_count(self) -> None:
        """Verify the solver finds the correct number of solutions."""
        # Build a matrix with exactly 2 solutions.
        #   Columns: A, B, C
        #   Row 1: A, B   | Row 2: C       -> Solution 1: {1, 2}
        #   Row 3: A      | Row 4: B, C    -> Solution 2: {3, 4}
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(1, ["A", "B"])
        matrix.add_row(2, ["C"])
        matrix.add_row(3, ["A"])
        matrix.add_row(4, ["B", "C"])

        solver = DLXSolver(matrix, max_solutions=0)
        solutions = list(solver.solve())

        assert len(solutions) == 2
        sorted_solutions = [sorted(s) for s in solutions]
        assert [1, 2] in sorted_solutions
        assert [3, 4] in sorted_solutions

    def test_empty_matrix_yields_empty_solution(self) -> None:
        """A matrix with zero columns should yield one solution: the empty set."""
        matrix = DLXMatrix([])
        solver = DLXSolver(matrix, max_solutions=0)
        solutions = list(solver.solve())

        assert len(solutions) == 1
        assert solutions[0] == []

    def test_all_rows_conflict_no_solution(self) -> None:
        """When every row covers the same single column, only one can be chosen.

        If there are other uncoverable columns, no exact cover exists.
        """
        matrix = DLXMatrix(["A", "B"])
        matrix.add_row("r1", ["A"])
        matrix.add_row("r2", ["A"])
        matrix.add_row("r3", ["A"])
        # All rows cover A but none cover B.
        solver = DLXSolver(matrix, max_solutions=0)
        solutions = list(solver.solve())

        assert len(solutions) == 0

    def test_single_row_single_column_exact_cover(self) -> None:
        """Minimal exact cover: one column, one row that covers it."""
        matrix = DLXMatrix(["A"])
        matrix.add_row("only", ["A"])

        solver = DLXSolver(matrix, max_solutions=0)
        solutions = list(solver.solve())

        assert len(solutions) == 1
        assert solutions[0] == ["only"]

    def test_identity_matrix_has_unique_solution(self) -> None:
        """An identity-like matrix (each row covers one unique column) has one solution."""
        cols = ["A", "B", "C", "D"]
        matrix = DLXMatrix(cols)
        matrix.add_row("rA", ["A"])
        matrix.add_row("rB", ["B"])
        matrix.add_row("rC", ["C"])
        matrix.add_row("rD", ["D"])

        solver = DLXSolver(matrix, max_solutions=0)
        solutions = list(solver.solve())

        assert len(solutions) == 1
        assert sorted(solutions[0]) == ["rA", "rB", "rC", "rD"]


# ---------------------------------------------------------------------------
# solve_one tests
# ---------------------------------------------------------------------------


class TestSolveOne:
    """Test the solve_one convenience method."""

    def test_returns_solution_for_solvable(self) -> None:
        """solve_one should return a list of row_ids for a solvable instance."""
        matrix = _knuth_matrix()
        solver = DLXSolver(matrix)
        result = solver.solve_one()

        assert result is not None
        assert sorted(result) == ["B", "D", "F"]

    def test_returns_none_for_unsolvable(self) -> None:
        """solve_one should return None when no exact cover exists."""
        matrix = DLXMatrix(["A", "B"])
        matrix.add_row("r1", ["A"])
        # Column B cannot be covered.
        solver = DLXSolver(matrix)
        result = solver.solve_one()

        assert result is None

    def test_restores_max_solutions(self) -> None:
        """solve_one should restore the original max_solutions value after returning."""
        matrix = _knuth_matrix()
        solver = DLXSolver(matrix, max_solutions=42)
        solver.solve_one()

        assert solver.max_solutions == 42


# ---------------------------------------------------------------------------
# count_solutions tests
# ---------------------------------------------------------------------------


class TestCountSolutions:
    """Test the count_solutions method."""

    def test_correct_count_for_known_instance(self) -> None:
        """count_solutions should return the exact number of solutions."""
        # Knuth's example has exactly 1 solution.
        matrix = _knuth_matrix()
        solver = DLXSolver(matrix)

        assert solver.count_solutions(limit=0) == 1

    def test_count_with_multiple_solutions(self) -> None:
        """count_solutions should count all solutions when limit is 0."""
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(1, ["A", "B"])
        matrix.add_row(2, ["C"])
        matrix.add_row(3, ["A"])
        matrix.add_row(4, ["B", "C"])

        solver = DLXSolver(matrix)
        assert solver.count_solutions(limit=0) == 2

    def test_limit_stops_counting_early(self) -> None:
        """count_solutions with limit=1 should stop after finding one solution."""
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(1, ["A", "B"])
        matrix.add_row(2, ["C"])
        matrix.add_row(3, ["A"])
        matrix.add_row(4, ["B", "C"])

        solver = DLXSolver(matrix)
        assert solver.count_solutions(limit=1) == 1

    def test_restores_max_solutions(self) -> None:
        """count_solutions should restore the original max_solutions value."""
        matrix = _knuth_matrix()
        solver = DLXSolver(matrix, max_solutions=7)
        solver.count_solutions(limit=0)

        assert solver.max_solutions == 7


# ---------------------------------------------------------------------------
# Matrix preservation tests
# ---------------------------------------------------------------------------


class TestMatrixPreservation:
    """Verify the matrix is restored to its original state after solving."""

    def test_matrix_restored_after_solve_exhausted(self) -> None:
        """After fully exhausting the solve generator, the matrix should be identical."""
        matrix = _knuth_matrix()
        before = _snapshot_matrix(matrix)

        solver = DLXSolver(matrix, max_solutions=0)
        list(solver.solve())  # exhaust the generator

        after = _snapshot_matrix(matrix)
        assert before == after

    def test_matrix_restored_after_solve_one(self) -> None:
        """After solve_one, the matrix should be identical to its pre-solve state."""
        matrix = _knuth_matrix()
        before = _snapshot_matrix(matrix)

        solver = DLXSolver(matrix)
        solver.solve_one()

        after = _snapshot_matrix(matrix)
        assert before == after

    def test_matrix_restored_after_no_solution(self) -> None:
        """Even when no solution exists, the matrix should be restored."""
        matrix = DLXMatrix(["A", "B"])
        matrix.add_row("r1", ["A"])
        before = _snapshot_matrix(matrix)

        solver = DLXSolver(matrix)
        solver.solve_one()

        after = _snapshot_matrix(matrix)
        assert before == after

    def test_matrix_restored_after_partial_iteration(self) -> None:
        """If the generator is abandoned mid-search, the matrix should still restore."""
        # Matrix with 2 solutions -- we take only the first.
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(1, ["A", "B"])
        matrix.add_row(2, ["C"])
        matrix.add_row(3, ["A"])
        matrix.add_row(4, ["B", "C"])

        before = _snapshot_matrix(matrix)

        solver = DLXSolver(matrix, max_solutions=0)
        gen = solver.solve()
        next(gen)  # take one solution
        gen.close()  # abandon the rest

        after = _snapshot_matrix(matrix)
        assert before == after

    def test_matrix_restored_after_count_solutions(self) -> None:
        """count_solutions should also leave the matrix in its original state."""
        matrix = _knuth_matrix()
        before = _snapshot_matrix(matrix)

        solver = DLXSolver(matrix)
        solver.count_solutions(limit=0)

        after = _snapshot_matrix(matrix)
        assert before == after


# ---------------------------------------------------------------------------
# End-to-end mini-test: Sudoku through DLXSolver
# ---------------------------------------------------------------------------


class TestEndToEndSudoku:
    """Build a Sudoku exact cover matrix, solve it, decode, and verify."""

    def test_solve_easy_sudoku_puzzle(self) -> None:
        """Full pipeline: puzzle -> constraint_mapper -> solver -> decoder -> verify."""
        from sudoku_solver.constraint_mapper import SudokuConstraintMapper
        from sudoku_solver.solution_decoder import SolutionDecoder

        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(EASY_PUZZLE)

        solver = DLXSolver(matrix)
        solution_row_ids = solver.solve_one()

        assert solution_row_ids is not None

        decoder = SolutionDecoder()
        grid = decoder.decode(solution_row_ids)

        assert grid == EASY_SOLUTION

    def test_solved_puzzle_preserves_givens(self) -> None:
        """All pre-filled cells in the original puzzle must appear in the solution."""
        from sudoku_solver.constraint_mapper import SudokuConstraintMapper
        from sudoku_solver.solution_decoder import SolutionDecoder

        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(EASY_PUZZLE)

        solver = DLXSolver(matrix)
        solution_row_ids = solver.solve_one()
        assert solution_row_ids is not None

        decoder = SolutionDecoder()
        grid = decoder.decode(solution_row_ids)

        for r in range(9):
            for c in range(9):
                if EASY_PUZZLE[r][c] != 0:
                    assert grid[r][c] == EASY_PUZZLE[r][c], (
                        f"Given cell ({r},{c})={EASY_PUZZLE[r][c]} "
                        f"but solution has {grid[r][c]}"
                    )

    def test_easy_puzzle_has_unique_solution(self) -> None:
        """The easy puzzle should have exactly one solution."""
        from sudoku_solver.constraint_mapper import SudokuConstraintMapper

        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(EASY_PUZZLE)

        solver = DLXSolver(matrix)
        count = solver.count_solutions(limit=2)

        assert count == 1
