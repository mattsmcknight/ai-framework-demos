"""Unit tests for SolutionDecoder and custom exceptions."""

from __future__ import annotations

import pytest

from sudoku_solver.solution_decoder import (
    MultipleSolutionsError,
    NoSolutionError,
    SolutionDecoder,
)


# ---------------------------------------------------------------------------
# Fixtures and test data
# ---------------------------------------------------------------------------


@pytest.fixture()
def decoder() -> SolutionDecoder:
    """Provide a fresh SolutionDecoder instance for each test."""
    return SolutionDecoder()


KNOWN_GRID: list[list[int]] = [
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

KNOWN_PUZZLE: list[list[int]] = [
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


def _grid_to_row_ids(grid: list[list[int]]) -> list[tuple[int, int, int]]:
    """Convert a solved grid into the (row, col, digit) tuples the solver would produce."""
    return [
        (r, c, grid[r][c])
        for r in range(9)
        for c in range(9)
    ]


# ---------------------------------------------------------------------------
# Custom exceptions tests
# ---------------------------------------------------------------------------


class TestCustomExceptions:
    """Verify custom exception classes."""

    def test_no_solution_error_is_exception_subclass(self) -> None:
        """NoSolutionError should be a subclass of Exception."""
        assert issubclass(NoSolutionError, Exception)

    def test_no_solution_error_message(self) -> None:
        """NoSolutionError should carry the provided message."""
        with pytest.raises(NoSolutionError, match="no valid solution"):
            raise NoSolutionError("no valid solution")

    def test_multiple_solutions_error_is_exception_subclass(self) -> None:
        """MultipleSolutionsError should be a subclass of Exception."""
        assert issubclass(MultipleSolutionsError, Exception)

    def test_multiple_solutions_error_message(self) -> None:
        """MultipleSolutionsError should carry the provided message."""
        with pytest.raises(MultipleSolutionsError, match="more than one"):
            raise MultipleSolutionsError("more than one solution")


# ---------------------------------------------------------------------------
# decode tests
# ---------------------------------------------------------------------------


class TestDecode:
    """Verify decoding of DLX solver output to Sudoku grid."""

    def test_decode_81_valid_tuples(self, decoder: SolutionDecoder) -> None:
        """decode with 81 valid (row, col, digit) tuples should produce the correct grid."""
        row_ids = _grid_to_row_ids(KNOWN_GRID)
        grid = decoder.decode(row_ids)
        assert grid == KNOWN_GRID

    def test_decode_order_independent(self, decoder: SolutionDecoder) -> None:
        """decode should work regardless of the order of row_ids."""
        row_ids = _grid_to_row_ids(KNOWN_GRID)
        # Reverse the order -- should still produce the same grid.
        reversed_ids = list(reversed(row_ids))
        grid = decoder.decode(reversed_ids)
        assert grid == KNOWN_GRID

    def test_decode_wrong_count_too_few_raises_value_error(
        self, decoder: SolutionDecoder
    ) -> None:
        """decode should raise ValueError when given fewer than 81 tuples."""
        row_ids = _grid_to_row_ids(KNOWN_GRID)[:80]
        with pytest.raises(ValueError, match="Expected 81"):
            decoder.decode(row_ids)

    def test_decode_wrong_count_too_many_raises_value_error(
        self, decoder: SolutionDecoder
    ) -> None:
        """decode should raise ValueError when given more than 81 tuples."""
        row_ids = _grid_to_row_ids(KNOWN_GRID) + [(0, 0, 1)]
        with pytest.raises(ValueError, match="Expected 81"):
            decoder.decode(row_ids)

    def test_decode_empty_list_raises_value_error(
        self, decoder: SolutionDecoder
    ) -> None:
        """decode should raise ValueError when given an empty list."""
        with pytest.raises(ValueError, match="Expected 81"):
            decoder.decode([])


# ---------------------------------------------------------------------------
# format_grid tests
# ---------------------------------------------------------------------------


class TestFormatGrid:
    """Verify the human-readable grid formatting."""

    def test_format_grid_has_correct_line_count(
        self, decoder: SolutionDecoder
    ) -> None:
        """format_grid should produce 13 lines: 4 separators + 9 data rows."""
        output = decoder.format_grid(KNOWN_GRID)
        lines = output.split("\n")
        assert len(lines) == 13

    def test_format_grid_separators(
        self, decoder: SolutionDecoder
    ) -> None:
        """Separator lines should appear at rows 0, 3, 6, 9 (0-indexed)."""
        output = decoder.format_grid(KNOWN_GRID)
        lines = output.split("\n")

        separator_indices = [0, 4, 8, 12]
        for idx in separator_indices:
            assert lines[idx].startswith("+"), (
                f"Line {idx} should be a separator, got: {lines[idx]!r}"
            )
            assert "-" in lines[idx]

    def test_format_grid_data_rows_have_pipes(
        self, decoder: SolutionDecoder
    ) -> None:
        """Data rows should have pipe characters as box separators."""
        output = decoder.format_grid(KNOWN_GRID)
        lines = output.split("\n")

        data_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11]
        for idx in data_indices:
            assert lines[idx].startswith("|"), (
                f"Line {idx} should start with '|', got: {lines[idx]!r}"
            )
            assert lines[idx].endswith("|"), (
                f"Line {idx} should end with '|', got: {lines[idx]!r}"
            )
            # Each data row should have exactly 4 pipe characters.
            assert lines[idx].count("|") == 4

    def test_format_grid_contains_all_digits(
        self, decoder: SolutionDecoder
    ) -> None:
        """The formatted grid should contain all digits from the grid data."""
        output = decoder.format_grid(KNOWN_GRID)
        for row in KNOWN_GRID:
            for digit in row:
                assert str(digit) in output


# ---------------------------------------------------------------------------
# format_grid_compact tests
# ---------------------------------------------------------------------------


class TestFormatGridCompact:
    """Verify the compact 81-character string formatting."""

    def test_compact_output_length(self, decoder: SolutionDecoder) -> None:
        """format_grid_compact should produce exactly 81 characters."""
        output = decoder.format_grid_compact(KNOWN_GRID)
        assert len(output) == 81

    def test_compact_output_content(self, decoder: SolutionDecoder) -> None:
        """The compact string should match row-by-row concatenation of digits."""
        output = decoder.format_grid_compact(KNOWN_GRID)
        expected = "".join(
            str(KNOWN_GRID[r][c]) for r in range(9) for c in range(9)
        )
        assert output == expected

    def test_compact_output_all_digits(
        self, decoder: SolutionDecoder
    ) -> None:
        """The compact string should contain only digit characters."""
        output = decoder.format_grid_compact(KNOWN_GRID)
        assert output.isdigit()


# ---------------------------------------------------------------------------
# Round-trip test: full pipeline
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end: encode Sudoku as exact cover, solve, decode, verify."""

    def test_round_trip_solve_and_decode(self) -> None:
        """Build matrix from puzzle, solve, decode, verify against known answer."""
        from sudoku_solver.constraint_mapper import SudokuConstraintMapper
        from sudoku_solver.solver import DLXSolver

        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(KNOWN_PUZZLE)

        solver = DLXSolver(matrix)
        solution_row_ids = solver.solve_one()
        assert solution_row_ids is not None

        decoder = SolutionDecoder()
        grid = decoder.decode(solution_row_ids)

        assert grid == KNOWN_GRID

    def test_round_trip_compact_format(self) -> None:
        """The compact format of a solved puzzle should be 81 valid digits."""
        from sudoku_solver.constraint_mapper import SudokuConstraintMapper
        from sudoku_solver.solver import DLXSolver

        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(KNOWN_PUZZLE)

        solver = DLXSolver(matrix)
        solution_row_ids = solver.solve_one()
        assert solution_row_ids is not None

        decoder = SolutionDecoder()
        grid = decoder.decode(solution_row_ids)
        compact = decoder.format_grid_compact(grid)

        assert len(compact) == 81
        assert compact.isdigit()
        assert "0" not in compact  # A solved grid should have no zeros.
