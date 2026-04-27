"""Unit tests for the SudokuValidator input validation module.

Tests cover:
- Structure validation: dimensions, types, value ranges
- Constraint validation: duplicate detection in rows, columns, boxes
- Solvability validation: clue count warnings
- Error collection: multiple errors gathered in a single pass
- InvalidPuzzleError exception behavior
"""

from __future__ import annotations

import pytest

from sudoku_solver.validator import InvalidPuzzleError, SudokuValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def validator() -> SudokuValidator:
    """Fresh validator instance for each test."""
    return SudokuValidator()


def _valid_empty_grid() -> list[list[int]]:
    """A structurally valid 9x9 grid of all zeros."""
    return [[0] * 9 for _ in range(9)]


# ---------------------------------------------------------------------------
# InvalidPuzzleError exception
# ---------------------------------------------------------------------------


class TestInvalidPuzzleError:
    """Test the custom exception class."""

    def test_is_value_error_subclass(self) -> None:
        """InvalidPuzzleError should be a ValueError subclass."""
        assert issubclass(InvalidPuzzleError, ValueError)

    def test_carries_message(self) -> None:
        """The exception should carry the error message."""
        err = InvalidPuzzleError("bad grid")
        assert str(err) == "bad grid"

    def test_can_be_caught_as_value_error(self) -> None:
        """InvalidPuzzleError should be catchable as ValueError."""
        with pytest.raises(ValueError):
            raise InvalidPuzzleError("test")


# ---------------------------------------------------------------------------
# Structure validation: valid grids
# ---------------------------------------------------------------------------


class TestValidStructure:
    """Structurally valid grids should produce no errors."""

    def test_valid_empty_grid(self, validator: SudokuValidator) -> None:
        """An all-zeros 9x9 grid is structurally valid."""
        grid = _valid_empty_grid()
        errors = validator.validate_structure(grid)
        assert errors == []

    def test_valid_puzzle_passes(
        self,
        validator: SudokuValidator,
        easy_puzzle: list[list[int]],
    ) -> None:
        """The easy puzzle fixture should pass structure validation."""
        errors = validator.validate_structure(easy_puzzle)
        assert errors == []

    def test_valid_solved_grid_passes(
        self,
        validator: SudokuValidator,
        solved_grid: list[list[int]],
    ) -> None:
        """A fully solved grid should pass structure validation."""
        errors = validator.validate_structure(solved_grid)
        assert errors == []


# ---------------------------------------------------------------------------
# Structure validation: wrong dimensions
# ---------------------------------------------------------------------------


class TestWrongDimensions:
    """Grids with wrong dimensions should be rejected."""

    def test_not_a_list(self, validator: SudokuValidator) -> None:
        """A non-list grid should produce an error."""
        errors = validator.validate_structure("not a list")  # type: ignore[arg-type]
        assert len(errors) == 1
        assert "must be a list" in errors[0]

    def test_wrong_row_count(self, validator: SudokuValidator) -> None:
        """A grid with fewer than 9 rows should produce an error."""
        grid = [[0] * 9 for _ in range(8)]
        errors = validator.validate_structure(grid)
        assert len(errors) == 1
        assert "9 rows" in errors[0]

    def test_too_many_rows(self, validator: SudokuValidator) -> None:
        """A grid with more than 9 rows should produce an error."""
        grid = [[0] * 9 for _ in range(10)]
        errors = validator.validate_structure(grid)
        assert len(errors) == 1
        assert "9 rows" in errors[0]

    def test_wrong_column_count(self, validator: SudokuValidator) -> None:
        """A row with fewer than 9 columns should produce an error."""
        grid = _valid_empty_grid()
        grid[3] = [0] * 8  # Row 3 has only 8 elements.
        errors = validator.validate_structure(grid)
        assert len(errors) == 1
        assert "9 columns" in errors[0]
        assert "Row 3" in errors[0]

    def test_row_not_a_list(self, validator: SudokuValidator) -> None:
        """A row that is not a list should produce an error."""
        grid = _valid_empty_grid()
        grid[5] = tuple(range(9))  # type: ignore[assignment]
        errors = validator.validate_structure(grid)
        assert len(errors) == 1
        assert "Row 5" in errors[0]
        assert "must be a list" in errors[0]


# ---------------------------------------------------------------------------
# Structure validation: bad cell values
# ---------------------------------------------------------------------------


class TestBadCellValues:
    """Cells with non-integer or out-of-range values should be rejected."""

    def test_non_integer_value(self, validator: SudokuValidator) -> None:
        """A string value in a cell should produce an error."""
        grid = _valid_empty_grid()
        grid[2][4] = "5"  # type: ignore[assignment]
        errors = validator.validate_structure(grid)
        assert len(errors) == 1
        assert "Cell (2, 4)" in errors[0]
        assert "int" in errors[0]

    def test_float_value(self, validator: SudokuValidator) -> None:
        """A float value in a cell should produce an error."""
        grid = _valid_empty_grid()
        grid[0][0] = 1.5  # type: ignore[assignment]
        errors = validator.validate_structure(grid)
        assert len(errors) == 1
        assert "Cell (0, 0)" in errors[0]

    def test_boolean_value(self, validator: SudokuValidator) -> None:
        """A boolean value in a cell should produce an error."""
        grid = _valid_empty_grid()
        grid[1][1] = True  # type: ignore[assignment]
        errors = validator.validate_structure(grid)
        assert len(errors) == 1
        assert "Cell (1, 1)" in errors[0]

    def test_negative_value(self, validator: SudokuValidator) -> None:
        """A negative integer should be out of range."""
        grid = _valid_empty_grid()
        grid[0][0] = -1
        errors = validator.validate_structure(grid)
        assert len(errors) == 1
        assert "out of range" in errors[0]

    def test_value_too_large(self, validator: SudokuValidator) -> None:
        """A value greater than 9 should be out of range."""
        grid = _valid_empty_grid()
        grid[8][8] = 10
        errors = validator.validate_structure(grid)
        assert len(errors) == 1
        assert "out of range" in errors[0]


# ---------------------------------------------------------------------------
# Constraint validation: duplicate detection
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    """Duplicate digits in rows, columns, and boxes should be detected."""

    def test_duplicate_in_row(
        self,
        validator: SudokuValidator,
        invalid_grid_duplicate_row: list[list[int]],
    ) -> None:
        """A grid with a duplicate digit in a row should be flagged."""
        errors = validator.validate_constraints(invalid_grid_duplicate_row)
        assert len(errors) >= 1
        assert any("row" in e.lower() for e in errors)

    def test_duplicate_in_column(
        self,
        validator: SudokuValidator,
        invalid_grid_duplicate_col: list[list[int]],
    ) -> None:
        """A grid with a duplicate digit in a column should be flagged."""
        errors = validator.validate_constraints(invalid_grid_duplicate_col)
        assert len(errors) >= 1
        assert any("column" in e.lower() for e in errors)

    def test_duplicate_in_box(
        self,
        validator: SudokuValidator,
        invalid_grid_duplicate_box: list[list[int]],
    ) -> None:
        """A grid with a duplicate digit in a box should be flagged."""
        errors = validator.validate_constraints(invalid_grid_duplicate_box)
        assert len(errors) >= 1
        assert any("box" in e.lower() for e in errors)

    def test_valid_puzzle_no_constraint_errors(
        self,
        validator: SudokuValidator,
        easy_puzzle: list[list[int]],
    ) -> None:
        """A valid puzzle should produce no constraint errors."""
        errors = validator.validate_constraints(easy_puzzle)
        assert errors == []


# ---------------------------------------------------------------------------
# Solvability validation
# ---------------------------------------------------------------------------


class TestSolvabilityValidation:
    """Puzzles with too few clues should produce warnings."""

    def test_below_minimum_clues(
        self, validator: SudokuValidator
    ) -> None:
        """A grid with fewer than 17 clues should produce a warning."""
        grid = _valid_empty_grid()
        # Place only 10 clues (well below 17).
        for i in range(10):
            grid[i // 9][i % 9] = (i % 9) + 1
        # Actually, just place digits in row 0 and row 1 partially.
        grid = _valid_empty_grid()
        grid[0][0] = 1
        grid[0][1] = 2
        grid[0][2] = 3
        warnings = validator.validate_solvability(grid)
        assert len(warnings) == 1
        assert "3 clues" in warnings[0]
        assert "17" in warnings[0]

    def test_empty_grid_warns(
        self, validator: SudokuValidator
    ) -> None:
        """An empty grid (0 clues) should produce a solvability warning."""
        grid = _valid_empty_grid()
        warnings = validator.validate_solvability(grid)
        assert len(warnings) == 1
        assert "0 clues" in warnings[0]

    def test_sufficient_clues_no_warning(
        self,
        validator: SudokuValidator,
        easy_puzzle: list[list[int]],
    ) -> None:
        """A puzzle with 30 clues should produce no warnings."""
        warnings = validator.validate_solvability(easy_puzzle)
        assert warnings == []

    def test_exactly_17_clues_no_warning(
        self,
        validator: SudokuValidator,
        minimal_puzzle: list[list[int]],
    ) -> None:
        """A puzzle with exactly 17 clues should produce no warning."""
        warnings = validator.validate_solvability(minimal_puzzle)
        assert warnings == []


# ---------------------------------------------------------------------------
# Error collection: validate_grid gathers all errors
# ---------------------------------------------------------------------------


class TestErrorCollection:
    """validate_grid should collect and report all errors at once."""

    def test_multiple_structure_errors(
        self, validator: SudokuValidator
    ) -> None:
        """Multiple bad cells should each produce an error message."""
        grid = _valid_empty_grid()
        grid[0][0] = -1
        grid[0][1] = 10
        grid[0][2] = "x"  # type: ignore[assignment]
        errors = validator.validate_structure(grid)
        assert len(errors) == 3

    def test_validate_grid_raises_for_constraint_violation(
        self,
        validator: SudokuValidator,
        invalid_grid_duplicate_row: list[list[int]],
    ) -> None:
        """validate_grid should raise InvalidPuzzleError for duplicates."""
        with pytest.raises(InvalidPuzzleError):
            validator.validate_grid(invalid_grid_duplicate_row)

    def test_validate_grid_raises_for_structure_error(
        self, validator: SudokuValidator
    ) -> None:
        """validate_grid should raise for non-list grid."""
        with pytest.raises(InvalidPuzzleError, match="must be a list"):
            validator.validate_grid("bad")  # type: ignore[arg-type]

    def test_validate_grid_passes_for_valid_puzzle(
        self,
        validator: SudokuValidator,
        easy_puzzle: list[list[int]],
    ) -> None:
        """validate_grid should not raise for a valid puzzle."""
        # Should not raise.
        validator.validate_grid(easy_puzzle)

    def test_validate_grid_error_message_contains_all_issues(
        self, validator: SudokuValidator
    ) -> None:
        """The error message should contain all detected problems."""
        grid = _valid_empty_grid()
        grid[0][0] = -1
        grid[0][1] = 10
        with pytest.raises(InvalidPuzzleError) as exc_info:
            validator.validate_grid(grid)
        message = str(exc_info.value)
        # Both errors should appear in the message.
        assert "out of range" in message
        # The message contains multiple lines (one per error).
        assert "\n" in message or message.count("out of range") >= 2
