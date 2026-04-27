"""Input validation and error handling for Sudoku puzzles.

This module validates Sudoku grids at the system boundary before they reach
the constraint mapper or solver.  Validation is split into three stages:

1. **Structure**: Is the grid the right shape with valid cell values?
2. **Constraints**: Does the grid obey Sudoku rules (no duplicate digits)?
3. **Solvability**: Are there enough clues for a likely-solvable puzzle?

The validator collects ALL errors rather than stopping at the first one,
so the caller receives a complete diagnosis in a single pass.

Custom exception :class:`InvalidPuzzleError` is defined here for use by
CLI and orchestration code.

Typical usage::

    from sudoku_solver.validator import SudokuValidator, InvalidPuzzleError

    validator = SudokuValidator()
    try:
        validator.validate_grid(grid)
    except InvalidPuzzleError as e:
        print(e)
"""

from __future__ import annotations


class InvalidPuzzleError(ValueError):
    """Raised when a Sudoku puzzle grid fails validation.

    The error message contains all detected problems, separated by
    newlines, so the caller can display the full diagnosis at once.
    """


class SudokuValidator:
    """Validate a 9x9 Sudoku grid for structural and logical correctness.

    This class is stateless: all configuration lives in class-level
    constants, and each method is a pure function of its arguments.

    Class Attributes:
        SIZE: The dimension of the Sudoku grid (9).
        BOX_SIZE: The dimension of each 3x3 box (3).
        MIN_CLUES: Minimum clues for a uniquely solvable puzzle (17).
    """

    SIZE: int = 9
    BOX_SIZE: int = 3
    MIN_CLUES: int = 17

    def validate_grid(self, grid: list[list[int]]) -> None:
        """Run all validation checks and raise if any errors are found.

        Runs structure validation first.  If the grid is structurally
        valid, constraint validation is also run.  Solvability warnings
        are appended but treated as errors to force explicit handling.

        Args:
            grid: The puzzle grid to validate.

        Raises:
            InvalidPuzzleError: If one or more validation errors are found.
                The message contains all errors separated by newlines.
        """
        errors = self.validate_structure(grid)
        if not errors:
            errors.extend(self.validate_constraints(grid))
        warnings = self.validate_solvability(grid) if not errors else []
        all_issues = errors + warnings
        if all_issues:
            raise InvalidPuzzleError("\n".join(all_issues))

    def validate_structure(self, grid: list[list[int]]) -> list[str]:
        """Check that the grid has the correct shape and cell values.

        Validates:
        - The grid is a list.
        - The grid contains exactly 9 rows.
        - Each row is a list of exactly 9 elements.
        - Every cell value is an integer in the range 0-9.

        Args:
            grid: The puzzle grid to validate.

        Returns:
            A list of error message strings.  Empty if the grid is
            structurally valid.
        """
        errors: list[str] = []

        if not isinstance(grid, list):
            errors.append(
                f"Grid must be a list, got {type(grid).__name__}"
            )
            return errors

        if len(grid) != self.SIZE:
            errors.append(
                f"Grid must have {self.SIZE} rows, got {len(grid)}"
            )
            return errors

        for row_idx, row in enumerate(grid):
            if not isinstance(row, list):
                errors.append(
                    f"Row {row_idx} must be a list, "
                    f"got {type(row).__name__}"
                )
                continue

            if len(row) != self.SIZE:
                errors.append(
                    f"Row {row_idx} must have {self.SIZE} columns, "
                    f"got {len(row)}"
                )
                continue

            for col_idx, value in enumerate(row):
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(
                        f"Cell ({row_idx}, {col_idx}) must be an int, "
                        f"got {type(value).__name__}"
                    )
                elif value < 0 or value > self.SIZE:
                    errors.append(
                        f"Cell ({row_idx}, {col_idx}) value {value} "
                        f"out of range 0-{self.SIZE}"
                    )

        return errors

    def validate_constraints(self, grid: list[list[int]]) -> list[str]:
        """Check that no Sudoku constraints are violated.

        Validates that no digit 1-9 appears more than once in any row,
        column, or 3x3 box.  Zeros (empty cells) are skipped.

        This method assumes the grid is structurally valid.  Call
        :meth:`validate_structure` first.

        Args:
            grid: A structurally valid 9x9 puzzle grid.

        Returns:
            A list of error message strings describing each constraint
            violation.  Empty if no violations are found.
        """
        errors: list[str] = []
        errors.extend(self._check_rows(grid))
        errors.extend(self._check_columns(grid))
        errors.extend(self._check_boxes(grid))
        return errors

    def validate_solvability(self, grid: list[list[int]]) -> list[str]:
        """Check for solvability warnings.

        Currently checks whether the puzzle has fewer than 17 clues,
        which is the proven minimum for a uniquely solvable 9x9 Sudoku.

        This method assumes the grid is structurally valid.  Call
        :meth:`validate_structure` first.

        Args:
            grid: A structurally valid 9x9 puzzle grid.

        Returns:
            A list of warning message strings.  Empty if no warnings.
        """
        warnings: list[str] = []
        clue_count = sum(
            1 for row in grid for value in row if value != 0
        )
        if clue_count < self.MIN_CLUES:
            warnings.append(
                f"Puzzle has {clue_count} clues, fewer than the minimum "
                f"{self.MIN_CLUES} required for a unique solution"
            )
        return warnings

    def _check_rows(self, grid: list[list[int]]) -> list[str]:
        """Find duplicate digits in each row.

        Args:
            grid: A structurally valid 9x9 puzzle grid.

        Returns:
            A list of error messages for each duplicate found.
        """
        errors: list[str] = []
        for row_idx in range(self.SIZE):
            seen: dict[int, int] = {}
            for col_idx in range(self.SIZE):
                digit = grid[row_idx][col_idx]
                if digit == 0:
                    continue
                if digit in seen:
                    errors.append(
                        f"Duplicate digit {digit} in row {row_idx}"
                    )
                else:
                    seen[digit] = col_idx
        return errors

    def _check_columns(self, grid: list[list[int]]) -> list[str]:
        """Find duplicate digits in each column.

        Args:
            grid: A structurally valid 9x9 puzzle grid.

        Returns:
            A list of error messages for each duplicate found.
        """
        errors: list[str] = []
        for col_idx in range(self.SIZE):
            seen: dict[int, int] = {}
            for row_idx in range(self.SIZE):
                digit = grid[row_idx][col_idx]
                if digit == 0:
                    continue
                if digit in seen:
                    errors.append(
                        f"Duplicate digit {digit} in column {col_idx}"
                    )
                else:
                    seen[digit] = row_idx
        return errors

    def _check_boxes(self, grid: list[list[int]]) -> list[str]:
        """Find duplicate digits in each 3x3 box.

        Args:
            grid: A structurally valid 9x9 puzzle grid.

        Returns:
            A list of error messages for each duplicate found.
        """
        errors: list[str] = []
        for box_idx in range(self.SIZE):
            box_row_start = (box_idx // self.BOX_SIZE) * self.BOX_SIZE
            box_col_start = (box_idx % self.BOX_SIZE) * self.BOX_SIZE
            seen: dict[int, tuple[int, int]] = {}
            for row_offset in range(self.BOX_SIZE):
                for col_offset in range(self.BOX_SIZE):
                    row_idx = box_row_start + row_offset
                    col_idx = box_col_start + col_offset
                    digit = grid[row_idx][col_idx]
                    if digit == 0:
                        continue
                    if digit in seen:
                        errors.append(
                            f"Duplicate digit {digit} in box {box_idx}"
                        )
                    else:
                        seen[digit] = (row_idx, col_idx)
        return errors
