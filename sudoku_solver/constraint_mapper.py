"""Sudoku-to-exact-cover constraint translation for Dancing Links.

This module is the bridge between the Sudoku domain and the generic DLX solver.
It encodes the four Sudoku constraints (cell, row-digit, column-digit, box-digit)
as columns in an exact cover matrix and maps each possible digit placement to a
row covering exactly four of those constraints.

A standard 9x9 Sudoku produces:
- 324 constraint columns (81 cell + 81 row + 81 col + 81 box)
- Up to 729 choice rows (9 digits x 9 rows x 9 cols)

Pre-filled cells are handled by adding only their single valid row to the
matrix (rather than all nine digit options), which lets the DLX solver
implicitly enforce givens without any special-case logic.

This module is the ONLY place that contains Sudoku-specific constraint
encoding.  The DLX solver, matrix, and node classes remain fully generic.

Typical usage::

    from sudoku_solver.constraint_mapper import SudokuConstraintMapper

    mapper = SudokuConstraintMapper()
    matrix = mapper.build_matrix(grid)
    # Pass matrix to DLXSolver to find solutions
"""

from __future__ import annotations

from sudoku_solver.dlx_matrix import DLXMatrix

SudokuGrid = list[list[int]]
"""A 9x9 grid where 0 indicates an empty cell and 1-9 are placed digits."""


class SudokuConstraintMapper:
    """Translate a Sudoku puzzle into an exact cover matrix for DLX.

    This class encodes the four standard Sudoku constraints as columns in
    a :class:`DLXMatrix` and converts each possible digit placement into
    a matrix row covering exactly four constraints.  It is stateless: all
    configuration lives in class-level constants, and each method is a
    pure function of its arguments.

    Class Attributes:
        SIZE: The dimension of the Sudoku grid (9).
        BOX_SIZE: The dimension of each 3x3 box (3).
        NUM_CONSTRAINTS: Total constraint columns (324).
        NUM_CHOICES: Total possible placements (729).
    """

    SIZE: int = 9
    BOX_SIZE: int = 3
    NUM_CONSTRAINTS: int = 324
    NUM_CHOICES: int = 729

    def build_column_names(self) -> list[str]:
        """Generate the 324 constraint column names for a 9x9 Sudoku.

        The columns are ordered in four groups of 81:

        1. **Cell constraints** (``cell_r{row}_c{col}``): each cell must
           contain exactly one digit.
        2. **Row-digit constraints** (``row_r{row}_d{digit}``): each row
           must contain each digit 1-9 exactly once.
        3. **Column-digit constraints** (``col_c{col}_d{digit}``): each
           column must contain each digit 1-9 exactly once.
        4. **Box-digit constraints** (``box_b{box}_d{digit}``): each 3x3
           box must contain each digit 1-9 exactly once.

        Returns:
            A list of 324 unique constraint name strings.
        """
        names: list[str] = []

        # Cell constraints: one digit per cell.
        for row in range(self.SIZE):
            for col in range(self.SIZE):
                names.append(f"cell_r{row}_c{col}")

        # Row-digit constraints: each digit appears once per row.
        for row in range(self.SIZE):
            for digit in range(1, self.SIZE + 1):
                names.append(f"row_r{row}_d{digit}")

        # Column-digit constraints: each digit appears once per column.
        for col in range(self.SIZE):
            for digit in range(1, self.SIZE + 1):
                names.append(f"col_c{col}_d{digit}")

        # Box-digit constraints: each digit appears once per 3x3 box.
        for box in range(self.SIZE):
            for digit in range(1, self.SIZE + 1):
                names.append(f"box_b{box}_d{digit}")

        return names

    def build_row_constraints(
        self, row: int, col: int, digit: int
    ) -> list[str]:
        """Return the four constraint names satisfied by a placement.

        Placing *digit* at position (*row*, *col*) satisfies exactly one
        constraint from each of the four groups: cell, row-digit,
        column-digit, and box-digit.

        Args:
            row: Row index (0-8).
            col: Column index (0-8).
            digit: Digit value (1-9).

        Returns:
            A list of four constraint column name strings.
        """
        box = (row // self.BOX_SIZE) * self.BOX_SIZE + (col // self.BOX_SIZE)
        return [
            f"cell_r{row}_c{col}",
            f"row_r{row}_d{digit}",
            f"col_c{col}_d{digit}",
            f"box_b{box}_d{digit}",
        ]

    def build_row_id(
        self, row: int, col: int, digit: int
    ) -> tuple[int, int, int]:
        """Build the row identifier for a digit placement.

        The returned tuple uniquely identifies a placement and is stored
        on every :class:`DLXNode` in the matrix row.  The solution decoder
        uses these tuples to reconstruct the solved grid.

        Args:
            row: Row index (0-8).
            col: Column index (0-8).
            digit: Digit value (1-9).

        Returns:
            A ``(row, col, digit)`` tuple.
        """
        return (row, col, digit)

    def build_matrix(self, grid: SudokuGrid) -> DLXMatrix:
        """Build the exact cover matrix for a Sudoku puzzle.

        Creates a :class:`DLXMatrix` with 324 constraint columns, then
        adds rows for each possible digit placement.  For pre-filled
        cells (nonzero values in *grid*), only the given digit's row is
        added.  For empty cells (zeros), all nine digit rows are added.

        Pre-filled cells still produce a matrix row (just one instead of
        nine) so that the solver can cover their constraints naturally
        without special-case handling.

        This method does NOT validate the grid.  Grid validation is the
        responsibility of ``validator.py``.

        Args:
            grid: A 9x9 list of lists where 0 indicates an empty cell
                and 1-9 are placed digits.

        Returns:
            A populated :class:`DLXMatrix` ready for the DLX solver.
        """
        column_names = self.build_column_names()
        matrix = DLXMatrix(column_names)

        for row in range(self.SIZE):
            for col in range(self.SIZE):
                cell_value = grid[row][col]
                digits = (
                    [cell_value]
                    if cell_value != 0
                    else list(range(1, self.SIZE + 1))
                )
                for digit in digits:
                    row_id = self.build_row_id(row, col, digit)
                    constraints = self.build_row_constraints(row, col, digit)
                    matrix.add_row(row_id, constraints)

        return matrix
