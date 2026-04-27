"""Convert DLX solver output back into a human-readable Sudoku grid.

This module is the reverse of :mod:`constraint_mapper`: it takes the list of
``(row, col, digit)`` row_ids that the DLX solver selected and reconstructs
the solved 9x9 grid.  It also provides formatters for displaying the grid
in both human-readable (box-separated) and compact (81-character) forms.

Custom exceptions :class:`NoSolutionError` and :class:`MultipleSolutionsError`
are defined here for use by higher-level orchestration code that needs to
distinguish between zero, one, and multiple solutions.

Typical usage::

    from sudoku_solver.solution_decoder import SolutionDecoder

    decoder = SolutionDecoder()
    grid = decoder.decode(row_ids)   # row_ids from DLXSolver
    print(decoder.format_grid(grid))
"""

from __future__ import annotations


class NoSolutionError(Exception):
    """Raised when a Sudoku puzzle has no valid solution."""


class MultipleSolutionsError(Exception):
    """Raised when a Sudoku puzzle has more than one valid solution."""


class SolutionDecoder:
    """Decode DLX solver output into a Sudoku grid and format for display.

    This class converts the ``(row, col, digit)`` tuples yielded by
    :class:`DLXSolver` back into a 9x9 grid and provides two formatting
    methods: a human-readable box-separated layout and a compact
    81-character string.

    The class is stateless: all methods are pure functions of their
    arguments plus the ``SIZE`` constant.

    Class Attributes:
        SIZE: The dimension of the Sudoku grid (9).
        BOX_SIZE: The dimension of each 3x3 box (3).
        EXPECTED_PLACEMENTS: Total cells in a solved grid (81).
    """

    SIZE: int = 9
    BOX_SIZE: int = 3
    EXPECTED_PLACEMENTS: int = 81

    def decode(
        self, row_ids: list[tuple[int, int, int]]
    ) -> list[list[int]]:
        """Reconstruct a 9x9 grid from DLX solver row identifiers.

        Each row_id is a ``(row, col, digit)`` tuple produced by
        :meth:`SudokuConstraintMapper.build_row_id`.  The method creates
        a blank grid and places each digit at its designated position.

        Args:
            row_ids: Exactly 81 ``(row, col, digit)`` tuples representing
                the selected placements from the DLX solver.

        Returns:
            A 9x9 list of lists with digits 1-9 in every cell.

        Raises:
            ValueError: If *row_ids* does not contain exactly 81 entries.
        """
        if len(row_ids) != self.EXPECTED_PLACEMENTS:
            raise ValueError(
                f"Expected {self.EXPECTED_PLACEMENTS} row_ids, "
                f"got {len(row_ids)}"
            )

        grid = self._empty_grid()
        for row, col, digit in row_ids:
            grid[row][col] = digit

        return grid

    def format_grid(self, grid: list[list[int]]) -> str:
        """Format a 9x9 grid as a human-readable string with box separators.

        Produces output like::

            +---------+---------+---------+
            | 5  3  4 | 6  7  8 | 9  1  2 |
            | 6  7  2 | 1  9  5 | 3  4  8 |
            | 1  9  8 | 3  4  2 | 5  6  7 |
            +---------+---------+---------+
            | ...

        Args:
            grid: A 9x9 list of lists containing digits 1-9.

        Returns:
            A multi-line string with box-separated formatting.
        """
        separator = self._horizontal_separator()
        lines: list[str] = []

        for row_idx in range(self.SIZE):
            if row_idx % self.BOX_SIZE == 0:
                lines.append(separator)
            lines.append(self._format_row(grid[row_idx]))

        lines.append(separator)
        return "\n".join(lines)

    def format_grid_compact(self, grid: list[list[int]]) -> str:
        """Format a 9x9 grid as a compact 81-character string.

        Digits are concatenated row by row with no separators, producing
        a single line like ``"534678912672195348198342567..."``.

        Args:
            grid: A 9x9 list of lists containing digits 1-9.

        Returns:
            An 81-character string of digits.
        """
        return "".join(
            str(grid[row][col])
            for row in range(self.SIZE)
            for col in range(self.SIZE)
        )

    def _empty_grid(self) -> list[list[int]]:
        """Create a 9x9 grid initialized to zeros.

        Returns:
            A 9x9 list of lists filled with zeros.
        """
        return [[0] * self.SIZE for _ in range(self.SIZE)]

    def _horizontal_separator(self) -> str:
        """Build the horizontal box separator line.

        Returns:
            A string like ``+---------+---------+---------+``.
        """
        box_segment = "-" * (self.BOX_SIZE * 3)
        return "+" + "+".join([box_segment] * self.BOX_SIZE) + "+"

    def _format_row(self, row: list[int]) -> str:
        """Format a single row with vertical box separators.

        Args:
            row: A list of 9 digits.

        Returns:
            A string like ``| 5  3  4 | 6  7  8 | 9  1  2 |``.
        """
        parts: list[str] = []
        for box_start in range(0, self.SIZE, self.BOX_SIZE):
            box_end = box_start + self.BOX_SIZE
            box_digits = row[box_start:box_end]
            parts.append(" " + "  ".join(str(d) for d in box_digits) + " ")
        return "|" + "|".join(parts) + "|"
