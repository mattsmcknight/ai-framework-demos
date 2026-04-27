"""Sudoku solver using Dancing Links (DLX) / Algorithm X.

This package provides a production-ready Sudoku solver that reformulates
standard 9x9 puzzles as exact cover problems and solves them using Donald
Knuth's Dancing Links technique.

The public API consists of:

- **Data structures**: :class:`DLXNode`, :class:`ColumnHeader` -- the
  circular doubly-linked list nodes that form the sparse matrix.
- **Matrix**: :class:`DLXMatrix` -- the sparse binary matrix supporting
  cover/uncover operations.
- **Solver**: :class:`DLXSolver` -- a generic Algorithm X solver that
  yields solutions as lists of row identifiers.
- **Constraint mapping**: :class:`SudokuConstraintMapper` -- translates
  a Sudoku grid into an exact cover matrix, and :data:`SudokuGrid` -- the
  type alias for a 9x9 grid.
- **Solution decoding**: :class:`SolutionDecoder` -- converts solver
  output back into a readable grid.  :class:`NoSolutionError` and
  :class:`MultipleSolutionsError` signal solve-result edge cases.

Typical usage::

    from sudoku_solver import SudokuConstraintMapper, DLXSolver, SolutionDecoder

    mapper = SudokuConstraintMapper()
    matrix = mapper.build_matrix(grid)
    solver = DLXSolver(matrix)
    solution = solver.solve_one()
    if solution is not None:
        decoder = SolutionDecoder()
        solved_grid = decoder.decode(solution)
        print(decoder.format_grid(solved_grid))
"""

__version__ = "1.0.0"

from sudoku_solver.constraint_mapper import SudokuConstraintMapper, SudokuGrid
from sudoku_solver.dlx_matrix import DLXMatrix
from sudoku_solver.dlx_node import ColumnHeader, DLXNode
from sudoku_solver.solution_decoder import (
    MultipleSolutionsError,
    NoSolutionError,
    SolutionDecoder,
)
from sudoku_solver.solver import DLXSolver

__all__ = [
    "ColumnHeader",
    "DLXMatrix",
    "DLXNode",
    "DLXSolver",
    "MultipleSolutionsError",
    "NoSolutionError",
    "SolutionDecoder",
    "SudokuConstraintMapper",
    "SudokuGrid",
]
