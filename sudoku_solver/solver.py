"""Generic Algorithm X solver using Dancing Links (DLX).

This module implements Donald Knuth's Algorithm X -- a recursive,
depth-first, backtracking algorithm that finds all solutions to the
exact cover problem.  It operates on a :class:`DLXMatrix` and is
completely domain-agnostic: it knows nothing about Sudoku or any other
problem.  The caller populates the matrix and interprets the resulting
row_ids.

The solver is generator-based: it yields solutions one at a time, keeping
memory usage bounded regardless of how many solutions exist.  The matrix
is left in its original state after the solve completes -- whether the
generator is fully exhausted, abandoned early, or closed explicitly --
because every cover operation is protected by a ``try/finally`` block
that guarantees the matching uncover.

Typical usage::

    from sudoku_solver.dlx_matrix import DLXMatrix
    from sudoku_solver.solver import DLXSolver

    matrix = DLXMatrix(["A", "B", "C"])
    matrix.add_row("r1", ["A", "C"])
    matrix.add_row("r2", ["B"])
    solver = DLXSolver(matrix)
    for solution in solver.solve():
        print(solution)  # e.g. ["r1", "r2"]
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from sudoku_solver.dlx_matrix import DLXMatrix
from sudoku_solver.dlx_node import DLXNode


class DLXSolver:
    """Solve exact cover problems on a DLX matrix using Algorithm X.

    The solver wraps a :class:`DLXMatrix` and searches for subsets of rows
    that cover every column exactly once.  Solutions are yielded as lists
    of ``row_id`` values (the identifiers passed to
    :meth:`DLXMatrix.add_row`), so the caller can map them back to
    domain-specific meanings.

    Attributes:
        matrix: The DLX matrix to solve.
        max_solutions: Stop after finding this many solutions.
            Zero means find all solutions.
    """

    __slots__ = ("matrix", "max_solutions", "_solution", "_solution_count")

    def __init__(
        self,
        matrix: DLXMatrix,
        max_solutions: int = 1,
    ) -> None:
        """Initialize the solver for the given matrix.

        Args:
            matrix: A populated :class:`DLXMatrix` representing the exact
                cover problem to solve.
            max_solutions: Maximum number of solutions to yield.  Defaults
                to 1 (find first solution only).  Pass 0 to find all
                solutions.
        """
        self.matrix = matrix
        self.max_solutions = max_solutions
        self._solution: list[Any] = []
        self._solution_count: int = 0

    def solve(self) -> Generator[list[Any], None, None]:
        """Search for exact covers and yield each solution found.

        Each solution is a list of ``row_id`` values identifying the rows
        that form a valid exact cover.  The matrix is guaranteed to be in
        its original state after the generator is exhausted or abandoned,
        because each cover/uncover pair in the recursive search is wrapped
        in ``try/finally``.

        Yields:
            A list of row_ids for each exact cover found.
        """
        self._solution = []
        self._solution_count = 0
        yield from self._search(depth=0)

    def _search(self, depth: int) -> Generator[list[Any], None, None]:
        """Recursively search for exact covers (Algorithm X core).

        This implements Knuth's Algorithm X:

        1. If the matrix is empty, the current partial solution is valid.
        2. Choose the column with the fewest nodes (MRV heuristic).
        3. If that column has no nodes, this is a dead end -- backtrack.
        4. Cover the chosen column.
        5. Try each row in the column: cover its other columns, recurse,
           then uncover in reverse order.
        6. Uncover the chosen column.

        The reverse iteration order when uncovering row nodes (left instead
        of right) is critical: it ensures the uncover operations mirror the
        cover operations in strict LIFO order.

        Every cover operation is wrapped in ``try/finally`` so that even if
        the generator is closed mid-search (via ``GeneratorExit``), all
        outstanding covers are properly reversed and the matrix is restored
        to its original state.

        Args:
            depth: Current recursion depth (used for diagnostic purposes).

        Yields:
            A list of row_ids for each exact cover found.
        """
        if self._reached_solution_limit():
            return

        if self.matrix.is_empty():
            self._solution_count += 1
            yield list(self._solution)
            return

        column = self.matrix.choose_column()

        if column.size == 0:
            return

        self.matrix.cover(column)
        try:
            row_node = column.down
            while row_node is not column:
                self._solution.append(row_node.row_id)

                # Cover all other columns in this row (iterate right).
                self._cover_row(row_node)
                try:
                    yield from self._search(depth + 1)
                finally:
                    # Uncover all other columns in reverse order (left).
                    self._uncover_row(row_node)

                self._solution.pop()

                if self._reached_solution_limit():
                    break

                row_node = row_node.down
        finally:
            self.matrix.uncover(column)

    def _cover_row(self, row_node: DLXNode) -> None:
        """Cover all columns touched by the other nodes in *row_node*'s row.

        Iterates right from *row_node*, covering each encountered node's
        column.  The starting *row_node* itself is skipped (its column was
        already covered by the caller).

        Args:
            row_node: A node in the row to cover.
        """
        other = row_node.right
        while other is not row_node:
            self.matrix.cover(other.column)
            other = other.right

    def _uncover_row(self, row_node: DLXNode) -> None:
        """Uncover all columns touched by the other nodes in *row_node*'s row.

        Iterates left from *row_node* (reverse of :meth:`_cover_row`),
        uncovering each encountered node's column.  The reverse order is
        critical for correctness -- it mirrors the cover order in LIFO
        fashion.

        Args:
            row_node: A node in the row to uncover.
        """
        other = row_node.left
        while other is not row_node:
            self.matrix.uncover(other.column)
            other = other.left

    def _reached_solution_limit(self) -> bool:
        """Check whether the solver has found enough solutions.

        Returns:
            ``True`` if ``max_solutions`` is nonzero and has been reached,
            ``False`` otherwise.
        """
        if self.max_solutions == 0:
            return False
        return self._solution_count >= self.max_solutions

    def solve_one(self) -> list[Any] | None:
        """Find and return the first solution, or ``None`` if none exists.

        This is a convenience method that internally uses :meth:`solve`
        with early termination after the first result.  The matrix is
        guaranteed to be restored to its original state.

        Returns:
            A list of row_ids forming an exact cover, or ``None`` if the
            problem has no solution.
        """
        original_max = self.max_solutions
        self.max_solutions = 1
        gen = self.solve()
        try:
            for solution in gen:
                return solution
            return None
        finally:
            gen.close()
            self.max_solutions = original_max

    def count_solutions(self, limit: int = 0) -> int:
        """Count the number of solutions without storing them.

        This is memory-efficient: it iterates through solutions but only
        keeps a running count.  Useful for checking uniqueness -- pass
        ``limit=2`` to answer "does this problem have more than one
        solution?" without enumerating all of them.

        Args:
            limit: Stop counting after this many solutions.  Zero means
                count all solutions.

        Returns:
            The number of solutions found (up to *limit* if nonzero).
        """
        original_max = self.max_solutions
        self.max_solutions = limit
        gen = self.solve()
        try:
            count = 0
            for _ in gen:
                count += 1
            return count
        finally:
            gen.close()
            self.max_solutions = original_max
