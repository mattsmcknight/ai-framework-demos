"""Sparse matrix with circular doubly-linked lists for Dancing Links (DLX).

This module implements Knuth's Dancing Links data structure: a sparse binary
matrix represented as a network of circularly linked nodes.  The matrix
supports two core operations -- *cover* and *uncover* -- that add and remove
columns (constraints) and their associated rows in O(1) per link, with full
reversibility for backtracking.

The matrix is generic: it knows nothing about Sudoku or any specific problem
domain.  Callers populate it with column names (constraints) and rows
(choices), then use cover/uncover together with Algorithm X to find exact
covers.

Key concepts:

- **Column headers** form a circular horizontal list anchored at a ``root``
  sentinel.  Each header tracks how many nodes remain in its column (``size``),
  enabling the minimum-remaining-values heuristic.

- **Data nodes** sit at every "1" position in the logical binary matrix.
  Each node belongs to exactly one column (vertical circular list) and one
  row (horizontal circular list).

- **Cover** removes a column header from the header list and removes every
  row containing a 1 in that column from all *other* columns those rows
  touch.  The removed nodes retain their link pointers, so the operation is
  reversible.

- **Uncover** is the exact reverse of cover, restoring all removed nodes and
  the column header to their original positions.

Typical usage::

    matrix = DLXMatrix(["A", "B", "C", "D"])
    matrix.add_row(0, ["A", "B"])
    matrix.add_row(1, ["B", "C"])
    matrix.add_row(2, ["A", "D"])
    # ... then use Algorithm X with cover/uncover to search for exact covers
"""

from __future__ import annotations

from typing import Any

from sudoku_solver.dlx_node import ColumnHeader, DLXNode


class DLXMatrix:
    """A sparse matrix for the exact cover problem using Dancing Links.

    The matrix is built from column headers (one per constraint) linked in a
    circular horizontal list, with a ``root`` sentinel acting as the anchor.
    Rows (choices) are added via :meth:`add_row`, which creates data nodes
    linked both horizontally within the row and vertically within their
    respective columns.

    Attributes:
        root: The master sentinel header that anchors the column header list.
            It is never covered or uncovered.
        columns: A dictionary mapping column names to their
            :class:`ColumnHeader` instances for O(1) lookup.
    """

    __slots__ = ("root", "columns")

    def __init__(self, column_names: list[str]) -> None:
        """Initialize the matrix with the given constraint columns.

        Creates one :class:`ColumnHeader` per name and links them into a
        circular doubly-linked list with the ``root`` sentinel.  Each
        column's vertical list starts empty (up/down pointing to itself).

        Args:
            column_names: The names of the constraints (columns) in the
                exact cover matrix.  Order determines the initial left-to-right
                arrangement of the header list.
        """
        self.root: ColumnHeader = ColumnHeader("root")
        self.columns: dict[str, ColumnHeader] = {}

        prev = self.root
        for name in column_names:
            header = ColumnHeader(name)
            self.columns[name] = header
            # Insert header to the right of prev in the horizontal list.
            header.left = prev
            header.right = prev.right
            prev.right.left = header
            prev.right = header
            prev = header

    def add_row(self, row_id: Any, column_names: list[str]) -> None:
        """Add a row (choice) to the matrix.

        Creates one :class:`DLXNode` for each column listed, links them
        horizontally into a circular row list, and appends each node to
        the bottom of its column's vertical list.

        Args:
            row_id: An identifier for this row, stored on every node in the
                row so that solutions can be reconstructed later.
            column_names: The columns in which this row has a 1.  Must be
                a non-empty subset of the column names provided at
                construction time.

        Raises:
            ValueError: If *column_names* is empty or contains a name that
                does not correspond to an existing column.
        """
        if not column_names:
            raise ValueError("column_names must not be empty")

        first_node: DLXNode | None = None

        for name in column_names:
            header = self.columns.get(name)
            if header is None:
                raise ValueError(
                    f"Unknown column name: {name!r}. "
                    f"Valid columns: {sorted(self.columns)}"
                )

            node = DLXNode(column=header, row_id=row_id)

            # -- Append to the bottom of the column's vertical list --
            # The column header's `up` always points to the last node in
            # the column.  Insert between that last node and the header.
            node.up = header.up
            node.down = header
            header.up.down = node
            header.up = node
            header.size += 1

            # -- Link horizontally within the row --
            if first_node is None:
                first_node = node
                # Single-node circular list: node points to itself (default).
            else:
                # Insert to the left of first_node, which puts the new node
                # at the "end" of the circular row list.
                node.left = first_node.left
                node.right = first_node
                first_node.left.right = node
                first_node.left = node

    def cover(self, column: ColumnHeader) -> None:
        """Remove *column* and all rows that intersect it from the matrix.

        This is Knuth's "cover" operation:

        1. Remove *column* from the header list (left/right unlinking).
        2. For each data node in the column (top to bottom):
           for each *other* node in that node's row (left to right):
           unlink that node from its own column (up/down unlinking)
           and decrement that column's size.

        The removed nodes retain their link pointers, so :meth:`uncover`
        can reverse this operation exactly.

        Args:
            column: The column header to cover.
        """
        # Remove column header from the horizontal header list.
        column.right.left = column.left
        column.left.right = column.right

        # Walk down the column, removing each row from other columns.
        row_node = column.down
        while row_node is not column:
            right_node = row_node.right
            while right_node is not row_node:
                right_node.down.up = right_node.up
                right_node.up.down = right_node.down
                right_node.column.size -= 1
                right_node = right_node.right
            row_node = row_node.down

    def uncover(self, column: ColumnHeader) -> None:
        """Reverse a previous :meth:`cover` of *column*.

        This is the exact inverse of cover: it iterates in the opposite
        direction (bottom to top, right to left) and re-links every node
        that was removed.

        Args:
            column: The column header to uncover.  Must have been the most
                recently covered column that has not yet been uncovered
                (i.e., cover/uncover calls must be balanced in LIFO order).
        """
        # Walk up the column (bottom to top), restoring each row.
        row_node = column.up
        while row_node is not column:
            left_node = row_node.left
            while left_node is not row_node:
                left_node.column.size += 1
                left_node.down.up = left_node
                left_node.up.down = left_node
                left_node = left_node.left
            row_node = row_node.up

        # Restore column header in the horizontal header list.
        column.right.left = column
        column.left.right = column

    def choose_column(self) -> ColumnHeader:
        """Choose the column with the fewest remaining nodes.

        This implements the "minimum remaining values" (MRV) heuristic,
        also called the "S heuristic" in Knuth's paper.  Choosing the
        smallest column minimizes the branching factor and dramatically
        prunes the search space.

        Returns:
            The :class:`ColumnHeader` with the smallest ``size``.

        Note:
            Behavior is undefined if the matrix is empty
            (``root.right is root``).  Callers should check
            :meth:`is_empty` before calling this method.
        """
        best: ColumnHeader = self.root.right  # type: ignore[assignment]
        current = best.right
        while current is not self.root:
            if current.size < best.size:  # type: ignore[union-attr]
                best = current  # type: ignore[assignment]
            current = current.right
        return best

    def is_empty(self) -> bool:
        """Check whether all columns have been covered.

        Returns:
            ``True`` if no columns remain in the header list (meaning the
            current partial solution is a valid exact cover), ``False``
            otherwise.
        """
        return self.root.right is self.root
