"""Node and column header data structures for Dancing Links (DLX).

This module provides the fundamental building blocks for Knuth's Dancing Links
technique: a sparse matrix representation using circular doubly-linked lists.

Each node participates in two circular lists (horizontal and vertical) and holds
a reference to its column header. Column headers track the number of nodes in
their column, enabling the minimum-remaining-values heuristic used by Algorithm X
to choose which constraint to satisfy next.

Both classes use ``__slots__`` for memory efficiency, since a typical Sudoku
exact-cover matrix contains thousands of nodes whose link-manipulation speed
directly determines solver performance.

Typical usage::

    header = ColumnHeader("cell_0_0")
    node = DLXNode(column=header, row_id=42)
"""

from __future__ import annotations


class DLXNode:
    """A single node in the DLX sparse matrix.

    Each node sits at the intersection of one row and one column in the
    exact-cover matrix.  It maintains four directional links (left, right,
    up, down) that form two circular doubly-linked lists -- one horizontal
    (connecting all 1s in the same row) and one vertical (connecting all 1s
    in the same column).

    Attributes:
        left: The node to the left in the same row.
        right: The node to the right in the same row.
        up: The node above in the same column.
        down: The node below in the same column.
        column: The column header this node belongs to.
        row_id: An optional identifier for the exact-cover row this node
            belongs to.  Used to reconstruct solutions after Algorithm X
            finds a valid cover.
    """

    __slots__ = ("left", "right", "up", "down", "column", "row_id")

    def __init__(
        self,
        column: ColumnHeader | None = None,
        row_id: int | None = None,
    ) -> None:
        """Initialize an isolated node with all links pointing to itself.

        Args:
            column: The column header this node belongs to.  May be ``None``
                for nodes that are not yet inserted into a matrix.
            row_id: An optional integer identifying the exact-cover row.
                For Sudoku, this typically encodes (row, col, digit).
        """
        self.left: DLXNode = self
        self.right: DLXNode = self
        self.up: DLXNode = self
        self.down: DLXNode = self
        self.column: ColumnHeader | None = column
        self.row_id: int | None = row_id

    def __repr__(self) -> str:
        """Return a developer-friendly representation of this node.

        Returns:
            A string showing the row_id if set, or a fallback
            indicating the node's identity.
        """
        if self.row_id is not None:
            return f"DLXNode(row_id={self.row_id})"
        return f"DLXNode(id={id(self):#x})"


class ColumnHeader(DLXNode):
    """A column header in the DLX sparse matrix.

    Column headers serve a dual role: they are sentinel nodes at the top of
    each column's circular vertical list, and they carry metadata used by
    Algorithm X -- specifically the column ``name`` (for debugging and
    constraint identification) and ``size`` (the count of 1s remaining in
    the column, used by the minimum-remaining-values heuristic).

    Column headers are also linked horizontally into their own circular list,
    anchored by a root header node.  Covering a column removes its header
    from this horizontal list; uncovering restores it.

    Attributes:
        name: A string identifier for the constraint this column represents
            (e.g., ``"cell_0_0"`` or ``"row_3_digit_7"``).
        size: The number of 1-nodes currently linked in this column.
            Maintained by matrix cover/uncover operations.
    """

    __slots__ = ("name", "size")

    def __init__(self, name: str) -> None:
        """Initialize a column header with a given name and zero size.

        The header's ``column`` attribute is set to itself, following the
        DLX convention where column headers are their own column reference.

        Args:
            name: A human-readable identifier for this constraint column.
        """
        super().__init__(column=None, row_id=None)
        self.column: ColumnHeader = self  # type: ignore[assignment]
        self.name: str = name
        self.size: int = 0

    def __repr__(self) -> str:
        """Return a developer-friendly representation of this column header.

        Returns:
            A string showing the column name and current size.
        """
        return f"ColumnHeader(name={self.name!r}, size={self.size})"
