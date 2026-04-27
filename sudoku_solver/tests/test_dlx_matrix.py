"""Unit tests for DLXMatrix sparse matrix data structure."""

from __future__ import annotations

from typing import Any

import pytest

from sudoku_solver.dlx_matrix import DLXMatrix
from sudoku_solver.dlx_node import ColumnHeader, DLXNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_headers(matrix: DLXMatrix) -> list[str]:
    """Traverse the header list left-to-right and return column names."""
    names: list[str] = []
    current = matrix.root.right
    while current is not matrix.root:
        assert isinstance(current, ColumnHeader)
        names.append(current.name)
        current = current.right
    return names


def _collect_headers_reverse(matrix: DLXMatrix) -> list[str]:
    """Traverse the header list right-to-left and return column names."""
    names: list[str] = []
    current = matrix.root.left
    while current is not matrix.root:
        assert isinstance(current, ColumnHeader)
        names.append(current.name)
        current = current.left
    return names


def _collect_column_row_ids(column: ColumnHeader) -> list[Any]:
    """Walk down a column and return the row_ids of all data nodes."""
    row_ids: list[Any] = []
    node = column.down
    while node is not column:
        row_ids.append(node.row_id)
        node = node.down
    return row_ids


def _collect_row_columns(node: DLXNode) -> list[str]:
    """Walk a row (starting from node) and return column names."""
    names: list[str] = [node.column.name]
    current = node.right
    while current is not node:
        names.append(current.column.name)
        current = current.right
    return names


def _snapshot_matrix(matrix: DLXMatrix) -> dict[str, Any]:
    """Capture full matrix state for comparison.

    Returns a dictionary containing:
    - header_names: the ordered list of visible column names
    - columns: for each column, its size and list of row_ids
    """
    header_names = _collect_headers(matrix)
    columns: dict[str, dict[str, Any]] = {}
    for name in header_names:
        col = matrix.columns[name]
        columns[name] = {
            "size": col.size,
            "row_ids": _collect_column_row_ids(col),
        }
    return {"header_names": header_names, "columns": columns}


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestDLXMatrixConstructor:
    """Verify matrix construction and header list structure."""

    def test_empty_matrix_root_links_to_self(self) -> None:
        """An empty matrix (no columns) should have root linking to itself."""
        matrix = DLXMatrix([])
        assert matrix.root.left is matrix.root
        assert matrix.root.right is matrix.root

    def test_empty_matrix_is_empty(self) -> None:
        """An empty matrix should report is_empty True."""
        matrix = DLXMatrix([])
        assert matrix.is_empty()

    def test_single_column(self) -> None:
        """A matrix with one column should have that column linked to root."""
        matrix = DLXMatrix(["A"])
        assert _collect_headers(matrix) == ["A"]
        assert not matrix.is_empty()

    def test_multiple_columns_order(self) -> None:
        """Columns should appear in the order they were provided."""
        names = ["A", "B", "C", "D", "E"]
        matrix = DLXMatrix(names)
        assert _collect_headers(matrix) == names

    def test_header_list_is_circular(self) -> None:
        """The header list should be circular: forward and reverse give the same names."""
        names = ["X", "Y", "Z"]
        matrix = DLXMatrix(names)
        assert _collect_headers(matrix) == names
        assert _collect_headers_reverse(matrix) == list(reversed(names))

    def test_column_lookup_by_name(self) -> None:
        """Each column should be retrievable from the columns dict by name."""
        names = ["cell_0_0", "row_1_digit_5", "box_3_digit_9"]
        matrix = DLXMatrix(names)
        for name in names:
            assert name in matrix.columns
            assert matrix.columns[name].name == name

    def test_columns_start_empty(self) -> None:
        """Each column should start with size 0 and up/down pointing to itself."""
        matrix = DLXMatrix(["A", "B"])
        for name in ["A", "B"]:
            col = matrix.columns[name]
            assert col.size == 0
            assert col.up is col
            assert col.down is col


# ---------------------------------------------------------------------------
# add_row tests
# ---------------------------------------------------------------------------


class TestAddRow:
    """Verify row addition, linking, and error handling."""

    def test_single_node_row(self) -> None:
        """A row with one column should create a single self-linked node."""
        matrix = DLXMatrix(["A", "B"])
        matrix.add_row(0, ["A"])

        col_a = matrix.columns["A"]
        assert col_a.size == 1
        node = col_a.down
        assert node.row_id == 0
        assert node.column is col_a
        # Horizontally self-linked (only node in the row).
        assert node.left is node
        assert node.right is node
        # Vertically linked back to header.
        assert node.down is col_a
        assert node.up is col_a
        # Column B unchanged.
        assert matrix.columns["B"].size == 0

    def test_multi_column_row_horizontal_circularity(self) -> None:
        """A row spanning multiple columns should form a circular horizontal list."""
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(0, ["A", "B", "C"])

        # Collect column names by walking the row starting from column A's node.
        first_node = matrix.columns["A"].down
        row_cols = _collect_row_columns(first_node)
        assert row_cols == ["A", "B", "C"]

        # Verify reverse direction.
        reverse_cols: list[str] = [first_node.column.name]
        current = first_node.left
        while current is not first_node:
            reverse_cols.append(current.column.name)
            current = current.left
        assert reverse_cols == ["A", "C", "B"]

    def test_multiple_rows_vertical_ordering(self) -> None:
        """Multiple rows in the same column should be ordered top-to-bottom by insertion."""
        matrix = DLXMatrix(["A"])
        matrix.add_row(10, ["A"])
        matrix.add_row(20, ["A"])
        matrix.add_row(30, ["A"])

        col_a = matrix.columns["A"]
        assert col_a.size == 3
        row_ids = _collect_column_row_ids(col_a)
        assert row_ids == [10, 20, 30]

    def test_multiple_rows_sizes(self) -> None:
        """Adding rows to different columns should increment each column's size correctly."""
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(0, ["A", "C"])
        matrix.add_row(1, ["B", "C"])
        matrix.add_row(2, ["A", "B", "C"])

        assert matrix.columns["A"].size == 2
        assert matrix.columns["B"].size == 2
        assert matrix.columns["C"].size == 3

    def test_add_row_invalid_column_raises_value_error(self) -> None:
        """Adding a row with an unknown column name should raise ValueError."""
        matrix = DLXMatrix(["A", "B"])
        with pytest.raises(ValueError, match="Unknown column name: 'Z'"):
            matrix.add_row(0, ["A", "Z"])

    def test_add_row_empty_columns_raises_value_error(self) -> None:
        """Adding a row with an empty column list should raise ValueError."""
        matrix = DLXMatrix(["A"])
        with pytest.raises(ValueError, match="must not be empty"):
            matrix.add_row(0, [])


# ---------------------------------------------------------------------------
# cover / uncover tests
# ---------------------------------------------------------------------------


class TestCoverUncover:
    """Verify cover/uncover operations and their exact invertibility."""

    def test_cover_removes_column_from_header_list(self) -> None:
        """Covering a column should remove it from the header traversal."""
        matrix = DLXMatrix(["A", "B", "C"])
        col_b = matrix.columns["B"]
        matrix.cover(col_b)
        assert _collect_headers(matrix) == ["A", "C"]

    def test_cover_removes_conflicting_rows(self) -> None:
        """Covering a column should remove rows that intersect it from other columns."""
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(0, ["A", "B"])
        matrix.add_row(1, ["B", "C"])
        matrix.add_row(2, ["A", "C"])

        # Cover B: rows 0 and 1 should be removed.
        # Row 0 touches A, so A loses row 0.
        # Row 1 touches C, so C loses row 1.
        # Row 2 (only in A, C) should remain.
        col_b = matrix.columns["B"]
        matrix.cover(col_b)

        assert matrix.columns["A"].size == 1
        assert _collect_column_row_ids(matrix.columns["A"]) == [2]
        assert matrix.columns["C"].size == 1
        assert _collect_column_row_ids(matrix.columns["C"]) == [2]

    def test_uncover_reverses_cover_exactly(self) -> None:
        """Matrix state after cover+uncover should be identical to the state before cover."""
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(0, ["A", "B"])
        matrix.add_row(1, ["B", "C"])
        matrix.add_row(2, ["A", "C"])

        before = _snapshot_matrix(matrix)

        col_b = matrix.columns["B"]
        matrix.cover(col_b)
        matrix.uncover(col_b)

        after = _snapshot_matrix(matrix)
        assert before == after

    def test_cover_uncover_multiple_columns(self) -> None:
        """Covering and uncovering multiple columns in LIFO order restores the matrix."""
        matrix = DLXMatrix(["A", "B", "C", "D"])
        matrix.add_row(0, ["A", "B"])
        matrix.add_row(1, ["B", "C"])
        matrix.add_row(2, ["C", "D"])
        matrix.add_row(3, ["A", "D"])

        before = _snapshot_matrix(matrix)

        col_a = matrix.columns["A"]
        col_c = matrix.columns["C"]

        # Cover A, then C.
        matrix.cover(col_a)
        matrix.cover(col_c)

        # Verify both are gone from headers.
        headers_during = _collect_headers(matrix)
        assert "A" not in headers_during
        assert "C" not in headers_during

        # Uncover in reverse order: C, then A.
        matrix.uncover(col_c)
        matrix.uncover(col_a)

        after = _snapshot_matrix(matrix)
        assert before == after

    def test_cover_all_and_uncover_all_restores_matrix(self) -> None:
        """Covering every column then uncovering in reverse should fully restore the matrix."""
        col_names = ["A", "B", "C", "D", "E"]
        matrix = DLXMatrix(col_names)
        matrix.add_row(0, ["A", "C", "E"])
        matrix.add_row(1, ["B", "D"])
        matrix.add_row(2, ["A", "B"])
        matrix.add_row(3, ["C", "D", "E"])

        before = _snapshot_matrix(matrix)

        # Cover all columns in order.
        covered: list[ColumnHeader] = []
        for name in col_names:
            col = matrix.columns[name]
            matrix.cover(col)
            covered.append(col)

        # Matrix should be empty.
        assert matrix.is_empty()

        # Uncover all in reverse order.
        for col in reversed(covered):
            matrix.uncover(col)

        after = _snapshot_matrix(matrix)
        assert before == after

    def test_cover_decrements_column_sizes(self) -> None:
        """Covering a column should decrement sizes of other affected columns."""
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(0, ["A", "B"])
        matrix.add_row(1, ["A", "C"])
        matrix.add_row(2, ["B", "C"])

        # Before cover: A=2, B=2, C=2
        assert matrix.columns["A"].size == 2
        assert matrix.columns["B"].size == 2
        assert matrix.columns["C"].size == 2

        # Cover A: removes rows 0 (touches B) and 1 (touches C).
        matrix.cover(matrix.columns["A"])
        assert matrix.columns["B"].size == 1  # lost row 0
        assert matrix.columns["C"].size == 1  # lost row 1

    def test_uncover_restores_column_sizes(self) -> None:
        """Uncovering should restore column sizes to their pre-cover values."""
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(0, ["A", "B"])
        matrix.add_row(1, ["A", "C"])
        matrix.add_row(2, ["B", "C"])

        col_a = matrix.columns["A"]
        matrix.cover(col_a)
        matrix.uncover(col_a)

        assert matrix.columns["A"].size == 2
        assert matrix.columns["B"].size == 2
        assert matrix.columns["C"].size == 2


# ---------------------------------------------------------------------------
# choose_column tests
# ---------------------------------------------------------------------------


class TestChooseColumn:
    """Verify the minimum-remaining-values column selection heuristic."""

    def test_returns_column_with_smallest_size(self) -> None:
        """choose_column should return the column with the fewest nodes."""
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(0, ["A", "B", "C"])
        matrix.add_row(1, ["A", "C"])
        # A=2, B=1, C=2 -> B is smallest.
        chosen = matrix.choose_column()
        assert chosen.name == "B"

    def test_tie_breaking_returns_leftmost(self) -> None:
        """When multiple columns have the same smallest size, return the leftmost."""
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(0, ["A"])
        matrix.add_row(1, ["B"])
        matrix.add_row(2, ["C"])
        # All size 1 -> should return A (leftmost).
        chosen = matrix.choose_column()
        assert chosen.name == "A"

    def test_empty_columns_preferred(self) -> None:
        """A column with size 0 should be chosen over columns with nodes."""
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(0, ["A", "C"])
        # B has size 0 -> should be chosen.
        chosen = matrix.choose_column()
        assert chosen.name == "B"

    def test_after_cover_reflects_updated_sizes(self) -> None:
        """choose_column should reflect size changes caused by cover operations."""
        matrix = DLXMatrix(["A", "B", "C"])
        matrix.add_row(0, ["A", "B"])
        matrix.add_row(1, ["A", "C"])
        matrix.add_row(2, ["B", "C"])
        # Before cover: A=2, B=2, C=2 -> A (leftmost tie).
        assert matrix.choose_column().name == "A"

        # Cover A: removes rows 0, 1.  B loses row 0 -> size 1.  C loses row 1 -> size 1.
        matrix.cover(matrix.columns["A"])
        # Now B=1, C=1 -> B (leftmost tie).
        assert matrix.choose_column().name == "B"


# ---------------------------------------------------------------------------
# Knuth's exact cover example
# ---------------------------------------------------------------------------


class TestKnuthExactCoverExample:
    """Build and verify the exact cover example from Knuth's Dancing Links paper.

    Universe: {1, 2, 3, 4, 5, 6, 7}
    Row A: {1, 4, 7}
    Row B: {1, 4}
    Row C: {4, 5, 7}
    Row D: {3, 5, 6}
    Row E: {2, 3, 6, 7}
    Row F: {2, 7}

    The unique exact cover solution is: {B, D, F}.
    """

    @pytest.fixture()
    def knuth_matrix(self) -> DLXMatrix:
        """Build Knuth's example matrix."""
        cols = ["1", "2", "3", "4", "5", "6", "7"]
        matrix = DLXMatrix(cols)
        matrix.add_row("A", ["1", "4", "7"])
        matrix.add_row("B", ["1", "4"])
        matrix.add_row("C", ["4", "5", "7"])
        matrix.add_row("D", ["3", "5", "6"])
        matrix.add_row("E", ["2", "3", "6", "7"])
        matrix.add_row("F", ["2", "7"])
        return matrix

    def test_column_sizes(self, knuth_matrix: DLXMatrix) -> None:
        """Verify that each column has the correct number of nodes."""
        expected_sizes = {
            "1": 2,  # A, B
            "2": 2,  # E, F
            "3": 2,  # D, E
            "4": 3,  # A, B, C
            "5": 2,  # C, D
            "6": 2,  # D, E
            "7": 4,  # A, C, E, F
        }
        for name, expected in expected_sizes.items():
            assert knuth_matrix.columns[name].size == expected, (
                f"Column {name}: expected size {expected}, "
                f"got {knuth_matrix.columns[name].size}"
            )

    def test_column_row_ids(self, knuth_matrix: DLXMatrix) -> None:
        """Verify that each column contains the correct rows (by row_id)."""
        expected_rows: dict[str, list[str]] = {
            "1": ["A", "B"],
            "2": ["E", "F"],
            "3": ["D", "E"],
            "4": ["A", "B", "C"],
            "5": ["C", "D"],
            "6": ["D", "E"],
            "7": ["A", "C", "E", "F"],
        }
        for name, expected in expected_rows.items():
            actual = _collect_column_row_ids(knuth_matrix.columns[name])
            assert actual == expected, (
                f"Column {name}: expected rows {expected}, got {actual}"
            )

    def test_row_horizontal_links(self, knuth_matrix: DLXMatrix) -> None:
        """Verify horizontal circularity of each row."""
        # Row A: columns 1, 4, 7
        node_a = knuth_matrix.columns["1"].down  # First node in col 1 is row A
        assert _collect_row_columns(node_a) == ["1", "4", "7"]

        # Row E: columns 2, 3, 6, 7
        col_2 = knuth_matrix.columns["2"]
        node_e = col_2.down  # First node in col 2 is row E
        assert node_e.row_id == "E"
        assert _collect_row_columns(node_e) == ["2", "3", "6", "7"]

    def test_all_headers_present(self, knuth_matrix: DLXMatrix) -> None:
        """All 7 columns should be in the header list."""
        assert _collect_headers(knuth_matrix) == ["1", "2", "3", "4", "5", "6", "7"]

    def test_link_integrity(self, knuth_matrix: DLXMatrix) -> None:
        """Every node's links should be reciprocal (n.right.left is n, etc.)."""
        # Check all header links.
        current = knuth_matrix.root.right
        while current is not knuth_matrix.root:
            assert current.right.left is current
            assert current.left.right is current
            # Check vertical links in this column.
            col = current
            node = col.down
            while node is not col:
                assert node.down.up is node
                assert node.up.down is node
                assert node.right.left is node
                assert node.left.right is node
                node = node.down
            current = current.right

    def test_cover_uncover_round_trip(self, knuth_matrix: DLXMatrix) -> None:
        """Cover and uncover all columns in sequence; matrix should be fully restored."""
        before = _snapshot_matrix(knuth_matrix)

        col_names = _collect_headers(knuth_matrix)
        covered: list[ColumnHeader] = []
        for name in col_names:
            col = knuth_matrix.columns[name]
            knuth_matrix.cover(col)
            covered.append(col)

        for col in reversed(covered):
            knuth_matrix.uncover(col)

        after = _snapshot_matrix(knuth_matrix)
        assert before == after

    def test_solve_finds_bdf(self, knuth_matrix: DLXMatrix) -> None:
        """A manual Algorithm X solve should find the unique solution {B, D, F}."""
        solutions: list[list[Any]] = []

        def solve(matrix: DLXMatrix, partial: list[Any]) -> None:
            if matrix.is_empty():
                solutions.append(list(partial))
                return

            col = matrix.choose_column()
            if col.size == 0:
                return  # Dead end.

            matrix.cover(col)
            row_node = col.down
            while row_node is not col:
                partial.append(row_node.row_id)

                # Cover all other columns in this row.
                j = row_node.right
                while j is not row_node:
                    matrix.cover(j.column)
                    j = j.right

                solve(matrix, partial)

                # Uncover in reverse.
                j = row_node.left
                while j is not row_node:
                    matrix.uncover(j.column)
                    j = j.left

                partial.pop()
                row_node = row_node.down

            matrix.uncover(col)

        solve(knuth_matrix, [])

        assert len(solutions) == 1
        assert sorted(solutions[0]) == ["B", "D", "F"]
