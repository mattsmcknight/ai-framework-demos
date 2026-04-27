"""Unit tests for DLXNode and ColumnHeader data structures."""

from __future__ import annotations

import pytest

from sudoku_solver.dlx_node import ColumnHeader, DLXNode


# ---------------------------------------------------------------------------
# DLXNode construction
# ---------------------------------------------------------------------------


class TestDLXNodeConstruction:
    """Verify that a freshly created DLXNode is properly self-linked."""

    def test_all_links_point_to_self(self) -> None:
        """A new node should have left, right, up, down all pointing to itself."""
        node = DLXNode()
        assert node.left is node
        assert node.right is node
        assert node.up is node
        assert node.down is node

    def test_column_defaults_to_none(self) -> None:
        """Column should be None when not provided."""
        node = DLXNode()
        assert node.column is None

    def test_row_id_defaults_to_none(self) -> None:
        """row_id should be None when not provided."""
        node = DLXNode()
        assert node.row_id is None

    def test_column_set_on_construction(self) -> None:
        """Column should be set when provided at construction time."""
        header = ColumnHeader("test")
        node = DLXNode(column=header, row_id=42)
        assert node.column is header
        assert node.row_id == 42


# ---------------------------------------------------------------------------
# ColumnHeader construction
# ---------------------------------------------------------------------------


class TestColumnHeaderConstruction:
    """Verify ColumnHeader initialization and defaults."""

    def test_name_is_set(self) -> None:
        """ColumnHeader should store the name passed to it."""
        header = ColumnHeader("cell_0_0")
        assert header.name == "cell_0_0"

    def test_size_starts_at_zero(self) -> None:
        """ColumnHeader size should initialize to zero."""
        header = ColumnHeader("col")
        assert header.size == 0

    def test_column_references_self(self) -> None:
        """ColumnHeader.column should reference itself (DLX convention)."""
        header = ColumnHeader("col")
        assert header.column is header

    def test_self_links(self) -> None:
        """ColumnHeader inherits self-linking from DLXNode."""
        header = ColumnHeader("col")
        assert header.left is header
        assert header.right is header
        assert header.up is header
        assert header.down is header

    def test_is_subclass_of_dlx_node(self) -> None:
        """ColumnHeader must be a subclass of DLXNode."""
        header = ColumnHeader("col")
        assert isinstance(header, DLXNode)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    """Verify developer-friendly string representations."""

    def test_dlx_node_repr_with_row_id(self) -> None:
        """DLXNode repr should show row_id when set."""
        node = DLXNode(row_id=7)
        assert repr(node) == "DLXNode(row_id=7)"

    def test_dlx_node_repr_without_row_id(self) -> None:
        """DLXNode repr should fall back to hex id when row_id is None."""
        node = DLXNode()
        r = repr(node)
        assert r.startswith("DLXNode(id=0x")
        assert r.endswith(")")

    def test_column_header_repr(self) -> None:
        """ColumnHeader repr should show name and size."""
        header = ColumnHeader("row_3_digit_7")
        header.size = 5
        assert repr(header) == "ColumnHeader(name='row_3_digit_7', size=5)"


# ---------------------------------------------------------------------------
# __slots__
# ---------------------------------------------------------------------------


class TestSlots:
    """Verify that both classes use __slots__ for memory efficiency."""

    def test_dlx_node_has_slots(self) -> None:
        """DLXNode should define __slots__ and reject arbitrary attributes."""
        node = DLXNode()
        assert hasattr(DLXNode, "__slots__")
        with pytest.raises(AttributeError):
            node.arbitrary_attribute = "should fail"  # type: ignore[attr-defined]

    def test_column_header_has_slots(self) -> None:
        """ColumnHeader should define __slots__ and reject arbitrary attributes."""
        header = ColumnHeader("col")
        assert hasattr(ColumnHeader, "__slots__")
        with pytest.raises(AttributeError):
            header.arbitrary_attribute = "should fail"  # type: ignore[attr-defined]

    def test_dlx_node_no_dict(self) -> None:
        """DLXNode instances should not have __dict__."""
        node = DLXNode()
        assert not hasattr(node, "__dict__")

    def test_column_header_no_dict(self) -> None:
        """ColumnHeader instances should not have __dict__."""
        header = ColumnHeader("col")
        assert not hasattr(header, "__dict__")
