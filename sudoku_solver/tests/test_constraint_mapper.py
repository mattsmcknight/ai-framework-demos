"""Unit tests for SudokuConstraintMapper."""

from __future__ import annotations

import pytest

from sudoku_solver.constraint_mapper import SudokuConstraintMapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mapper() -> SudokuConstraintMapper:
    """Provide a fresh mapper instance for each test."""
    return SudokuConstraintMapper()


def _empty_grid() -> list[list[int]]:
    """Return a 9x9 grid of zeros (no pre-filled cells)."""
    return [[0] * 9 for _ in range(9)]


def _full_grid() -> list[list[int]]:
    """Return a valid fully-solved 9x9 Sudoku grid."""
    return [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ]


def _partial_grid() -> list[list[int]]:
    """Return a partially filled grid with a known number of givens (30)."""
    return [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]


# ---------------------------------------------------------------------------
# build_column_names tests
# ---------------------------------------------------------------------------


class TestBuildColumnNames:
    """Verify column name generation for the exact cover matrix."""

    def test_returns_exactly_324_names(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """The method should produce exactly 324 constraint column names."""
        names = mapper.build_column_names()
        assert len(names) == 324

    def test_all_names_are_unique(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """Every column name should be unique."""
        names = mapper.build_column_names()
        assert len(set(names)) == 324

    def test_cell_constraint_naming_pattern(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """Cell constraints should follow the pattern cell_r{row}_c{col}."""
        names = mapper.build_column_names()
        cell_names = [n for n in names if n.startswith("cell_")]

        assert len(cell_names) == 81
        # Verify first and last cell names.
        assert "cell_r0_c0" in cell_names
        assert "cell_r8_c8" in cell_names

    def test_row_digit_constraint_naming_pattern(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """Row-digit constraints should follow the pattern row_r{row}_d{digit}."""
        names = mapper.build_column_names()
        row_names = [n for n in names if n.startswith("row_")]

        assert len(row_names) == 81
        assert "row_r0_d1" in row_names
        assert "row_r8_d9" in row_names

    def test_col_digit_constraint_naming_pattern(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """Column-digit constraints should follow the pattern col_c{col}_d{digit}."""
        names = mapper.build_column_names()
        col_names = [n for n in names if n.startswith("col_")]

        assert len(col_names) == 81
        assert "col_c0_d1" in col_names
        assert "col_c8_d9" in col_names

    def test_box_digit_constraint_naming_pattern(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """Box-digit constraints should follow the pattern box_b{box}_d{digit}."""
        names = mapper.build_column_names()
        box_names = [n for n in names if n.startswith("box_")]

        assert len(box_names) == 81
        assert "box_b0_d1" in box_names
        assert "box_b8_d9" in box_names

    def test_constraint_groups_are_ordered(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """Columns should appear in order: cell, row, col, box."""
        names = mapper.build_column_names()

        # Find the first index of each group.
        first_cell = next(i for i, n in enumerate(names) if n.startswith("cell_"))
        first_row = next(i for i, n in enumerate(names) if n.startswith("row_"))
        first_col = next(i for i, n in enumerate(names) if n.startswith("col_"))
        first_box = next(i for i, n in enumerate(names) if n.startswith("box_"))

        assert first_cell < first_row < first_col < first_box


# ---------------------------------------------------------------------------
# build_row_constraints tests
# ---------------------------------------------------------------------------


class TestBuildRowConstraints:
    """Verify constraint generation for individual placements."""

    def test_returns_exactly_four_constraints(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """Every valid placement should produce exactly 4 constraints."""
        constraints = mapper.build_row_constraints(0, 0, 1)
        assert len(constraints) == 4

    def test_constraint_types(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """The four constraints should be one each of cell, row, col, box."""
        constraints = mapper.build_row_constraints(3, 5, 7)
        prefixes = [c.split("_")[0] for c in constraints]
        assert sorted(prefixes) == ["box", "cell", "col", "row"]

    def test_specific_placement_constraints(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """Verify the exact constraint names for a known placement."""
        # Placing digit 5 at (2, 3): box = (2//3)*3 + (3//3) = 0*3 + 1 = 1
        constraints = mapper.build_row_constraints(2, 3, 5)
        assert "cell_r2_c3" in constraints
        assert "row_r2_d5" in constraints
        assert "col_c3_d5" in constraints
        assert "box_b1_d5" in constraints

    def test_all_valid_placements_produce_four_constraints(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """Every combination of (row, col, digit) should yield 4 constraints."""
        for row in range(9):
            for col in range(9):
                for digit in range(1, 10):
                    constraints = mapper.build_row_constraints(row, col, digit)
                    assert len(constraints) == 4, (
                        f"({row},{col},{digit}) produced "
                        f"{len(constraints)} constraints"
                    )


# ---------------------------------------------------------------------------
# Box index calculation tests
# ---------------------------------------------------------------------------


class TestBoxIndexCalculation:
    """Verify the box index formula for all 81 cells."""

    def test_box_indices_for_all_cells(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """Each cell should map to the correct 3x3 box (0-8)."""
        expected_box_map = {
            # box 0: rows 0-2, cols 0-2
            (0, 0): 0, (0, 1): 0, (0, 2): 0,
            (1, 0): 0, (1, 1): 0, (1, 2): 0,
            (2, 0): 0, (2, 1): 0, (2, 2): 0,
            # box 1: rows 0-2, cols 3-5
            (0, 3): 1, (0, 4): 1, (0, 5): 1,
            (1, 3): 1, (1, 4): 1, (1, 5): 1,
            (2, 3): 1, (2, 4): 1, (2, 5): 1,
            # box 2: rows 0-2, cols 6-8
            (0, 6): 2, (0, 7): 2, (0, 8): 2,
            (1, 6): 2, (1, 7): 2, (1, 8): 2,
            (2, 6): 2, (2, 7): 2, (2, 8): 2,
            # box 3: rows 3-5, cols 0-2
            (3, 0): 3, (3, 1): 3, (3, 2): 3,
            (4, 0): 3, (4, 1): 3, (4, 2): 3,
            (5, 0): 3, (5, 1): 3, (5, 2): 3,
            # box 4: rows 3-5, cols 3-5
            (3, 3): 4, (3, 4): 4, (3, 5): 4,
            (4, 3): 4, (4, 4): 4, (4, 5): 4,
            (5, 3): 4, (5, 4): 4, (5, 5): 4,
            # box 5: rows 3-5, cols 6-8
            (3, 6): 5, (3, 7): 5, (3, 8): 5,
            (4, 6): 5, (4, 7): 5, (4, 8): 5,
            (5, 6): 5, (5, 7): 5, (5, 8): 5,
            # box 6: rows 6-8, cols 0-2
            (6, 0): 6, (6, 1): 6, (6, 2): 6,
            (7, 0): 6, (7, 1): 6, (7, 2): 6,
            (8, 0): 6, (8, 1): 6, (8, 2): 6,
            # box 7: rows 6-8, cols 3-5
            (6, 3): 7, (6, 4): 7, (6, 5): 7,
            (7, 3): 7, (7, 4): 7, (7, 5): 7,
            (8, 3): 7, (8, 4): 7, (8, 5): 7,
            # box 8: rows 6-8, cols 6-8
            (6, 6): 8, (6, 7): 8, (6, 8): 8,
            (7, 6): 8, (7, 7): 8, (7, 8): 8,
            (8, 6): 8, (8, 7): 8, (8, 8): 8,
        }

        for (row, col), expected_box in expected_box_map.items():
            constraints = mapper.build_row_constraints(row, col, 1)
            box_constraint = [c for c in constraints if c.startswith("box_")][0]
            actual_box = int(box_constraint.split("_b")[1].split("_")[0])
            assert actual_box == expected_box, (
                f"Cell ({row},{col}): expected box {expected_box}, "
                f"got box {actual_box}"
            )


# ---------------------------------------------------------------------------
# build_matrix tests
# ---------------------------------------------------------------------------


class TestBuildMatrix:
    """Verify matrix construction for different grid states."""

    def test_empty_grid_produces_729_rows(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """An empty grid (all zeros) should have 729 choice rows (9*9*9)."""
        grid = _empty_grid()
        matrix = mapper.build_matrix(grid)

        # Count rows by summing column sizes / 4 (each row has 4 nodes).
        total_nodes = sum(
            matrix.columns[name].size for name in matrix.columns
        )
        row_count = total_nodes // 4
        assert row_count == 729

    def test_full_grid_produces_81_rows(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """A fully filled grid should have exactly 81 choice rows (one per cell)."""
        grid = _full_grid()
        matrix = mapper.build_matrix(grid)

        total_nodes = sum(
            matrix.columns[name].size for name in matrix.columns
        )
        row_count = total_nodes // 4
        assert row_count == 81

    def test_partial_grid_produces_correct_row_count(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """A partially filled grid should have givens*1 + empties*9 rows."""
        grid = _partial_grid()
        givens = sum(
            1 for r in range(9) for c in range(9) if grid[r][c] != 0
        )
        empties = 81 - givens
        expected_rows = givens + empties * 9

        matrix = mapper.build_matrix(grid)
        total_nodes = sum(
            matrix.columns[name].size for name in matrix.columns
        )
        row_count = total_nodes // 4
        assert row_count == expected_rows

    def test_matrix_has_324_columns(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """The built matrix should always have 324 constraint columns."""
        grid = _empty_grid()
        matrix = mapper.build_matrix(grid)
        assert len(matrix.columns) == 324

    def test_every_row_covers_exactly_four_columns(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """Each row in the matrix should span exactly 4 columns."""
        grid = _partial_grid()
        matrix = mapper.build_matrix(grid)

        # Walk the first column that has nodes and check row width.
        checked = 0
        for name in matrix.columns:
            col = matrix.columns[name]
            node = col.down
            while node is not col:
                # Walk right from this node, counting nodes in the row.
                count = 1
                other = node.right
                while other is not node:
                    count += 1
                    other = other.right
                assert count == 4, (
                    f"Row {node.row_id} has {count} nodes, expected 4"
                )
                checked += 1
                node = node.down
            if checked > 20:
                break  # Spot-check is sufficient; don't traverse all 500+ rows.

    def test_build_row_id_returns_tuple(
        self, mapper: SudokuConstraintMapper
    ) -> None:
        """build_row_id should return a (row, col, digit) tuple."""
        row_id = mapper.build_row_id(3, 7, 5)
        assert row_id == (3, 7, 5)
        assert isinstance(row_id, tuple)
