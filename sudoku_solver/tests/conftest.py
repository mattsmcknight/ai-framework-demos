"""Shared test fixtures for Sudoku solver test suite.

Provides known puzzles and their solutions as pytest fixtures. Puzzles are
stored as compact 81-character strings and converted to 9x9 grids.

Puzzle sources:
- easy_puzzle: Classic "first Sudoku example" from Wikipedia
- medium_puzzle: Moderate difficulty, 25 clues
- hard_puzzle: High difficulty, 21 clues
- minimal_puzzle: 17-clue puzzle (proven minimum for unique solution)
- empty_grid: All zeros (0 clues, many solutions)
- solved_grid: Fully filled valid grid (0 empty cells)
- invalid grids: Intentionally broken for validation testing
- unsolvable_puzzle: Structurally valid but logically impossible
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _string_to_grid(s: str) -> list[list[int]]:
    """Convert an 81-character digit string to a 9x9 grid.

    Args:
        s: An 81-character string of digits 0-9, row-major order.
           '0' represents an empty cell.

    Returns:
        A 9x9 list of lists of integers.
    """
    assert len(s) == 81, f"Expected 81 characters, got {len(s)}"
    return [
        [int(s[row * 9 + col]) for col in range(9)]
        for row in range(9)
    ]


# ---------------------------------------------------------------------------
# Puzzle data: compact 81-character strings
# ---------------------------------------------------------------------------

# Easy: 30 clues. Classic Wikipedia example.
_EASY_PUZZLE_STR = (
    "530070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)
_EASY_SOLUTION_STR = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)

# Medium: 25 clues.
_MEDIUM_PUZZLE_STR = (
    "000260701"
    "680070090"
    "190004500"
    "820100040"
    "004602900"
    "050003028"
    "009300074"
    "040050036"
    "703018000"
)
_MEDIUM_SOLUTION_STR = (
    "435269781"
    "682571493"
    "197834562"
    "826195347"
    "374682915"
    "951743628"
    "519326874"
    "248957136"
    "763418259"
)

# Hard: 21 clues.
_HARD_PUZZLE_STR = (
    "000000010"
    "400000000"
    "020000000"
    "000050407"
    "008000300"
    "001090000"
    "300400200"
    "050100000"
    "000806000"
)
_HARD_SOLUTION_STR = (
    "693784512"
    "487512936"
    "125963874"
    "932651487"
    "568247391"
    "741398625"
    "319475268"
    "856129743"
    "274836159"
)

# Minimal: 17 clues (proven minimum for unique solution).
# Source: Gordon Royle's catalog of 17-clue puzzles.
_MINIMAL_PUZZLE_STR = (
    "000000000"
    "000003085"
    "001020000"
    "000507000"
    "004000100"
    "090000000"
    "500000073"
    "002010000"
    "000040009"
)
_MINIMAL_SOLUTION_STR = (
    "987654321"
    "246173985"
    "351928746"
    "128537694"
    "634892157"
    "795461832"
    "519286473"
    "472319568"
    "863745219"
)

# Solved grid: the easy puzzle's solution, also used as a standalone fixture.
_SOLVED_GRID_STR = _EASY_SOLUTION_STR

# Empty grid: all zeros.
_EMPTY_GRID_STR = "0" * 81

# Invalid grids: each has exactly one violation type.

# Duplicate in row 0: two 5s at (0,0) and (0,1).
_INVALID_ROW_STR = (
    "550070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)

# Duplicate in column 0: two 5s at (0,0) and (1,0).
_INVALID_COL_STR = (
    "530070000"
    "500195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)

# Duplicate in box 0: two 9s at (1,0) and (2,2). Place a 9 at (1,0)
# which conflicts with the 9 already at (2,2) in the original easy puzzle.
_INVALID_BOX_STR = (
    "530070000"
    "900195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)

# Unsolvable: structurally valid but logically impossible.
# Row 0 has digits 1-8 placed, with 9 forced into (0,8). But column 8 already
# has a 9 at (4,8), making it impossible.
_UNSOLVABLE_PUZZLE_STR = (
    "123456780"
    "000000000"
    "000000000"
    "000000000"
    "000000009"
    "000000000"
    "000000000"
    "000000000"
    "000000000"
)


# ---------------------------------------------------------------------------
# Fixtures: puzzles as 9x9 grids
# ---------------------------------------------------------------------------


@pytest.fixture()
def easy_puzzle() -> list[list[int]]:
    """Easy puzzle with 30 clues."""
    return _string_to_grid(_EASY_PUZZLE_STR)


@pytest.fixture()
def easy_solution() -> list[list[int]]:
    """Known solution for the easy puzzle."""
    return _string_to_grid(_EASY_SOLUTION_STR)


@pytest.fixture()
def medium_puzzle() -> list[list[int]]:
    """Medium puzzle with 25 clues."""
    return _string_to_grid(_MEDIUM_PUZZLE_STR)


@pytest.fixture()
def medium_solution() -> list[list[int]]:
    """Known solution for the medium puzzle."""
    return _string_to_grid(_MEDIUM_SOLUTION_STR)


@pytest.fixture()
def hard_puzzle() -> list[list[int]]:
    """Hard puzzle with 21 clues."""
    return _string_to_grid(_HARD_PUZZLE_STR)


@pytest.fixture()
def hard_solution() -> list[list[int]]:
    """Known solution for the hard puzzle."""
    return _string_to_grid(_HARD_SOLUTION_STR)


@pytest.fixture()
def minimal_puzzle() -> list[list[int]]:
    """Minimal 17-clue puzzle."""
    return _string_to_grid(_MINIMAL_PUZZLE_STR)


@pytest.fixture()
def minimal_solution() -> list[list[int]]:
    """Known solution for the minimal puzzle."""
    return _string_to_grid(_MINIMAL_SOLUTION_STR)


@pytest.fixture()
def empty_grid() -> list[list[int]]:
    """All-zeros grid (no clues)."""
    return _string_to_grid(_EMPTY_GRID_STR)


@pytest.fixture()
def solved_grid() -> list[list[int]]:
    """Fully filled valid grid (0 empty cells)."""
    return _string_to_grid(_SOLVED_GRID_STR)


@pytest.fixture()
def invalid_grid_duplicate_row() -> list[list[int]]:
    """Grid with a duplicate digit in row 0."""
    return _string_to_grid(_INVALID_ROW_STR)


@pytest.fixture()
def invalid_grid_duplicate_col() -> list[list[int]]:
    """Grid with a duplicate digit in column 0."""
    return _string_to_grid(_INVALID_COL_STR)


@pytest.fixture()
def invalid_grid_duplicate_box() -> list[list[int]]:
    """Grid with a duplicate digit in box 0."""
    return _string_to_grid(_INVALID_BOX_STR)


@pytest.fixture()
def unsolvable_puzzle() -> list[list[int]]:
    """Structurally valid but logically unsolvable puzzle."""
    return _string_to_grid(_UNSOLVABLE_PUZZLE_STR)
