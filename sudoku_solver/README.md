# Sudoku Solver (Dancing Links)

A production-ready Sudoku solver in pure Python that uses Donald Knuth's Dancing Links (DLX) technique to implement Algorithm X for the exact cover problem. Given a standard 9x9 Sudoku puzzle, the solver reformulates it as an exact cover instance and finds all valid completions.

The solver is designed for correctness and clarity. The DLX engine is fully generic -- it knows nothing about Sudoku -- while all puzzle-specific logic lives in a separate constraint mapper and solution decoder. This separation means the same DLX solver can be reused for any exact cover problem with zero modification.


## Algorithm Explanation

### Exact Cover Problems

An exact cover problem asks: given a set of elements (the "universe") and a collection of subsets, can you choose subsets so that every element appears in exactly one chosen subset? This is an NP-complete problem in general, but many practical instances -- including Sudoku -- have enough structure for efficient search.

### How Sudoku Maps to Exact Cover

A 9x9 Sudoku puzzle translates into an exact cover matrix with **324 constraint columns** and up to **729 choice rows**.

The 324 columns encode four types of constraints, each contributing 81 columns:

| Constraint Type | Count | Meaning |
|----------------|-------|---------|
| Cell           | 81    | Each cell contains exactly one digit |
| Row-Digit      | 81    | Each row contains each digit 1-9 exactly once |
| Column-Digit   | 81    | Each column contains each digit 1-9 exactly once |
| Box-Digit      | 81    | Each 3x3 box contains each digit 1-9 exactly once |

The 729 rows represent every possible placement: "digit d in cell (r, c)" for all 9 digits across all 81 cells. Each row covers exactly 4 constraints (one from each type). Pre-filled cells reduce the matrix by contributing only their given digit's row instead of all nine possibilities.

### How Dancing Links Works

Dancing Links (DLX) is Knuth's technique for efficiently implementing the cover and uncover operations that Algorithm X requires. The matrix is stored as a network of nodes connected by circular doubly-linked lists -- one horizontal list per row, one vertical list per column, plus a horizontal list of column headers.

The key insight is that removing a node from a doubly-linked list is reversible if you keep the removed node's pointers intact:

- **Cover** a column: unlink its header from the header list, then for every row in that column, unlink all nodes in that row from their respective columns.
- **Uncover** a column: reverse the cover in exact LIFO order. Because the removed nodes still hold their original link pointers, they "dance" back into position.

Algorithm X then works recursively:

1. If no columns remain, the current selection of rows is an exact cover -- yield it.
2. Choose the column with the fewest remaining rows (the minimum remaining values heuristic, which minimizes branching).
3. Try each row in that column: cover it, recurse, then uncover to backtrack.

### Why DLX Is Efficient

- **O(1) link manipulation**: Cover and uncover operations modify a constant number of pointers per node, regardless of matrix size.
- **MRV heuristic**: Choosing the smallest column first prunes the search tree dramatically. For most Sudoku puzzles this reduces the search to near-zero backtracking.
- **Perfect reversibility**: The cover/uncover dance restores the matrix exactly, with no copying or allocation during backtracking.
- **Generator-based search**: Solutions are yielded one at a time, so memory usage stays bounded regardless of how many solutions exist.


## Installation

Requires Python 3.10 or later. No external dependencies.

```bash
cd /path/to/ai-leetcode
pip install -e .
```

This installs the `sudoku-solver` command and the `sudoku_solver` Python package.


## Usage

### Command-Line Interface

**Solve a puzzle from an argument:**

```bash
python -m sudoku_solver "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
```

**Solve puzzles from a file (one puzzle per line):**

```bash
python -m sudoku_solver -f puzzles.txt
```

**Pipe a puzzle through stdin:**

```bash
echo "003020600900305001001806400008102900700000008006708200002609500800203009005010300" | python -m sudoku_solver
```

**Count solutions instead of displaying them:**

```bash
python -m sudoku_solver --count "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
```

**Find all solutions:**

```bash
python -m sudoku_solver --all "000000000000000000000000000000000000000000000000000000000000000000000000000000000"
```

**Output formats:**

```bash
python -m sudoku_solver -o pretty "..."   # Box-separated grid (default)
python -m sudoku_solver -o compact "..."  # 81-character string
python -m sudoku_solver -o both "..."     # Both formats
```

**Other flags:**

```bash
python -m sudoku_solver -v "..."   # Verbose (debug logging)
python -m sudoku_solver -q "..."   # Quiet (solution only, no headers)
```

### Python API

```python
from sudoku_solver import SudokuConstraintMapper, DLXSolver, SolutionDecoder

# Define a puzzle as a 9x9 grid (0 = empty cell)
grid = [
    [0, 0, 3, 0, 2, 0, 6, 0, 0],
    [9, 0, 0, 3, 0, 5, 0, 0, 1],
    [0, 0, 1, 8, 0, 6, 4, 0, 0],
    [0, 0, 8, 1, 0, 2, 9, 0, 0],
    [7, 0, 0, 0, 0, 0, 0, 0, 8],
    [0, 0, 6, 7, 0, 8, 2, 0, 0],
    [0, 0, 2, 6, 0, 9, 5, 0, 0],
    [8, 0, 0, 2, 0, 3, 0, 0, 9],
    [0, 0, 5, 0, 1, 0, 3, 0, 0],
]

# Build the exact cover matrix
mapper = SudokuConstraintMapper()
matrix = mapper.build_matrix(grid)

# Solve
solver = DLXSolver(matrix)
solution = solver.solve_one()

if solution is not None:
    # Decode the solution back into a grid
    decoder = SolutionDecoder()
    solved_grid = decoder.decode(solution)

    # Display
    print(decoder.format_grid(solved_grid))
    # +---------+---------+---------+
    # | 4  8  3 | 9  2  1 | 6  5  7 |
    # | 9  6  7 | 3  4  5 | 8  2  1 |
    # | 2  5  1 | 8  7  6 | 4  9  3 |
    # +---------+---------+---------+
    # | ...

    # Or as a compact string
    print(decoder.format_grid_compact(solved_grid))
    # "483921657967345821251876493..."
```

**Counting solutions:**

```python
solver = DLXSolver(matrix, max_solutions=0)  # 0 = find all
count = solver.count_solutions(limit=2)      # Stop after 2 if you only care about uniqueness
```

**Iterating over all solutions:**

```python
solver = DLXSolver(matrix, max_solutions=0)
for solution in solver.solve():
    grid = decoder.decode(solution)
    # process each solution...
```

**Input validation:**

```python
from sudoku_solver.validator import SudokuValidator, InvalidPuzzleError

validator = SudokuValidator()
try:
    validator.validate_grid(grid)
except InvalidPuzzleError as e:
    print(e)  # All errors listed, one per line
```


## Architecture

```
sudoku_solver/
    __init__.py              Public API exports and package metadata
    __main__.py              Entry point for python -m sudoku_solver
    cli.py                   Argument parsing, input routing, output formatting
    validator.py             Input validation (structure, constraints, solvability)
    constraint_mapper.py     Sudoku grid -> exact cover matrix translation
    solution_decoder.py      DLX solution -> Sudoku grid reconstruction
    solver.py                Generic Algorithm X solver (domain-agnostic)
    dlx_matrix.py            Sparse matrix with cover/uncover operations
    dlx_node.py              DLXNode and ColumnHeader data structures
    metrics.py               Timing and search-node metrics collection
    tests/
        conftest.py          Shared fixtures and puzzle data
        test_dlx_node.py     Unit tests for node structures
        test_dlx_matrix.py   Unit tests for sparse matrix
        test_solver.py       Unit tests for Algorithm X solver
        test_constraint_mapper.py  Unit tests for constraint mapping
        test_solution_decoder.py   Unit tests for solution decoding
        test_validator.py    Unit tests for input validation
        test_integration.py  End-to-end solve tests with known puzzles
        test_edge_cases.py   Boundary conditions and error paths
        test_performance.py  Timing benchmarks across difficulty levels
```

**Key design principle**: The DLX engine (`dlx_node.py`, `dlx_matrix.py`, `solver.py`) is completely generic. It operates on abstract column names and row identifiers with no knowledge of Sudoku. All Sudoku-specific logic lives in `constraint_mapper.py` (encoding) and `solution_decoder.py` (decoding). This means the DLX solver can be reused for any exact cover problem -- N-queens, polyomino tiling, scheduling -- by writing a different mapper.


## Input Format

Puzzles are provided as 81-character strings where each character represents a cell, read left-to-right, top-to-bottom:

- Digits `1`-`9` are pre-filled (given) cells
- `0` or `.` represents an empty cell

Example -- the string:

```
003020600900305001001806400008102900700000008006708200002609500800203009005010300
```

Represents this grid:

```
. . 3 | . 2 . | 6 . .
9 . . | 3 . 5 | . . 1
. . 1 | 8 . 6 | 4 . .
------+-------+------
. . 8 | 1 . 2 | 9 . .
7 . . | . . . | . . 8
. . 6 | 7 . 8 | 2 . .
------+-------+------
. . 2 | 6 . 9 | 5 . .
8 . . | 2 . 3 | . . 9
. . 5 | . 1 . | 3 . .
```

When reading from a file (`-f`), each line is one puzzle. Blank lines and lines starting with `#` are ignored.


## Performance

Typical solve times on modern hardware (single-threaded, pure Python):

| Puzzle Difficulty   | Clues | Expected Time |
|--------------------|-------|---------------|
| Easy (30+ clues)   | 30+   | 2-5 ms        |
| Medium (25-30)     | 25-30 | 5-15 ms       |
| Hard (20-25)       | 20-25 | 10-40 ms      |
| Minimal (17 clues) | 17    | 20-100 ms     |
| "World's Hardest"  | 21    | 50-200 ms     |

The solver uses a generator-based architecture, yielding solutions one at a time. Memory usage stays bounded regardless of how many solutions a puzzle has.


## Testing

Run the full test suite (174 tests):

```bash
python -m pytest sudoku_solver/tests/
```

Run without slow performance benchmarks:

```bash
python -m pytest sudoku_solver/tests/ -m "not slow"
```

Run only performance benchmarks:

```bash
python -m pytest sudoku_solver/tests/ -m slow
```

Run with verbose output:

```bash
python -m pytest sudoku_solver/tests/ -v
```

The test suite covers:

- **Unit tests**: Every module (nodes, matrix, solver, constraint mapper, solution decoder, validator)
- **Integration tests**: End-to-end solves across four difficulty levels with solution verification
- **Edge cases**: Empty grids, solved grids, unsolvable puzzles, multiple solutions, single empty cell
- **Performance benchmarks**: Timing thresholds from easy through "AI Escargot" difficulty


## License

MIT
