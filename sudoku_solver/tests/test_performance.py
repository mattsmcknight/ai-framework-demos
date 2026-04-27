"""Performance benchmarks for the DLX Sudoku solver.

Each benchmark solves a puzzle multiple times and uses the median elapsed
time for the assertion.  Using the median rather than the minimum or mean
avoids spurious failures caused by GC pauses, OS scheduling jitter, or
background processes -- while still detecting genuine regressions.

All tests in this module are marked ``@pytest.mark.slow`` so that they
can be excluded from fast feedback loops (``pytest -m "not slow"``).

Performance targets (with 5x generous margins over expected performance):

- Easy puzzle:        < 50 ms
- Medium puzzle:      < 100 ms
- Hard puzzle:        < 200 ms
- Minimal 17-clue:   < 500 ms
- AI Escargot class:  < 2000 ms
- 50-puzzle throughput: < 10 s total
"""

from __future__ import annotations

import statistics
import time

import pytest

from sudoku_solver.constraint_mapper import SudokuConstraintMapper
from sudoku_solver.solution_decoder import SolutionDecoder
from sudoku_solver.solver import DLXSolver

# ---------------------------------------------------------------------------
# Puzzle data -- organized by difficulty
# ---------------------------------------------------------------------------

# Easy puzzles (30+ clues, single forced propagation chains)

EASY_PUZZLE_1: list[list[int]] = [
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

EASY_PUZZLE_2: list[list[int]] = [
    [0, 0, 0, 2, 6, 0, 7, 0, 1],
    [6, 8, 0, 0, 7, 0, 0, 9, 0],
    [1, 9, 0, 0, 0, 4, 5, 0, 0],
    [8, 2, 0, 1, 0, 0, 0, 4, 0],
    [0, 0, 4, 6, 0, 2, 9, 0, 0],
    [0, 5, 0, 0, 0, 3, 0, 2, 8],
    [0, 0, 9, 3, 0, 0, 0, 7, 4],
    [0, 4, 0, 0, 5, 0, 0, 3, 6],
    [7, 0, 3, 0, 1, 8, 0, 0, 0],
]

EASY_PUZZLE_3: list[list[int]] = [
    [1, 0, 0, 4, 8, 9, 0, 0, 6],
    [7, 3, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 1, 2, 9, 5],
    [0, 0, 7, 1, 2, 0, 6, 0, 0],
    [5, 0, 0, 7, 0, 3, 0, 0, 8],
    [0, 0, 6, 0, 9, 5, 7, 0, 0],
    [9, 1, 4, 6, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 3, 7],
    [8, 0, 0, 5, 1, 2, 0, 0, 4],
]

# Medium puzzles (25-30 clues, some trial-and-error needed)

MEDIUM_PUZZLE: list[list[int]] = [
    [0, 0, 0, 6, 0, 0, 4, 0, 0],
    [7, 0, 0, 0, 0, 3, 6, 0, 0],
    [0, 0, 0, 0, 9, 1, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 1, 8, 0, 0, 0, 3],
    [0, 0, 0, 3, 0, 6, 0, 4, 5],
    [0, 4, 0, 2, 0, 0, 0, 6, 0],
    [9, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 1, 0, 0],
]

# Hard puzzles (fewer clues, significant backtracking required)

HARD_PUZZLE: list[list[int]] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 8, 5],
    [0, 0, 1, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 7, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 1, 0, 0],
    [0, 9, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 0, 7, 3],
    [0, 0, 2, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 9],
]

# Minimal 17-clue puzzle (proven minimum for unique solution)

MINIMAL_17_CLUE: list[list[int]] = [
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 4, 0, 7],
    [0, 0, 8, 0, 0, 0, 3, 0, 0],
    [0, 0, 1, 0, 9, 0, 0, 0, 0],
    [3, 0, 0, 4, 0, 0, 2, 0, 0],
    [0, 5, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 6, 0, 0, 0],
]

# "AI Escargot" -- widely cited as one of the hardest 9x9 Sudoku puzzles.
# Created by Arto Inkala in 2006.

AI_ESCARGOT: list[list[int]] = [
    [1, 0, 0, 0, 0, 7, 0, 9, 0],
    [0, 3, 0, 0, 2, 0, 0, 0, 8],
    [0, 0, 9, 6, 0, 0, 5, 0, 0],
    [0, 0, 5, 3, 0, 0, 9, 0, 0],
    [0, 1, 0, 0, 8, 0, 0, 0, 2],
    [6, 0, 0, 0, 0, 4, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 7],
    [0, 0, 7, 0, 0, 0, 3, 0, 0],
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAPPER = SudokuConstraintMapper()
_DECODER = SolutionDecoder()
_WARMUP_RUNS = 2
_BENCHMARK_RUNS = 5


def _solve_puzzle(puzzle: list[list[int]]) -> list[list[int]]:
    """Solve a single puzzle through the full pipeline and return the grid.

    This exercises the complete path: constraint mapping, DLX solve,
    solution decoding.  The returned grid is validated to be non-None
    to catch regressions where a puzzle becomes unsolvable due to code
    changes.
    """
    matrix = _MAPPER.build_matrix(puzzle)
    solver = DLXSolver(matrix)
    row_ids = solver.solve_one()
    assert row_ids is not None, "Solver returned None -- puzzle has no solution"
    return _DECODER.decode(row_ids)


def _time_solve(puzzle: list[list[int]]) -> float:
    """Time a single puzzle solve and return elapsed milliseconds.

    Uses ``time.perf_counter()`` for high-resolution wall-clock timing.
    """
    start = time.perf_counter()
    _solve_puzzle(puzzle)
    elapsed_s = time.perf_counter() - start
    return elapsed_s * 1000.0


def _benchmark_puzzle(puzzle: list[list[int]]) -> float:
    """Run warmup + benchmark iterations and return the median time in ms.

    Warmup runs are discarded.  The median of the benchmark runs is
    returned for stability.
    """
    # Warmup: get JIT-like effects and page faults out of the way.
    for _ in range(_WARMUP_RUNS):
        _solve_puzzle(puzzle)

    # Timed runs.
    times: list[float] = []
    for _ in range(_BENCHMARK_RUNS):
        times.append(_time_solve(puzzle))

    return statistics.median(times)


def _is_valid_solution(grid: list[list[int]]) -> bool:
    """Quick structural check that a 9x9 grid is a valid Sudoku solution."""
    for row in grid:
        if sorted(row) != list(range(1, 10)):
            return False
    for col in range(9):
        column_vals = [grid[r][col] for r in range(9)]
        if sorted(column_vals) != list(range(1, 10)):
            return False
    for box_row in range(3):
        for box_col in range(3):
            box_vals = [
                grid[r][c]
                for r in range(box_row * 3, box_row * 3 + 3)
                for c in range(box_col * 3, box_col * 3 + 3)
            ]
            if sorted(box_vals) != list(range(1, 10)):
                return False
    return True


# ---------------------------------------------------------------------------
# Timing benchmarks -- parametrized by difficulty
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEasyPuzzleTiming:
    """Easy puzzles should solve well under 50 ms (median of 5 runs)."""

    @pytest.mark.parametrize(
        "puzzle, label",
        [
            (EASY_PUZZLE_1, "easy_1"),
            (EASY_PUZZLE_2, "easy_2"),
            (EASY_PUZZLE_3, "easy_3"),
        ],
        ids=["easy_1", "easy_2", "easy_3"],
    )
    def test_easy_under_50ms(
        self, puzzle: list[list[int]], label: str
    ) -> None:
        """Each easy puzzle solves in < 50 ms (median of 5 runs)."""
        median_ms = _benchmark_puzzle(puzzle)
        assert median_ms < 50.0, (
            f"Easy puzzle {label}: median {median_ms:.2f} ms exceeds 50 ms"
        )


@pytest.mark.slow
class TestMediumPuzzleTiming:
    """Medium puzzles should solve in < 100 ms (median of 5 runs)."""

    def test_medium_under_100ms(self) -> None:
        """Medium puzzle solves in < 100 ms."""
        median_ms = _benchmark_puzzle(MEDIUM_PUZZLE)
        assert median_ms < 100.0, (
            f"Medium puzzle: median {median_ms:.2f} ms exceeds 100 ms"
        )


@pytest.mark.slow
class TestHardPuzzleTiming:
    """Hard puzzles should solve in < 200 ms (median of 5 runs)."""

    def test_hard_under_200ms(self) -> None:
        """Hard puzzle solves in < 200 ms."""
        median_ms = _benchmark_puzzle(HARD_PUZZLE)
        assert median_ms < 200.0, (
            f"Hard puzzle: median {median_ms:.2f} ms exceeds 200 ms"
        )


@pytest.mark.slow
class TestMinimal17ClueTiming:
    """Minimal 17-clue puzzles should solve in < 500 ms (median of 5 runs)."""

    def test_17_clue_under_500ms(self) -> None:
        """Minimal 17-clue puzzle solves in < 500 ms."""
        median_ms = _benchmark_puzzle(MINIMAL_17_CLUE)
        assert median_ms < 500.0, (
            f"17-clue puzzle: median {median_ms:.2f} ms exceeds 500 ms"
        )


# ---------------------------------------------------------------------------
# Known hard puzzle: AI Escargot
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAIEscargot:
    """AI Escargot (Arto Inkala, 2006) -- one of the hardest known puzzles."""

    def test_ai_escargot_solves_correctly(self) -> None:
        """AI Escargot produces a valid Sudoku solution."""
        grid = _solve_puzzle(AI_ESCARGOT)
        assert _is_valid_solution(grid), "AI Escargot solution is not valid"

    def test_ai_escargot_under_2_seconds(self) -> None:
        """AI Escargot solves in < 2 seconds (median of 5 runs)."""
        median_ms = _benchmark_puzzle(AI_ESCARGOT)
        assert median_ms < 2000.0, (
            f"AI Escargot: median {median_ms:.2f} ms exceeds 2000 ms"
        )


# ---------------------------------------------------------------------------
# Throughput test: batch solving
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestThroughput:
    """Batch solve throughput: 50 easy puzzles sequentially."""

    def test_50_easy_puzzles_under_10_seconds(self) -> None:
        """Solving 50 easy puzzles sequentially completes in < 10 seconds.

        Uses all three easy puzzles in a round-robin pattern to avoid
        identical-puzzle caching effects (though this pure-Python solver
        has no internal cache).
        """
        puzzles = [EASY_PUZZLE_1, EASY_PUZZLE_2, EASY_PUZZLE_3]
        total_puzzles = 50

        start = time.perf_counter()
        for i in range(total_puzzles):
            puzzle = puzzles[i % len(puzzles)]
            grid = _solve_puzzle(puzzle)
            assert _is_valid_solution(grid), f"Puzzle {i} produced invalid solution"
        elapsed_s = time.perf_counter() - start

        assert elapsed_s < 10.0, (
            f"50 puzzles took {elapsed_s:.2f}s, exceeds 10s budget"
        )


# ---------------------------------------------------------------------------
# MetricsCollector integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMetricsCollectorIntegration:
    """Verify MetricsCollector populates fields when wrapping a solve."""

    def test_metrics_populated_after_solve(self) -> None:
        """MetricsCollector records timing and dimensional data."""
        metrics_mod = pytest.importorskip(
            "sudoku_solver.metrics",
            reason="metrics module not available",
        )
        MetricsCollector = metrics_mod.MetricsCollector
        SolveMetrics = metrics_mod.SolveMetrics

        mapper = SudokuConstraintMapper()
        matrix = mapper.build_matrix(EASY_PUZZLE_1)

        collector = MetricsCollector()
        collector.matrix_columns = 324
        collector.matrix_rows = matrix.row_count if hasattr(matrix, "row_count") else 0
        collector.clues_given = sum(
            1 for row in EASY_PUZZLE_1 for v in row if v != 0
        )

        with collector:
            solver = DLXSolver(matrix)
            result = solver.solve_one()
            assert result is not None
            collector.solutions_found = 1

        metrics = collector.to_metrics()

        # Verify the returned object is a SolveMetrics instance.
        assert isinstance(metrics, SolveMetrics)

        # Timing must be positive (we just solved a puzzle).
        assert metrics.solve_time_ms > 0.0, (
            "solve_time_ms should be positive after a solve"
        )

        # Dimensional fields should reflect what we set.
        assert metrics.matrix_columns == 324
        assert metrics.solutions_found == 1
        assert metrics.clues_given == sum(
            1 for row in EASY_PUZZLE_1 for v in row if v != 0
        )

    def test_metrics_str_is_human_readable(self) -> None:
        """SolveMetrics.__str__ produces a non-empty, informative string."""
        metrics_mod = pytest.importorskip(
            "sudoku_solver.metrics",
            reason="metrics module not available",
        )
        SolveMetrics = metrics_mod.SolveMetrics

        metrics = SolveMetrics(
            solve_time_ms=12.345,
            nodes_explored=1000,
            solutions_found=1,
            matrix_columns=324,
            matrix_rows=200,
            clues_given=30,
        )
        text = str(metrics)

        assert "12.345" in text
        assert "1000" in text
        assert "324" in text
