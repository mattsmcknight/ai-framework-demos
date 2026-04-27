"""Logging configuration and performance metrics for the Sudoku solver.

This module provides two concerns:

1. **Metrics collection**: A :class:`MetricsCollector` context manager that
   times solver execution and counts explored search nodes, producing a
   :class:`SolveMetrics` summary.

2. **Logging configuration**: A :func:`configure_logging` helper that sets up
   the ``sudoku_solver`` package logger with appropriate verbosity levels.

Both concerns use only the standard library (``time``, ``logging``,
``dataclasses``) -- no external dependencies.

Typical usage::

    from sudoku_solver.metrics import MetricsCollector, configure_logging

    configure_logging(verbose=True)

    collector = MetricsCollector()
    with collector:
        collector.record_node()
        # ... solver work ...
    print(collector.to_metrics())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

_PACKAGE_LOGGER_NAME = "sudoku_solver"


@dataclass(frozen=True, slots=True)
class SolveMetrics:
    """Immutable summary of a single solve run.

    All fields are populated by :meth:`MetricsCollector.to_metrics` after the
    collector's context manager exits (or after :meth:`stop` is called
    explicitly).

    Attributes:
        solve_time_ms: Wall-clock time in milliseconds.
        nodes_explored: Number of search-tree nodes visited.
        solutions_found: Number of exact-cover solutions found.
        matrix_columns: Number of constraint columns in the DLX matrix.
        matrix_rows: Number of choice rows in the DLX matrix.
        clues_given: Number of pre-filled cells in the puzzle.
    """

    solve_time_ms: float = 0.0
    nodes_explored: int = 0
    solutions_found: int = 0
    matrix_columns: int = 0
    matrix_rows: int = 0
    clues_given: int = 0

    def __str__(self) -> str:
        """Return a human-readable summary of the solve metrics."""
        return (
            f"Solve time:      {self.solve_time_ms:.3f} ms\n"
            f"Nodes explored:  {self.nodes_explored}\n"
            f"Solutions found: {self.solutions_found}\n"
            f"Matrix size:     {self.matrix_rows} rows x "
            f"{self.matrix_columns} columns\n"
            f"Clues given:     {self.clues_given}"
        )


class MetricsCollector:
    """Collect timing and node-count metrics during a solve.

    Use as a context manager to automatically start and stop the timer::

        collector = MetricsCollector()
        with collector:
            for node in search():
                collector.record_node()
        print(collector.to_metrics())

    Alternatively, call :meth:`start` and :meth:`stop` explicitly.

    Attributes:
        nodes_explored: Running count of search nodes visited.
        solutions_found: Running count of solutions discovered.
        matrix_columns: Number of constraint columns (set by caller).
        matrix_rows: Number of choice rows (set by caller).
        clues_given: Number of pre-filled cells (set by caller).
    """

    __slots__ = (
        "nodes_explored",
        "solutions_found",
        "matrix_columns",
        "matrix_rows",
        "clues_given",
        "_start_time",
        "_elapsed_ms",
    )

    def __init__(self) -> None:
        """Initialize the collector with zeroed counters."""
        self.nodes_explored: int = 0
        self.solutions_found: int = 0
        self.matrix_columns: int = 0
        self.matrix_rows: int = 0
        self.clues_given: int = 0
        self._start_time: float = 0.0
        self._elapsed_ms: float = 0.0

    def start(self) -> None:
        """Begin timing.

        Records the current high-resolution timestamp.  Resets the
        elapsed time so the collector can be reused.
        """
        self._elapsed_ms = 0.0
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop timing and compute elapsed milliseconds.

        Safe to call multiple times -- subsequent calls have no effect
        if the timer has already been stopped (elapsed is nonzero).
        """
        if self._elapsed_ms == 0.0 and self._start_time != 0.0:
            elapsed_s = time.perf_counter() - self._start_time
            self._elapsed_ms = elapsed_s * 1000.0

    def record_node(self) -> None:
        """Increment the search-node counter by one."""
        self.nodes_explored += 1

    def record_solution(self) -> None:
        """Increment the solution counter by one."""
        self.solutions_found += 1

    def to_metrics(self) -> SolveMetrics:
        """Build an immutable :class:`SolveMetrics` snapshot.

        If the timer is still running, it is stopped first so that the
        returned metrics reflect the time up to this call.

        Returns:
            A frozen :class:`SolveMetrics` dataclass with all collected
            values.
        """
        self.stop()
        return SolveMetrics(
            solve_time_ms=self._elapsed_ms,
            nodes_explored=self.nodes_explored,
            solutions_found=self.solutions_found,
            matrix_columns=self.matrix_columns,
            matrix_rows=self.matrix_rows,
            clues_given=self.clues_given,
        )

    def __enter__(self) -> MetricsCollector:
        """Enter the context manager and start timing."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the context manager and stop timing."""
        self.stop()


def configure_logging(
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """Configure the ``sudoku_solver`` package logger.

    Sets the log level for the ``sudoku_solver`` logger hierarchy and
    attaches a :class:`logging.StreamHandler` with a simple format.
    Calling this function multiple times replaces previous handlers.

    Verbosity precedence (if both flags are set, ``quiet`` wins):
    - ``quiet=True``: WARNING and above only
    - ``verbose=True``: DEBUG and above
    - Neither: INFO and above (default)

    Args:
        verbose: If ``True``, set level to DEBUG for detailed output.
        quiet: If ``True``, set level to WARNING to suppress
            informational messages.  Takes precedence over *verbose*.
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate output on repeated calls.
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
