"""Command-line interface for the Sudoku DLX solver.

This module provides a CLI for solving Sudoku puzzles using the Dancing Links
(DLX) exact cover solver.  Puzzles can be provided as command-line arguments,
read from files, or piped through stdin.

Exit codes:
    0 -- puzzle solved successfully
    1 -- puzzle has no solution
    2 -- error (invalid input, file not found, etc.)

Typical usage::

    # Solve a puzzle from the command line
    python -m sudoku_solver "003020600..."

    # Solve from a file
    python -m sudoku_solver -f puzzles.txt

    # Pipe from stdin
    echo "003020600..." | python -m sudoku_solver
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sudoku_solver.constraint_mapper import SudokuConstraintMapper
from sudoku_solver.solution_decoder import (
    NoSolutionError,
    SolutionDecoder,
)
from sudoku_solver.solver import DLXSolver

EXIT_SUCCESS: int = 0
EXIT_NO_SOLUTION: int = 1
EXIT_ERROR: int = 2

PUZZLE_LENGTH: int = 81

logger = logging.getLogger("sudoku_solver")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Sudoku solver.

    Args:
        args: Argument list to parse.  Defaults to ``sys.argv[1:]`` when
            ``None``.

    Returns:
        A namespace with the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="sudoku_solver",
        description="Solve Sudoku puzzles using Dancing Links (Algorithm X).",
    )
    parser.add_argument(
        "puzzle",
        nargs="?",
        default=None,
        help="81-character puzzle string (0 or . for empty cells).",
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="Path to a file containing one puzzle per line.",
    )
    parser.add_argument(
        "-o", "--format",
        choices=("pretty", "compact", "both"),
        default="pretty",
        dest="output_format",
        help="Output format: pretty (default), compact, or both.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        default=False,
        help="Count solutions instead of displaying them.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        dest="find_all",
        help="Find all solutions (default: stop after first).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress output except the solution itself.",
    )
    return parser.parse_args(args)


def parse_puzzle_string(puzzle_str: str) -> list[list[int]]:
    """Convert an 81-character puzzle string into a 9x9 grid.

    Strips whitespace and newlines, replaces '.' with '0', and validates
    that the result is exactly 81 characters of digits 0-9.

    Args:
        puzzle_str: An 81-character string where 1-9 are placed digits
            and 0 or '.' represent empty cells.

    Returns:
        A 9x9 list of lists with integers 0-9.

    Raises:
        ValueError: If the cleaned string is not exactly 81 characters or
            contains non-digit characters.
    """
    cleaned = puzzle_str.strip().replace("\n", "").replace("\r", "")
    cleaned = cleaned.replace(".", "0")

    if len(cleaned) != PUZZLE_LENGTH:
        raise ValueError(
            f"Puzzle must be exactly {PUZZLE_LENGTH} characters, "
            f"got {len(cleaned)}"
        )

    if not cleaned.isdigit():
        raise ValueError(
            "Puzzle must contain only digits 0-9 (or '.' for empty cells)"
        )

    return [
        [int(cleaned[row * 9 + col]) for col in range(9)]
        for row in range(9)
    ]


def _display_solution(
    grid: list[list[int]],
    args: argparse.Namespace,
    decoder: SolutionDecoder,
) -> None:
    """Display a solved grid in the requested format.

    Args:
        grid: A solved 9x9 grid.
        args: Parsed command-line arguments (for output_format).
        decoder: The decoder instance used for formatting.
    """
    if args.output_format in ("pretty", "both"):
        print(decoder.format_grid(grid))
    if args.output_format in ("compact", "both"):
        print(decoder.format_grid_compact(grid))


def solve_and_display(
    grid: list[list[int]], args: argparse.Namespace
) -> int:
    """Build the exact cover matrix, solve, and display results.

    Args:
        grid: A 9x9 Sudoku grid with 0 for empty cells.
        args: Parsed command-line arguments controlling output format,
            count mode, and all-solutions mode.

    Returns:
        Exit code: 0 if solved, 1 if no solution, 2 on error.
    """
    mapper = SudokuConstraintMapper()
    decoder = SolutionDecoder()

    try:
        matrix = mapper.build_matrix(grid)
    except Exception as exc:
        logger.error("Failed to build constraint matrix: %s", exc)
        return EXIT_ERROR

    max_solutions = 0 if (args.find_all or args.count) else 1
    solver = DLXSolver(matrix, max_solutions=max_solutions)

    if args.count:
        return _handle_count_mode(solver, args)

    return _handle_solve_mode(solver, decoder, args)


def _handle_count_mode(
    solver: DLXSolver, args: argparse.Namespace
) -> int:
    """Count solutions and display the count.

    Args:
        solver: A configured DLX solver.
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 if at least one solution exists, 1 otherwise.
    """
    count = solver.count_solutions(limit=0)
    if not args.quiet:
        print(f"Solutions: {count}")
    return EXIT_SUCCESS if count > 0 else EXIT_NO_SOLUTION


def _handle_solve_mode(
    solver: DLXSolver,
    decoder: SolutionDecoder,
    args: argparse.Namespace,
) -> int:
    """Solve and display solutions.

    Args:
        solver: A configured DLX solver.
        decoder: The solution decoder for grid reconstruction.
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 if at least one solution found, 1 otherwise.
    """
    found = 0
    for solution in solver.solve():
        grid = decoder.decode(solution)
        _display_solution(grid, args, decoder)
        found += 1
        if found > 1 and not args.quiet:
            print()  # Blank line between multiple solutions.

    if found == 0:
        if not args.quiet:
            print("No solution found.", file=sys.stderr)
        return EXIT_NO_SOLUTION

    return EXIT_SUCCESS


def _read_puzzles_from_file(path: str) -> list[str]:
    """Read puzzle strings from a file, one per line.

    Blank lines and lines starting with '#' are skipped.

    Args:
        path: Path to the puzzle file.

    Returns:
        A list of puzzle strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Puzzle file not found: {path}")

    lines: list[str] = []
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines


def _read_puzzle_from_stdin() -> str:
    """Read a puzzle string from stdin.

    Returns:
        The puzzle string read from stdin.

    Raises:
        ValueError: If stdin is a terminal with no data or is empty.
    """
    if sys.stdin.isatty():
        raise ValueError(
            "No puzzle provided. Pass a puzzle string, use --file, "
            "or pipe input via stdin."
        )
    data = sys.stdin.read().strip()
    if not data:
        raise ValueError("No input received from stdin.")
    return data


def _solve_single_puzzle(puzzle_str: str, args: argparse.Namespace) -> int:
    """Parse and solve a single puzzle string.

    Args:
        puzzle_str: An 81-character puzzle string.
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    try:
        grid = parse_puzzle_string(puzzle_str)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    return solve_and_display(grid, args)


def main() -> None:
    """Entry point for the Sudoku solver CLI.

    Parses arguments, reads puzzles from the appropriate source,
    solves them, and exits with the appropriate code.
    """
    args = parse_args()
    _configure_logging(args)

    try:
        exit_code = _dispatch(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        exit_code = EXIT_ERROR

    sys.exit(exit_code)


def _configure_logging(args: argparse.Namespace) -> None:
    """Configure logging based on verbosity flags.

    Args:
        args: Parsed command-line arguments.
    """
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def _dispatch(args: argparse.Namespace) -> int:
    """Route to the correct input source and solve.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    if args.puzzle is not None:
        return _solve_single_puzzle(args.puzzle, args)

    if args.file is not None:
        return _dispatch_file(args)

    return _dispatch_stdin(args)


def _dispatch_file(args: argparse.Namespace) -> int:
    """Read and solve all puzzles from a file.

    Args:
        args: Parsed command-line arguments (must have args.file set).

    Returns:
        The worst (highest) exit code across all puzzles.
    """
    try:
        puzzle_strings = _read_puzzles_from_file(args.file)
    except (FileNotFoundError, PermissionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if not puzzle_strings:
        print("Error: No puzzles found in file.", file=sys.stderr)
        return EXIT_ERROR

    worst_exit = EXIT_SUCCESS
    for i, puzzle_str in enumerate(puzzle_strings):
        if i > 0 and not args.quiet:
            print()  # Blank line between puzzles.
        result = _solve_single_puzzle(puzzle_str, args)
        worst_exit = max(worst_exit, result)

    return worst_exit


def _dispatch_stdin(args: argparse.Namespace) -> int:
    """Read and solve a puzzle from stdin.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    try:
        puzzle_str = _read_puzzle_from_stdin()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    return _solve_single_puzzle(puzzle_str, args)


if __name__ == "__main__":
    main()
