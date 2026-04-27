#!/usr/bin/env python3
"""Benchmark runner: 7-solver comparison including LLM Oracle.

Runs all seven SAT solvers on random 3-SAT instances at the phase
transition and produces a comparative scaling report. The LLM Oracle
is benchmarked alongside the six existing solvers, with success rate
vs problem size as its primary metric.

Modes:
    --live     Use real LLM API (requires ANTHROPIC_API_KEY env var).
    --mock     Use MockLLMBackend (no API key needed, default).
    --quick    Reduce sizes and instances for fast validation.

Usage:
    python -m p_equals_np.benchmark_llm_oracle
    python -m p_equals_np.benchmark_llm_oracle --mock --quick
    python -m p_equals_np.benchmark_llm_oracle --live --output results/

Notes:
    Mock results are NOT representative of actual LLM performance.
    The mock backend generates uniformly random assignments, giving
    it a success probability of ~2^(-n) on n-variable formulas.
    Real LLM backends may exploit structural patterns to achieve
    higher success rates on small instances.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure the package is importable when run as a script
_package_dir = Path(__file__).resolve().parent
if str(_package_dir.parent) not in sys.path:
    sys.path.insert(0, str(_package_dir.parent))

from p_equals_np.brute_force import BruteForceSolver
from p_equals_np.complexity_analysis import ScalingExperiment
from p_equals_np.dpll import DPLLSolver
from p_equals_np.experimental.algebraic_approach import AlgebraicSolver
from p_equals_np.experimental.geometric_approach import LPRelaxationSolver
from p_equals_np.experimental.llm_oracle_approach import (
    LLMOracleSolver,
    MockLLMBackend,
)
from p_equals_np.experimental.oracle_scaling_experiment import (
    OracleScalingExperiment,
)
from p_equals_np.experimental.spectral_approach import SpectralSolver
from p_equals_np.experimental.structural_approach import StructuralSolver


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Full benchmark: 7 sizes, 5 instances each
FULL_VARIABLE_SIZES = [5, 8, 10, 12, 15, 18, 20]
FULL_INSTANCES_PER_SIZE = 5

# Quick benchmark: 4 sizes, 2 instances each (for validation)
QUICK_VARIABLE_SIZES = [5, 8, 10, 12]
QUICK_INSTANCES_PER_SIZE = 2

CLAUSE_RATIO = 4.267  # 3-SAT phase transition
TIMEOUT_PER_INSTANCE = 30.0  # seconds
ORACLE_TIMEOUT = 60.0  # longer timeout for LLM API calls


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------


def build_solvers(
    use_live_api: bool = False,
    oracle_only: bool = False,
) -> list:
    """Build the list of solvers for the benchmark.

    Args:
        use_live_api: If True, use the real Anthropic API for the oracle.
            If False, use MockLLMBackend.
        oracle_only: If True, only include the LLM Oracle solver.

    Returns:
        List of solver instances.
    """
    # Build the LLM Oracle solver
    if use_live_api:
        oracle = LLMOracleSolver(
            encoding="structured",
            strategy="baseline",
            max_attempts=5,
            timeout_seconds=ORACLE_TIMEOUT,
        )
    else:
        oracle = LLMOracleSolver(
            backend=MockLLMBackend(seed=42),
            encoding="structured",
            strategy="baseline",
            max_attempts=5,
            timeout_seconds=ORACLE_TIMEOUT,
        )

    if oracle_only:
        return [oracle]

    # All 7 solvers
    return [
        BruteForceSolver(timeout_seconds=TIMEOUT_PER_INSTANCE),
        DPLLSolver(timeout_seconds=TIMEOUT_PER_INSTANCE),
        AlgebraicSolver(),
        SpectralSolver(timeout_seconds=TIMEOUT_PER_INSTANCE),
        LPRelaxationSolver(),
        StructuralSolver(),
        oracle,
    ]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_comparison_table(
    report,
    measurements: list,
) -> str:
    """Format a 7-solver comparison table.

    Args:
        report: OracleBenchmarkReport from the experiment.
        measurements: List of RuntimeMeasurement records.

    Returns:
        Formatted text table comparing all solvers.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("7-SOLVER COMPARISON: SAT SOLVER BENCHMARK")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Clause ratio: {CLAUSE_RATIO} (3-SAT phase transition)")
    lines.append("")

    # Group measurements by solver
    from collections import defaultdict

    by_solver: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for m in measurements:
        by_solver[m.solver_name][m.num_variables].append(m)

    # Get all sizes
    all_sizes = sorted(
        set(m.num_variables for m in measurements)
    )

    # Header
    size_headers = "".join(f"{'n=' + str(s):>10s}" for s in all_sizes)
    lines.append(f"  {'Solver':<25s} {size_headers}")
    lines.append("  " + "-" * (25 + 10 * len(all_sizes)))

    # Rows: median runtime per solver per size
    lines.append("  MEDIAN RUNTIME (seconds):")
    for solver_name in sorted(by_solver.keys()):
        solver_data = by_solver[solver_name]
        row = f"  {solver_name:<25s}"
        for size in all_sizes:
            if size in solver_data:
                times = [m.elapsed_seconds for m in solver_data[size]]
                median_t = sorted(times)[len(times) // 2]
                if median_t >= TIMEOUT_PER_INSTANCE * 0.99:
                    row += f"{'T/O':>10s}"
                else:
                    row += f"{median_t:>10.4f}"
            else:
                row += f"{'--':>10s}"
        lines.append(row)

    lines.append("")

    # Success rate section for oracle
    if report.success_rate_by_size:
        lines.append("  LLM ORACLE SUCCESS RATE:")
        row = f"  {'LLMOracle':.<25s}"
        for size in all_sizes:
            if size in report.success_rate_by_size:
                rate = report.success_rate_by_size[size]
                row += f"{rate:>9.0%} "
            else:
                row += f"{'--':>10s}"
        lines.append(row)
        lines.append("")

    # Solve rate (fraction of instances solved) for all solvers
    lines.append("  SOLVE RATE (fraction of instances producing a result):")
    for solver_name in sorted(by_solver.keys()):
        solver_data = by_solver[solver_name]
        row = f"  {solver_name:<25s}"
        for size in all_sizes:
            if size in solver_data:
                total = len(solver_data[size])
                solved = sum(1 for m in solver_data[size] if m.solved)
                rate = solved / total if total > 0 else 0.0
                row += f"{rate:>9.0%} "
            else:
                row += f"{'--':>10s}"
        lines.append(row)

    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------


def run_benchmark(
    use_live_api: bool = False,
    quick: bool = False,
    oracle_only: bool = False,
    output_dir: str = ".",
) -> None:
    """Run the full 7-solver benchmark.

    Args:
        use_live_api: Use real Anthropic API (requires ANTHROPIC_API_KEY).
        quick: Use reduced sizes and instances for fast validation.
        oracle_only: Only benchmark the LLM Oracle solver.
        output_dir: Directory for output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    variable_sizes = QUICK_VARIABLE_SIZES if quick else FULL_VARIABLE_SIZES
    instances_per_size = QUICK_INSTANCES_PER_SIZE if quick else FULL_INSTANCES_PER_SIZE

    solvers = build_solvers(
        use_live_api=use_live_api,
        oracle_only=oracle_only,
    )

    mode_label = "LIVE API" if use_live_api else "MOCK (random baseline)"
    print(f"\nLLM Oracle mode: {mode_label}")
    if not use_live_api:
        print(
            "  NOTE: Mock results use uniformly random assignments and are"
        )
        print(
            "  NOT representative of actual LLM performance."
        )
    print(f"Variable sizes: {variable_sizes}")
    print(f"Instances per size: {instances_per_size}")
    print(f"Solvers: {[s.name() for s in solvers]}")
    print()

    # Build and run the oracle-aware experiment
    experiment = OracleScalingExperiment(
        solvers=solvers,
        variable_sizes=variable_sizes,
        instances_per_size=instances_per_size,
        timeout_per_instance=TIMEOUT_PER_INSTANCE,
    )

    start_time = time.time()
    print("Running benchmark...")
    print("-" * 60)

    measurements, report = experiment.run_with_oracle_metrics(
        clause_ratio=CLAUSE_RATIO,
    )

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Benchmark completed in {elapsed:.1f} seconds.")
    print()

    # Generate reports
    comparison = format_comparison_table(report, measurements)
    print(comparison)

    oracle_text = experiment.generate_oracle_report(report)
    print(oracle_text)

    # Export results
    report_path = os.path.join(output_dir, "benchmark_report.json")
    experiment.export_oracle_report(report, report_path)
    print(f"\nOracle report exported to: {report_path}")

    # Export full text report
    text_report_path = os.path.join(output_dir, "benchmark_report.txt")
    with open(text_report_path, "w") as f:
        f.write(comparison)
        f.write("\n\n")
        f.write(oracle_text)
    print(f"Text report exported to: {text_report_path}")

    # Export oracle-specific metrics JSON (if oracle was run)
    for solver in solvers:
        if isinstance(solver, LLMOracleSolver):
            metrics = solver.get_metrics()
            if metrics:
                oracle_metrics_path = os.path.join(
                    output_dir, "oracle_metrics.json"
                )
                solver.export_metrics(oracle_metrics_path)
                print(f"Oracle metrics exported to: {oracle_metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="7-solver SAT benchmark including LLM Oracle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real Anthropic API (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use MockLLMBackend (default, no API key needed)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced sizes and instances for fast validation",
    )
    parser.add_argument(
        "--oracle-only",
        action="store_true",
        help="Only benchmark the LLM Oracle solver",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output directory for results (default: current directory)",
    )

    args = parser.parse_args()

    use_live = args.live and not args.mock

    run_benchmark(
        use_live_api=use_live,
        quick=args.quick,
        oracle_only=args.oracle_only,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
