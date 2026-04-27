"""Oracle-aware scaling experiment for LLM Oracle benchmark integration.

Extends the standard ``ScalingExperiment`` with LLM-specific metrics
collection: success rate vs problem size, per-attempt diagnostics,
and token cost tracking. This enables direct 7-solver comparison
while preserving the oracle's unique measurement dimensions.

The key insight: for most solvers, the primary metric is *runtime*.
For the LLM Oracle, the primary metric is *success rate* -- the
probability that the oracle produces a correct assignment within
its attempt budget at each problem size. Runtime is dominated by
API latency and is less informative than the success probability
decay curve.

Example:
    >>> from p_equals_np.experimental.oracle_scaling_experiment import (
    ...     OracleScalingExperiment,
    ... )
    >>> from p_equals_np.brute_force import BruteForceSolver
    >>> from p_equals_np.experimental.llm_oracle_approach import (
    ...     LLMOracleSolver, MockLLMBackend,
    ... )
    >>> experiment = OracleScalingExperiment(
    ...     solvers=[BruteForceSolver(), LLMOracleSolver(backend=MockLLMBackend())],
    ...     variable_sizes=[5, 8],
    ...     instances_per_size=2,
    ...     timeout_per_instance=10.0,
    ... )
    >>> measurements, oracle_metrics = experiment.run_with_oracle_metrics()
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from p_equals_np.complexity_analysis import RuntimeMeasurement, ScalingExperiment
from p_equals_np.definitions import Solver
from p_equals_np.experimental.llm_oracle_approach import (
    LLMOracleSolver,
    OracleAggregateMetrics,
    OracleSolveMetrics,
    _compute_aggregate_metrics,
)
from p_equals_np.sat_generator import generate_random_ksat
from p_equals_np.sat_types import CNFFormula


# ---------------------------------------------------------------------------
# Oracle-specific measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OracleInstanceResult:
    """Result of running the LLM Oracle on a single SAT instance.

    Captures both the standard RuntimeMeasurement data (for comparison
    with other solvers) and the oracle-specific OracleSolveMetrics
    (for success rate analysis).

    Attributes:
        runtime: Standard RuntimeMeasurement for this (solver, instance) pair.
        oracle_metrics: Per-solve OracleSolveMetrics, or None if the solver
            is not an LLMOracleSolver.
    """

    runtime: RuntimeMeasurement
    oracle_metrics: Optional[OracleSolveMetrics]


@dataclass
class OracleBenchmarkReport:
    """Complete benchmark report including oracle-specific analysis.

    Combines the standard scaling analysis with LLM Oracle success
    rate data, enabling the 7-solver comparison table.

    Attributes:
        standard_measurements: All RuntimeMeasurement records for all solvers.
        oracle_instance_results: Per-instance oracle results (oracle solvers only).
        oracle_aggregate: Aggregate oracle metrics across all instances.
        scaling_analysis: Standard scaling analysis dict from ScalingExperiment.
        success_rate_by_size: Oracle success rate keyed by num_variables.
        mean_attempts_by_size: Mean attempts to success keyed by num_variables.
    """

    standard_measurements: list[RuntimeMeasurement] = field(default_factory=list)
    oracle_instance_results: list[OracleInstanceResult] = field(default_factory=list)
    oracle_aggregate: Optional[OracleAggregateMetrics] = None
    scaling_analysis: dict = field(default_factory=dict)
    success_rate_by_size: dict[int, float] = field(default_factory=dict)
    mean_attempts_by_size: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize the report to a JSON-compatible dictionary.

        Returns:
            Dict containing all report fields with nested serialization.
        """
        result: dict = {
            "num_standard_measurements": len(self.standard_measurements),
            "num_oracle_results": len(self.oracle_instance_results),
            "success_rate_by_size": {
                str(k): v for k, v in self.success_rate_by_size.items()
            },
            "mean_attempts_by_size": {
                str(k): v for k, v in self.mean_attempts_by_size.items()
            },
        }
        if self.oracle_aggregate is not None:
            result["oracle_aggregate"] = self.oracle_aggregate.to_dict()
        return result


# ---------------------------------------------------------------------------
# OracleScalingExperiment
# ---------------------------------------------------------------------------


class OracleScalingExperiment(ScalingExperiment):
    """ScalingExperiment extended with LLM Oracle metrics collection.

    Inherits all standard functionality (run_experiment, analyze_scaling,
    generate_scaling_report, plot_scaling) and adds:

    - ``run_with_oracle_metrics()``: runs the experiment and collects
      oracle-specific metrics alongside standard RuntimeMeasurements.
    - ``generate_oracle_report()``: produces a combined report with
      success rate vs problem size as the primary oracle metric.
    - ``export_oracle_report()``: writes the report to JSON.

    Any solver passed to this experiment is benchmarked normally.
    Solvers that are instances of ``LLMOracleSolver`` additionally
    have their per-solve metrics captured.
    """

    def run_with_oracle_metrics(
        self, clause_ratio: float = 4.267
    ) -> tuple[list[RuntimeMeasurement], OracleBenchmarkReport]:
        """Run the experiment and collect oracle-specific metrics.

        Behaves like ``run_experiment()`` but additionally captures
        OracleSolveMetrics for any LLMOracleSolver in the solver list.

        Args:
            clause_ratio: Clause-to-variable ratio for instance
                generation (default 4.267, the 3-SAT phase transition).

        Returns:
            A tuple of (measurements, report) where measurements is the
            standard list and report contains oracle-specific analysis.
        """
        all_measurements: list[RuntimeMeasurement] = []
        oracle_results: list[OracleInstanceResult] = []

        # Reset metrics on all oracle solvers before the run
        for solver in self.solvers:
            if isinstance(solver, LLMOracleSolver):
                solver.reset_metrics()

        for size in self.variable_sizes:
            num_clauses = round(size * clause_ratio)
            instances = self._generate_instances(size, num_clauses)

            for solver in self.solvers:
                solver_name = solver.name()
                is_oracle = isinstance(solver, LLMOracleSolver)

                for i, formula in enumerate(instances):
                    print(
                        f"  {solver_name}: n={size}, "
                        f"instance {i + 1}/{self.instances_per_size}",
                        flush=True,
                    )

                    # Track metrics count before solve for oracle solvers
                    metrics_before = (
                        len(solver.get_metrics()) if is_oracle else 0
                    )

                    measurement = self._run_single(
                        solver, solver_name, formula
                    )
                    all_measurements.append(measurement)

                    # Capture oracle-specific metrics
                    oracle_metric = None
                    if is_oracle:
                        current_metrics = solver.get_metrics()
                        if len(current_metrics) > metrics_before:
                            oracle_metric = current_metrics[-1]

                    oracle_results.append(
                        OracleInstanceResult(
                            runtime=measurement,
                            oracle_metrics=oracle_metric,
                        )
                    )

        # Build the report
        report = self._build_oracle_report(all_measurements, oracle_results)
        return all_measurements, report

    def _build_oracle_report(
        self,
        measurements: list[RuntimeMeasurement],
        oracle_results: list[OracleInstanceResult],
    ) -> OracleBenchmarkReport:
        """Build the OracleBenchmarkReport from collected data.

        Args:
            measurements: All standard measurements.
            oracle_results: Per-instance oracle results.

        Returns:
            A fully populated OracleBenchmarkReport.
        """
        # Extract oracle-specific metrics
        oracle_metrics_list = [
            r.oracle_metrics
            for r in oracle_results
            if r.oracle_metrics is not None
        ]

        oracle_aggregate = None
        success_by_size: dict[int, float] = {}
        attempts_by_size: dict[int, float] = {}

        if oracle_metrics_list:
            oracle_aggregate = _compute_aggregate_metrics(oracle_metrics_list)
            success_by_size = dict(oracle_aggregate.success_rate_by_size)

            # Compute mean attempts to success by size
            size_attempts: dict[int, list[int]] = defaultdict(list)
            for m in oracle_metrics_list:
                if m.successful and m.successful_attempt is not None:
                    size_attempts[m.num_variables].append(m.successful_attempt)
            attempts_by_size = {
                n: sum(attempts) / len(attempts)
                for n, attempts in sorted(size_attempts.items())
                if attempts
            }

        # Standard scaling analysis
        scaling_analysis = self.analyze_scaling(measurements)

        return OracleBenchmarkReport(
            standard_measurements=measurements,
            oracle_instance_results=oracle_results,
            oracle_aggregate=oracle_aggregate,
            scaling_analysis=scaling_analysis,
            success_rate_by_size=success_by_size,
            mean_attempts_by_size=attempts_by_size,
        )

    def generate_oracle_report(
        self, report: OracleBenchmarkReport
    ) -> str:
        """Generate a text report including oracle-specific metrics.

        Extends the standard scaling report with a section on LLM Oracle
        success rate vs problem size -- the key metric for evaluating
        whether the oracle's guess probability decays exponentially.

        Args:
            report: The benchmark report from run_with_oracle_metrics.

        Returns:
            A multi-line string containing the full report.
        """
        lines: list[str] = []

        # Standard scaling report
        if report.scaling_analysis:
            lines.append(self.generate_scaling_report(report.scaling_analysis))
            lines.append("")

        # Oracle-specific section
        lines.append("=" * 72)
        lines.append("LLM ORACLE: SUCCESS RATE VS PROBLEM SIZE")
        lines.append("=" * 72)
        lines.append("")

        if not report.success_rate_by_size:
            lines.append("  No oracle metrics collected.")
            lines.append("")
            return "\n".join(lines)

        lines.append("  Size | Success Rate | Mean Attempts (success)")
        lines.append("  " + "-" * 50)
        for size in sorted(report.success_rate_by_size.keys()):
            rate = report.success_rate_by_size[size]
            attempts_str = "N/A"
            if size in report.mean_attempts_by_size:
                attempts_str = f"{report.mean_attempts_by_size[size]:.1f}"
            lines.append(f"  {size:>4d} | {rate:>11.1%} | {attempts_str}")

        lines.append("")

        if report.oracle_aggregate is not None:
            agg = report.oracle_aggregate
            lines.append(f"  Overall success rate: {agg.success_rate:.1%}")
            lines.append(f"  Total instances: {agg.total_instances}")
            lines.append(f"  Successful: {agg.successful_instances}")
            if agg.mean_attempts_to_success > 0:
                lines.append(
                    f"  Mean attempts to success: "
                    f"{agg.mean_attempts_to_success:.1f}"
                )
            lines.append(
                f"  Mean first-attempt clause violations: "
                f"{agg.mean_clauses_violated_first_attempt:.1f}"
            )
            lines.append("")

        lines.append("  NOTE: The exponential decay of success rate with")
        lines.append("  problem size is the key empirical signature of the")
        lines.append("  P/NP boundary in the probabilistic oracle setting.")
        lines.append("  The LLM hits an *information-theoretic wall*, not")
        lines.append("  a computational wall.")
        lines.append("")
        lines.append("=" * 72)

        return "\n".join(lines)

    def export_oracle_report(
        self,
        report: OracleBenchmarkReport,
        filepath: str,
    ) -> None:
        """Export the oracle benchmark report to a JSON file.

        Args:
            report: The benchmark report to export.
            filepath: Output file path.
        """
        data = report.to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
