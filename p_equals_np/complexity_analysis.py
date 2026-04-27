"""Empirical complexity analysis: runtime measurement, curve fitting, scaling.

Provides tools to empirically measure the runtime scaling of SAT solvers,
fit polynomial and exponential models to the observed data, and produce
reports and visualizations demonstrating the gap between polynomial-time
and exponential-time behavior.

Design decisions:
    - Pure Python implementations for all curve fitting (median, linear
      regression, polynomial fitting). numpy is used if available for
      improved numerical stability, but is not required.
    - matplotlib is used for plotting if available. Falls back to CSV
      export and text-based output when matplotlib is not installed.
    - Signal-based timeout (SIGALRM) on Unix for per-instance timeout,
      with cooperative timeout fallback on non-Unix platforms.
    - Median rather than mean for robustness to outlier instances.
    - Fixed seeds for reproducibility of all generated instances.

Important caveats:
    Polynomial curve fits on finite data CANNOT establish asymptotic
    complexity class membership. Small instances may appear polynomial
    even for exponential algorithms. The analysis here provides empirical
    *evidence* and *illustration*, not formal proof.

Example:
    >>> from p_equals_np.complexity_analysis import ScalingExperiment
    >>> from p_equals_np.brute_force import BruteForceSolver
    >>> from p_equals_np.dpll import DPLLSolver
    >>> experiment = ScalingExperiment(
    ...     solvers=[BruteForceSolver(), DPLLSolver()],
    ...     variable_sizes=[5, 8, 10],
    ...     instances_per_size=3,
    ...     timeout_per_instance=5.0,
    ... )
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

from p_equals_np.definitions import Solver
from p_equals_np.sat_generator import generate_random_ksat
from p_equals_np.sat_types import CNFFormula


# ---------------------------------------------------------------------------
# RuntimeMeasurement dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuntimeMeasurement:
    """Record of a single solver run on a single SAT instance.

    Captures timing data and outcome for empirical scaling analysis.
    Each measurement represents one (solver, instance) pair.

    Attributes:
        solver_name: Name of the solver that produced this measurement.
        num_variables: Number of Boolean variables in the instance.
        num_clauses: Number of clauses in the instance.
        clause_ratio: Clause-to-variable ratio of the instance.
        elapsed_seconds: Wall-clock time in seconds for the solve call.
        solved: Whether the solver produced a result (SAT or UNSAT
            determination). False if timed out or gave up.
        timed_out: Whether the solver exceeded the timeout limit.
    """

    solver_name: str
    num_variables: int
    num_clauses: int
    clause_ratio: float
    elapsed_seconds: float
    solved: bool
    timed_out: bool


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def median(values: list[float]) -> float:
    """Compute the median of a list of numeric values.

    For even-length lists, returns the average of the two middle values.

    Args:
        values: A non-empty list of numeric values.

    Returns:
        The median value.

    Raises:
        ValueError: If the list is empty.
    """
    if not values:
        raise ValueError("Cannot compute median of an empty list")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def linear_regression(
    x: list[float], y: list[float]
) -> tuple[float, float, float]:
    """Simple ordinary least squares linear regression.

    Fits the model y = slope * x + intercept and computes R-squared.

    Args:
        x: Independent variable values.
        y: Dependent variable values (same length as x).

    Returns:
        A tuple of (slope, intercept, r_squared).

    Raises:
        ValueError: If x and y have different lengths or fewer than
            2 data points.
    """
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length, got {len(x)} and {len(y)}"
        )
    if len(x) < 2:
        raise ValueError(
            f"Need at least 2 data points, got {len(x)}"
        )

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(xi * xi for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-15:
        return 0.0, sum_y / n, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R-squared
    mean_y = sum_y / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))

    if ss_tot < 1e-15:
        r_squared = 1.0 if ss_res < 1e-15 else 0.0
    else:
        r_squared = 1.0 - ss_res / ss_tot

    return slope, intercept, r_squared


# ---------------------------------------------------------------------------
# ScalingExperiment
# ---------------------------------------------------------------------------


class ScalingExperiment:
    """Run solvers on generated SAT instances and analyze scaling behavior.

    Generates random 3-SAT instances at a controlled clause ratio,
    measures each solver's runtime, and fits polynomial and exponential
    models to the observed data to empirically characterize scaling.

    Attributes:
        solvers: List of solvers conforming to the Solver protocol.
        variable_sizes: List of problem sizes (number of variables)
            to test across.
        instances_per_size: Number of random instances to generate
            at each size for statistical robustness.
        timeout_per_instance: Maximum seconds allowed per solver
            per instance before declaring timeout.
    """

    def __init__(
        self,
        solvers: list[Solver],
        variable_sizes: list[int],
        instances_per_size: int = 10,
        timeout_per_instance: float = 30.0,
    ) -> None:
        """Initialize a scaling experiment.

        Args:
            solvers: Solvers to benchmark.
            variable_sizes: Problem sizes to test.
            instances_per_size: Instances per size (default 10).
            timeout_per_instance: Timeout in seconds per instance
                (default 30.0).

        Raises:
            ValueError: If solvers or variable_sizes is empty, or if
                instances_per_size or timeout_per_instance is not positive.
        """
        if not solvers:
            raise ValueError("solvers must be non-empty")
        if not variable_sizes:
            raise ValueError("variable_sizes must be non-empty")
        if instances_per_size < 1:
            raise ValueError(
                f"instances_per_size must be >= 1, got {instances_per_size}"
            )
        if timeout_per_instance <= 0:
            raise ValueError(
                f"timeout_per_instance must be > 0, got {timeout_per_instance}"
            )

        self.solvers = list(solvers)
        self.variable_sizes = sorted(variable_sizes)
        self.instances_per_size = instances_per_size
        self.timeout_per_instance = timeout_per_instance

    # --- Experiment execution ---

    def run_experiment(
        self, clause_ratio: float = 4.267
    ) -> list[RuntimeMeasurement]:
        """Run all solvers on generated instances and collect measurements.

        For each variable size, generates ``instances_per_size`` random
        3-SAT instances at the given clause ratio. Each solver is run on
        each instance with timeout protection. Progress is printed to
        stdout.

        Seeds are computed as ``size * 1000 + instance_index`` for
        reproducibility.

        Args:
            clause_ratio: Clause-to-variable ratio for instance
                generation (default 4.267, the 3-SAT phase transition).

        Returns:
            A list of RuntimeMeasurement records, one per
            (solver, size, instance) triple.
        """
        measurements: list[RuntimeMeasurement] = []

        for size in self.variable_sizes:
            num_clauses = round(size * clause_ratio)
            instances = self._generate_instances(size, num_clauses)

            for solver in self.solvers:
                solver_name = solver.name()
                for i, formula in enumerate(instances):
                    print(
                        f"  {solver_name}: n={size}, "
                        f"instance {i + 1}/{self.instances_per_size}",
                        flush=True,
                    )
                    measurement = self._run_single(
                        solver, solver_name, formula
                    )
                    measurements.append(measurement)

        return measurements

    def _generate_instances(
        self, num_vars: int, num_clauses: int
    ) -> list[CNFFormula]:
        """Generate reproducible random 3-SAT instances for one size.

        Args:
            num_vars: Number of variables.
            num_clauses: Number of clauses.

        Returns:
            A list of CNFFormula instances.
        """
        instances: list[CNFFormula] = []
        for i in range(self.instances_per_size):
            seed = num_vars * 1000 + i
            formula = generate_random_ksat(
                k=3, num_vars=num_vars, num_clauses=num_clauses, seed=seed
            )
            instances.append(formula)
        return instances

    def _run_single(
        self,
        solver: Solver,
        solver_name: str,
        formula: CNFFormula,
    ) -> RuntimeMeasurement:
        """Run a single solver on a single instance with timeout.

        Uses signal-based timeout on Unix (SIGALRM) if available,
        otherwise relies on solver-internal timeout mechanisms.

        Args:
            solver: The solver to run.
            solver_name: Cached solver name string.
            formula: The SAT instance to solve.

        Returns:
            A RuntimeMeasurement recording the outcome.
        """
        num_vars = formula.num_variables
        num_clauses = formula.num_clauses
        ratio = formula.clause_variable_ratio

        timed_out = False
        solved = False
        elapsed = self.timeout_per_instance

        start = time.perf_counter()
        try:
            result = _run_with_timeout(
                solver, formula, self.timeout_per_instance
            )
            elapsed = time.perf_counter() - start
            solved = True
        except TimeoutError:
            elapsed = time.perf_counter() - start
            timed_out = True
            solved = False
        except Exception:
            elapsed = time.perf_counter() - start
            solved = False

        return RuntimeMeasurement(
            solver_name=solver_name,
            num_variables=num_vars,
            num_clauses=num_clauses,
            clause_ratio=ratio,
            elapsed_seconds=elapsed,
            solved=solved,
            timed_out=timed_out,
        )

    # --- Curve fitting ---

    def fit_polynomial(
        self, sizes: list[int], times: list[float]
    ) -> tuple[list[float], float]:
        """Fit the best polynomial model to observed scaling data.

        Tries polynomial degrees 1 through 6 and selects the degree
        with the highest R-squared value. Uses numpy.polyfit if
        available, otherwise falls back to pure Python least squares.

        Args:
            sizes: Problem sizes (independent variable).
            times: Observed median runtimes (dependent variable).

        Returns:
            A tuple of (coefficients, r_squared) where coefficients
            are ordered [a_0, a_1, ..., a_k] (constant term first)
            for the best-fitting polynomial of degree k.

        Raises:
            ValueError: If sizes and times have different lengths
                or fewer than 2 data points.
        """
        if len(sizes) != len(times):
            raise ValueError(
                f"sizes and times must have the same length, "
                f"got {len(sizes)} and {len(times)}"
            )
        if len(sizes) < 2:
            raise ValueError(
                f"Need at least 2 data points, got {len(sizes)}"
            )

        best_coeffs: list[float] = []
        best_r2 = -float("inf")

        max_degree = min(6, len(sizes) - 1)

        for degree in range(1, max_degree + 1):
            coeffs, r2 = self._fit_poly_degree(sizes, times, degree)
            if r2 > best_r2:
                best_r2 = r2
                best_coeffs = coeffs

        return best_coeffs, best_r2

    def fit_exponential(
        self, sizes: list[int], times: list[float]
    ) -> tuple[float, float, float]:
        """Fit an exponential model t = a * b^n to observed scaling data.

        Takes logarithms and performs linear regression on
        (n, log(t)) to find parameters a and b. Data points with
        non-positive times are excluded.

        Args:
            sizes: Problem sizes (independent variable).
            times: Observed median runtimes (dependent variable).

        Returns:
            A tuple of (a, b, r_squared) for the model t = a * b^n.

        Raises:
            ValueError: If sizes and times have different lengths
                or fewer than 2 valid (positive-time) data points.
        """
        if len(sizes) != len(times):
            raise ValueError(
                f"sizes and times must have the same length, "
                f"got {len(sizes)} and {len(times)}"
            )

        valid_sizes: list[float] = []
        log_times: list[float] = []
        for s, t in zip(sizes, times):
            if t > 0:
                valid_sizes.append(float(s))
                log_times.append(math.log(t))

        if len(valid_sizes) < 2:
            raise ValueError(
                f"Need at least 2 data points with positive times, "
                f"got {len(valid_sizes)}"
            )

        slope, intercept, log_r2 = linear_regression(valid_sizes, log_times)

        a = math.exp(intercept)
        b = math.exp(slope)

        # Compute R-squared in original (non-log) space
        mean_t = sum(times) / len(times)
        ss_tot = sum((t - mean_t) ** 2 for t in times)
        ss_res = 0.0
        for s, t in zip(sizes, times):
            predicted = a * (b ** float(s))
            ss_res += (t - predicted) ** 2

        if ss_tot < 1e-15:
            r_squared = 1.0 if ss_res < 1e-15 else 0.0
        else:
            r_squared = 1.0 - ss_res / ss_tot

        return a, b, r_squared

    # --- Scaling analysis ---

    def analyze_scaling(
        self, measurements: list[RuntimeMeasurement]
    ) -> dict[str, dict]:
        """Analyze scaling behavior for each solver from measurements.

        For each solver, extracts median runtimes at each problem size,
        fits both polynomial and exponential models, and determines
        which model provides a better fit (higher R-squared).

        Args:
            measurements: List of RuntimeMeasurement records from
                ``run_experiment``.

        Returns:
            A dict mapping solver_name to an analysis dict containing:
            - ``sizes``: list of problem sizes
            - ``median_times``: list of median runtimes per size
            - ``poly_fit``: (coefficients, r_squared) from fit_polynomial
            - ``exp_fit``: (a, b, r_squared) from fit_exponential
            - ``best_model``: "polynomial" or "exponential"
            - ``best_r2``: R-squared of the best model
        """
        # Group measurements by solver
        by_solver: dict[str, list[RuntimeMeasurement]] = {}
        for m in measurements:
            by_solver.setdefault(m.solver_name, []).append(m)

        analysis: dict[str, dict] = {}

        for solver_name, solver_measurements in by_solver.items():
            sizes, median_times = self._extract_median_times(
                solver_measurements
            )

            if len(sizes) < 2:
                analysis[solver_name] = {
                    "sizes": sizes,
                    "median_times": median_times,
                    "poly_fit": ([], 0.0),
                    "exp_fit": (0.0, 0.0, 0.0),
                    "best_model": "insufficient_data",
                    "best_r2": 0.0,
                }
                continue

            poly_coeffs, poly_r2 = self.fit_polynomial(sizes, median_times)
            try:
                exp_a, exp_b, exp_r2 = self.fit_exponential(
                    sizes, median_times
                )
            except ValueError:
                exp_a, exp_b, exp_r2 = 0.0, 0.0, -1.0

            if poly_r2 >= exp_r2:
                best_model = "polynomial"
                best_r2 = poly_r2
            else:
                best_model = "exponential"
                best_r2 = exp_r2

            analysis[solver_name] = {
                "sizes": sizes,
                "median_times": median_times,
                "poly_fit": (poly_coeffs, poly_r2),
                "exp_fit": (exp_a, exp_b, exp_r2),
                "best_model": best_model,
                "best_r2": best_r2,
            }

        return analysis

    # --- Report generation ---

    def generate_scaling_report(self, analysis: dict[str, dict]) -> str:
        """Generate a text report summarizing scaling analysis results.

        Includes model fits, R-squared values, and an honest assessment
        of what the empirical evidence shows (and does not show).

        Args:
            analysis: Output from ``analyze_scaling``.

        Returns:
            A multi-line string containing the formatted report.
        """
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("SCALING ANALYSIS REPORT")
        lines.append("=" * 72)
        lines.append("")
        lines.append(
            "CAVEAT: Polynomial curve fits on finite data CANNOT establish"
        )
        lines.append(
            "asymptotic complexity. Small instances may appear polynomial"
        )
        lines.append(
            "even for exponential algorithms. This is empirical evidence,"
        )
        lines.append("not formal proof.")
        lines.append("")

        for solver_name, data in sorted(analysis.items()):
            lines.extend(
                self._format_solver_section(solver_name, data)
            )

        lines.append("=" * 72)
        lines.append("END OF REPORT")
        lines.append("=" * 72)

        return "\n".join(lines)

    # --- Plotting ---

    def plot_scaling(
        self, analysis: dict[str, dict], output_path: str
    ) -> None:
        """Generate scaling plots or CSV fallback.

        If matplotlib is available, generates:
        1. A combined log-log plot of time vs size for all solvers.
        2. Individual plots per solver with polynomial and exponential
           fit curves overlaid.

        If matplotlib is not available, exports data as CSV files.

        Args:
            analysis: Output from ``analyze_scaling``.
            output_path: Directory path where output files will be saved.
        """
        os.makedirs(output_path, exist_ok=True)

        try:
            self._plot_with_matplotlib(analysis, output_path)
        except ImportError:
            self._export_csv_fallback(analysis, output_path)

    # --- Private helpers ---

    def _extract_median_times(
        self, measurements: list[RuntimeMeasurement]
    ) -> tuple[list[int], list[float]]:
        """Extract sizes and median times from measurements for one solver.

        Groups measurements by num_variables and computes the median
        elapsed time at each size. Timed-out instances use the timeout
        value as their elapsed time (conservative: actual time could be
        longer).

        Args:
            measurements: Measurements for a single solver.

        Returns:
            A tuple of (sizes, median_times), both sorted by size.
        """
        by_size: dict[int, list[float]] = {}
        for m in measurements:
            by_size.setdefault(m.num_variables, []).append(m.elapsed_seconds)

        sizes: list[int] = []
        median_times: list[float] = []
        for size in sorted(by_size.keys()):
            times = by_size[size]
            sizes.append(size)
            median_times.append(median(times))

        return sizes, median_times

    def _fit_poly_degree(
        self, sizes: list[int], times: list[float], degree: int
    ) -> tuple[list[float], float]:
        """Fit a polynomial of a specific degree and compute R-squared.

        Attempts to use numpy for numerical stability. Falls back to
        pure Python Vandermonde + Gaussian elimination if numpy is
        not available.

        Args:
            sizes: Problem sizes.
            times: Observed runtimes.
            degree: Polynomial degree to fit.

        Returns:
            A tuple of (coefficients, r_squared) where coefficients
            are [a_0, a_1, ..., a_degree] (constant term first).
        """
        try:
            return self._fit_poly_numpy(sizes, times, degree)
        except ImportError:
            return self._fit_poly_pure(sizes, times, degree)

    def _fit_poly_numpy(
        self, sizes: list[int], times: list[float], degree: int
    ) -> tuple[list[float], float]:
        """Fit polynomial using numpy.polyfit.

        Args:
            sizes: Problem sizes.
            times: Observed runtimes.
            degree: Polynomial degree.

        Returns:
            (coefficients, r_squared) with coefficients in
            ascending order [a_0, a_1, ..., a_degree].
        """
        import numpy as np

        x = np.array(sizes, dtype=float)
        y = np.array(times, dtype=float)
        coeffs_descending = np.polyfit(x, y, degree)
        coeffs_ascending = list(reversed(coeffs_descending))

        # R-squared
        predicted = np.polyval(coeffs_descending, x)
        ss_res = float(np.sum((y - predicted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))

        if ss_tot < 1e-15:
            r2 = 1.0 if ss_res < 1e-15 else 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        return coeffs_ascending, r2

    def _fit_poly_pure(
        self, sizes: list[int], times: list[float], degree: int
    ) -> tuple[list[float], float]:
        """Fit polynomial using pure Python Vandermonde + Gaussian elimination.

        Constructs the normal equations (V^T V) c = V^T y and solves
        via Gaussian elimination with partial pivoting.

        Args:
            sizes: Problem sizes.
            times: Observed runtimes.
            degree: Polynomial degree.

        Returns:
            (coefficients, r_squared) with coefficients in
            ascending order [a_0, a_1, ..., a_degree].
        """
        n = len(sizes)
        cols = degree + 1

        # Build normal equations
        vtv = [[0.0] * cols for _ in range(cols)]
        vty = [0.0] * cols

        for i in range(n):
            x = float(sizes[i])
            y = times[i]
            powers = [x ** p for p in range(cols)]
            for r in range(cols):
                for c in range(cols):
                    vtv[r][c] += powers[r] * powers[c]
                vty[r] += powers[r] * y

        coeffs = _solve_linear_system(vtv, vty)
        if coeffs is None:
            return [0.0] * cols, 0.0

        # R-squared
        mean_y = sum(times) / n
        ss_tot = sum((t - mean_y) ** 2 for t in times)
        ss_res = 0.0
        for i in range(n):
            x = float(sizes[i])
            predicted = sum(coeffs[p] * (x ** p) for p in range(cols))
            ss_res += (times[i] - predicted) ** 2

        if ss_tot < 1e-15:
            r2 = 1.0 if ss_res < 1e-15 else 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        return coeffs, r2

    def _format_solver_section(
        self, solver_name: str, data: dict
    ) -> list[str]:
        """Format a single solver's analysis results for the report.

        Args:
            solver_name: Name of the solver.
            data: Analysis dict for this solver.

        Returns:
            A list of formatted text lines.
        """
        lines: list[str] = []
        lines.append("-" * 72)
        lines.append(f"Solver: {solver_name}")
        lines.append("-" * 72)

        sizes = data["sizes"]
        median_times = data["median_times"]

        lines.append("")
        lines.append("  Size | Median Time (s)")
        lines.append("  " + "-" * 30)
        for s, t in zip(sizes, median_times):
            lines.append(f"  {s:>4d} | {t:.6f}")

        lines.append("")

        if data["best_model"] == "insufficient_data":
            lines.append("  Insufficient data for curve fitting.")
            lines.append("")
            return lines

        poly_coeffs, poly_r2 = data["poly_fit"]
        exp_a, exp_b, exp_r2 = data["exp_fit"]

        lines.append(f"  Polynomial fit (degree {len(poly_coeffs) - 1}):")
        lines.append(f"    R-squared: {poly_r2:.6f}")
        coeff_str = ", ".join(f"{c:.6e}" for c in poly_coeffs)
        lines.append(f"    Coefficients [a_0..a_k]: [{coeff_str}]")

        lines.append("")
        lines.append(f"  Exponential fit (t = a * b^n):")
        lines.append(f"    R-squared: {exp_r2:.6f}")
        lines.append(f"    a = {exp_a:.6e}, b = {exp_b:.6f}")

        lines.append("")
        best = data["best_model"]
        best_r2 = data["best_r2"]
        lines.append(f"  Best fit: {best} (R-squared = {best_r2:.6f})")

        if best == "polynomial" and len(sizes) > 0 and max(sizes) < 30:
            lines.append(
                "  WARNING: Polynomial fit on small instances is NOT"
            )
            lines.append(
                "  evidence of polynomial-time complexity. Exponential"
            )
            lines.append(
                "  behavior may only manifest at larger sizes."
            )

        lines.append("")
        return lines

    def _plot_with_matplotlib(
        self, analysis: dict[str, dict], output_path: str
    ) -> None:
        """Generate matplotlib scaling plots.

        Creates a combined log-log plot and individual per-solver
        plots with fit curves.

        Args:
            analysis: Output from ``analyze_scaling``.
            output_path: Directory path for output files.

        Raises:
            ImportError: If matplotlib is not available.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Combined log-log plot
        fig, ax = plt.subplots(figsize=(10, 7))
        for solver_name, data in sorted(analysis.items()):
            sizes = data["sizes"]
            times = data["median_times"]
            if sizes and times:
                ax.plot(sizes, times, "o-", label=solver_name, markersize=4)

        ax.set_xlabel("Number of Variables (n)")
        ax.set_ylabel("Median Runtime (seconds)")
        ax.set_title("SAT Solver Scaling: Runtime vs Problem Size")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, "scaling_combined.png"), dpi=150)
        plt.close(fig)

        # Individual solver plots with fit curves
        for solver_name, data in sorted(analysis.items()):
            if data["best_model"] == "insufficient_data":
                continue
            self._plot_single_solver(solver_name, data, output_path)

    def _plot_single_solver(
        self, solver_name: str, data: dict, output_path: str
    ) -> None:
        """Generate a plot for a single solver with fit curves overlaid.

        Args:
            solver_name: Name of the solver.
            data: Analysis dict for this solver.
            output_path: Directory for output files.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        sizes = data["sizes"]
        times = data["median_times"]
        poly_coeffs, poly_r2 = data["poly_fit"]
        exp_a, exp_b, exp_r2 = data["exp_fit"]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(sizes, times, "ko", label="Observed", markersize=6)

        # Generate smooth x values for fit curves
        x_smooth = np.linspace(min(sizes), max(sizes), 200)

        # Polynomial fit curve
        if poly_coeffs:
            poly_y = sum(
                c * x_smooth ** p for p, c in enumerate(poly_coeffs)
            )
            degree = len(poly_coeffs) - 1
            ax.plot(
                x_smooth, poly_y, "b--",
                label=f"Polynomial (deg {degree}, R2={poly_r2:.4f})",
            )

        # Exponential fit curve
        if exp_b > 0:
            exp_y = exp_a * exp_b ** x_smooth
            ax.plot(
                x_smooth, exp_y, "r-",
                label=f"Exponential (b={exp_b:.3f}, R2={exp_r2:.4f})",
            )

        ax.set_xlabel("Number of Variables (n)")
        ax.set_ylabel("Median Runtime (seconds)")
        ax.set_title(f"Scaling Analysis: {solver_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_name = solver_name.replace(" ", "_").replace("/", "_")
        fig.savefig(
            os.path.join(output_path, f"scaling_{safe_name}.png"), dpi=150
        )
        plt.close(fig)

    def _export_csv_fallback(
        self, analysis: dict[str, dict], output_path: str
    ) -> None:
        """Export scaling data as CSV when matplotlib is not available.

        Creates one CSV file per solver with columns:
        size, median_time, poly_predicted, exp_predicted.

        Also creates a summary text file.

        Args:
            analysis: Output from ``analyze_scaling``.
            output_path: Directory for output files.
        """
        for solver_name, data in sorted(analysis.items()):
            sizes = data["sizes"]
            times = data["median_times"]

            safe_name = solver_name.replace(" ", "_").replace("/", "_")
            csv_path = os.path.join(output_path, f"scaling_{safe_name}.csv")

            with open(csv_path, "w") as f:
                f.write("size,median_time\n")
                for s, t in zip(sizes, times):
                    f.write(f"{s},{t:.8f}\n")

        # Summary text
        summary_path = os.path.join(output_path, "scaling_summary.txt")
        report = self.generate_scaling_report(analysis)
        with open(summary_path, "w") as f:
            f.write(report)


# ---------------------------------------------------------------------------
# Timeout helpers
# ---------------------------------------------------------------------------


def _run_with_timeout(
    solver: Solver, formula: CNFFormula, timeout: float
) -> Optional[dict[int, bool]]:
    """Run a solver with timeout protection.

    On Unix, uses signal.SIGALRM for reliable timeout enforcement.
    On other platforms, relies on the solver's internal timeout
    mechanism (sets solver.timeout_seconds if the attribute exists).

    Args:
        solver: The solver to run.
        formula: The SAT instance.
        timeout: Maximum seconds allowed.

    Returns:
        The solver's result (satisfying assignment or None).

    Raises:
        TimeoutError: If the solver exceeds the timeout.
    """
    # Try to set solver-level timeout if available
    if hasattr(solver, "timeout_seconds"):
        solver.timeout_seconds = timeout  # type: ignore[attr-defined]

    # Try signal-based timeout on Unix
    if _has_signal_alarm():
        return _run_with_signal_timeout(solver, formula, timeout)

    # Fallback: rely on solver-internal timeout
    return solver.solve(formula)


def _has_signal_alarm() -> bool:
    """Check if signal.SIGALRM is available (Unix only).

    Returns:
        True if SIGALRM can be used for timeout enforcement.
    """
    try:
        import signal
        return hasattr(signal, "SIGALRM")
    except ImportError:
        return False


def _run_with_signal_timeout(
    solver: Solver, formula: CNFFormula, timeout: float
) -> Optional[dict[int, bool]]:
    """Run solver with Unix signal-based timeout.

    Args:
        solver: The solver to run.
        formula: The SAT instance.
        timeout: Maximum seconds allowed.

    Returns:
        The solver's result.

    Raises:
        TimeoutError: If the alarm fires before completion.
    """
    import signal

    def _alarm_handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"Solver exceeded {timeout}s timeout (SIGALRM)")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    # Use integer seconds (ceiling) for alarm
    alarm_seconds = max(1, int(math.ceil(timeout)))
    signal.alarm(alarm_seconds)

    try:
        result = solver.solve(formula)
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Linear system solver (pure Python fallback)
# ---------------------------------------------------------------------------


def _solve_linear_system(
    matrix: list[list[float]], rhs: list[float]
) -> Optional[list[float]]:
    """Solve a linear system Ax = b via Gaussian elimination with pivoting.

    Makes copies of the input to avoid mutation.

    Args:
        matrix: Square coefficient matrix.
        rhs: Right-hand side vector.

    Returns:
        Solution vector, or None if the system is singular.
    """
    n = len(rhs)
    a = [row[:] for row in matrix]
    b = rhs[:]

    for col in range(n):
        # Partial pivoting
        max_row = col
        max_val = abs(a[col][col])
        for row in range(col + 1, n):
            if abs(a[row][col]) > max_val:
                max_val = abs(a[row][col])
                max_row = row
        if max_val < 1e-12:
            return None
        a[col], a[max_row] = a[max_row], a[col]
        b[col], b[max_row] = b[max_row], b[col]

        # Eliminate below
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for k in range(col, n):
                a[row][k] -= factor * a[col][k]
            b[row] -= factor * b[col]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(a[i][i]) < 1e-12:
            return None
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= a[i][j] * x[j]
        x[i] /= a[i][i]

    return x
