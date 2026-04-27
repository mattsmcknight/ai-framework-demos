"""Formal complexity class definitions and reduction framework.

Provides executable Python representations of core complexity theory
concepts: decision problems, complexity classes, polynomial-time
reductions, solver protocols, and empirical scaling analysis.

These definitions serve as the formal foundation for the P=NP research
exploration. They encode the mathematical definitions as closely as
possible while remaining executable and testable.

Note:
    The empirical analysis utilities (``is_polynomial``, ``measure_scaling``)
    provide heuristic evidence about runtime scaling. They are NOT formal
    proofs of complexity class membership. A polynomial curve fit on
    finite data cannot establish asymptotic complexity.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Decision Problem (Abstract Base)
# ---------------------------------------------------------------------------


class DecisionProblem(ABC):
    """Abstract base class for a formal decision problem.

    A decision problem is a function from instances to {True, False}.
    Subclasses must define how to encode instances, decide them, and
    verify a proposed certificate (witness) for "yes" instances.

    The separation between ``decide`` and ``verify`` captures the P vs NP
    distinction: P problems can be *decided* efficiently, while NP
    problems can at least be *verified* efficiently given a certificate.

    Attributes:
        name: Human-readable name of the decision problem.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name identifying this decision problem."""

    @abstractmethod
    def encode(self, instance: Any) -> str:
        """Encode a problem instance as a canonical string.

        Args:
            instance: A problem instance in its native representation.

        Returns:
            A string encoding of the instance suitable for measuring
            input size (len of the encoding).
        """

    @abstractmethod
    def decide(self, instance: Any) -> bool:
        """Decide whether the instance is a "yes" instance.

        This corresponds to the formal notion of a decision procedure.
        For NP-complete problems, this may require exponential time.

        Args:
            instance: A problem instance in its native representation.

        Returns:
            True if the instance is a "yes" instance, False otherwise.
        """

    @abstractmethod
    def verify(self, instance: Any, certificate: Any) -> bool:
        """Verify that a certificate proves the instance is a "yes" instance.

        For problems in NP, this verification must run in polynomial
        time with respect to the input size. The certificate (witness)
        is additional information that makes verification easy.

        Args:
            instance: A problem instance in its native representation.
            certificate: A proposed witness for a "yes" answer.

        Returns:
            True if the certificate is valid proof of a "yes" answer.
        """


# ---------------------------------------------------------------------------
# Complexity Classes
# ---------------------------------------------------------------------------


class ComplexityClass(Enum):
    """Enumeration of standard computational complexity classes.

    Each member represents a complexity class with its formal definition
    accessible via the ``description`` property.
    """

    P = "P"
    NP = "NP"
    NP_COMPLETE = "NP_COMPLETE"
    NP_HARD = "NP_HARD"
    UNKNOWN = "UNKNOWN"

    @property
    def description(self) -> str:
        """Formal definition of this complexity class.

        Returns:
            A string containing the formal definition.
        """
        return _CLASS_DESCRIPTIONS[self]


_CLASS_DESCRIPTIONS: dict[ComplexityClass, str] = {
    ComplexityClass.P: (
        "The class of decision problems solvable by a deterministic "
        "Turing machine in time O(n^k) for some constant k, where n "
        "is the input size."
    ),
    ComplexityClass.NP: (
        "The class of decision problems for which a 'yes' answer can "
        "be verified in polynomial time given a certificate (witness). "
        "Equivalently, solvable by a nondeterministic Turing machine "
        "in polynomial time."
    ),
    ComplexityClass.NP_COMPLETE: (
        "A decision problem L is NP-complete if (1) L is in NP, and "
        "(2) every problem in NP is polynomial-time reducible to L. "
        "These are the 'hardest' problems in NP. A polynomial-time "
        "algorithm for any NP-complete problem would imply P = NP."
    ),
    ComplexityClass.NP_HARD: (
        "A problem H is NP-hard if every problem in NP is polynomial-time "
        "reducible to H. NP-hard problems are at least as hard as the "
        "hardest problems in NP but need not be in NP themselves (they "
        "may not even be decision problems)."
    ),
    ComplexityClass.UNKNOWN: (
        "The complexity class of this problem has not been determined. "
        "This is the honest default for problems whose classification "
        "has not been formally established."
    ),
}


# ---------------------------------------------------------------------------
# Polynomial-Time Reduction
# ---------------------------------------------------------------------------


class PolynomialReduction(ABC):
    """A polynomial-time many-one reduction from problem A to problem B.

    A reduction transforms instances of problem A into instances of
    problem B such that the answer is preserved: instance_a is a "yes"
    instance of A if and only if reduce(instance_a) is a "yes" instance
    of B. The transformation must run in polynomial time.

    This is the formal mechanism behind NP-completeness proofs:
    showing that problem A reduces to problem B means B is at least
    as hard as A.

    Attributes:
        source_problem: The problem being reduced FROM (problem A).
        target_problem: The problem being reduced TO (problem B).
    """

    @property
    @abstractmethod
    def source_problem(self) -> DecisionProblem:
        """The source problem (A) of this reduction."""

    @property
    @abstractmethod
    def target_problem(self) -> DecisionProblem:
        """The target problem (B) of this reduction."""

    @abstractmethod
    def reduce(self, instance_a: Any) -> Any:
        """Transform an instance of problem A into an instance of problem B.

        The reduction must satisfy: A.decide(instance_a) == True if and
        only if B.decide(reduce(instance_a)) == True.

        Args:
            instance_a: An instance of the source problem.

        Returns:
            An instance of the target problem.
        """

    def verify_reduction(self, instance_a: Any) -> bool:
        """Verify that the reduction preserves the answer for a given instance.

        Checks that deciding the original instance and deciding the
        reduced instance yield the same answer. This is a correctness
        check, not a proof that the reduction is always correct.

        Args:
            instance_a: An instance of the source problem to check.

        Returns:
            True if the answer is preserved by the reduction.
        """
        answer_a = self.source_problem.decide(instance_a)
        instance_b = self.reduce(instance_a)
        answer_b = self.target_problem.decide(instance_b)
        return answer_a == answer_b

    def measure_reduction_time(self, instance_a: Any) -> tuple[Any, float]:
        """Time the reduction transformation on a single instance.

        Args:
            instance_a: An instance of the source problem.

        Returns:
            A tuple of (reduced_instance, elapsed_seconds).
        """
        start = time.perf_counter()
        instance_b = self.reduce(instance_a)
        elapsed = time.perf_counter() - start
        return instance_b, elapsed


# ---------------------------------------------------------------------------
# Solver Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Solver(Protocol):
    """Protocol defining the interface for a decision problem solver.

    Any object with ``solve``, ``name``, and ``complexity_claim``
    methods matching these signatures satisfies the Solver protocol.
    Using ``typing.Protocol`` allows structural subtyping: solvers
    need not inherit from this class, only conform to its shape.

    The ``solve`` method returns a certificate (witness) if the
    instance is satisfiable, or None if unsatisfiable.
    """

    def solve(self, instance: Any) -> Optional[Any]:
        """Attempt to solve a decision problem instance.

        Args:
            instance: A problem instance (e.g., a CNF formula).

        Returns:
            A certificate (witness) if the instance is a "yes"
            instance, or None if the instance is a "no" instance
            or no solution was found.
        """
        ...

    def name(self) -> str:
        """Human-readable name identifying this solver.

        Returns:
            The solver's name.
        """
        ...

    def complexity_claim(self) -> str:
        """State the claimed time complexity of this solver.

        This should be an honest statement like "O(2^n)" or
        "O(n^3) (claimed, not proven)". Experimental solvers
        should clearly indicate that their claims are unproven.

        Returns:
            A string describing the claimed complexity.
        """
        ...


# ---------------------------------------------------------------------------
# Solver Benchmark Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolverBenchmark:
    """Record of a single solver benchmark measurement.

    Captures the result of running a solver on one problem instance,
    including timing data for empirical scaling analysis.

    Attributes:
        solver_name: Name of the solver that produced this result.
        instance_size: Size of the problem instance (e.g., number
            of variables for SAT).
        elapsed_seconds: Wall-clock time in seconds for the solve call.
        result: Whether the solver found a satisfying assignment
            (True), determined unsatisfiability (False), or timed
            out / gave up (None).
        certificate: The certificate (witness) returned by the solver,
            or None if no certificate was produced.
    """

    solver_name: str
    instance_size: int
    elapsed_seconds: float
    result: Optional[bool]
    certificate: Optional[Any] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Empirical Scaling Analysis Utilities
# ---------------------------------------------------------------------------


def is_polynomial(
    times: list[float],
    sizes: list[int],
    max_degree: int = 6,
) -> tuple[bool, float, int]:
    """Heuristically determine if observed runtimes follow polynomial scaling.

    Fits polynomial curves of increasing degree and an exponential
    curve to the (size, time) data, then compares residual errors.
    If the best polynomial fit has lower residual error than the
    exponential fit, the scaling is classified as polynomial.

    IMPORTANT: This is an empirical heuristic, NOT a formal proof.
    Polynomial curve fits on finite data cannot establish asymptotic
    complexity class membership. Small instances may appear polynomial
    even for exponential algorithms due to constant factors and
    low-order terms dominating at small scales.

    Args:
        times: Observed runtimes (one per instance).
        sizes: Corresponding problem sizes (one per instance).
        max_degree: Maximum polynomial degree to try (default 6).

    Returns:
        A tuple of (is_poly, best_residual, best_degree) where:
        - is_poly: True if polynomial fit is better than exponential.
        - best_residual: Residual error of the best polynomial fit.
        - best_degree: Degree of the best-fitting polynomial.

    Raises:
        ValueError: If times and sizes have different lengths or
            fewer than 3 data points (minimum for meaningful fitting).
    """
    if len(times) != len(sizes):
        raise ValueError(
            f"times and sizes must have the same length, "
            f"got {len(times)} and {len(sizes)}"
        )
    if len(times) < 3:
        raise ValueError(
            f"Need at least 3 data points for curve fitting, got {len(times)}"
        )

    best_poly_residual = float("inf")
    best_degree = 1

    for degree in range(1, max_degree + 1):
        residual = _polynomial_residual(sizes, times, degree)
        if residual < best_poly_residual:
            best_poly_residual = residual
            best_degree = degree

    exp_residual = _exponential_residual(sizes, times)

    is_poly = best_poly_residual <= exp_residual
    return is_poly, best_poly_residual, best_degree


def _polynomial_residual(
    sizes: list[int], times: list[float], degree: int
) -> float:
    """Compute sum-of-squares residual for a polynomial fit.

    Uses a simple least-squares approach without numpy: constructs the
    Vandermonde system and solves via normal equations using Gaussian
    elimination. This keeps the core module dependency-free.

    Args:
        sizes: Problem sizes (x-values).
        times: Observed runtimes (y-values).
        degree: Polynomial degree to fit.

    Returns:
        Sum of squared residuals.
    """
    n = len(sizes)
    # Build Vandermonde matrix columns: [1, x, x^2, ..., x^degree]
    cols = degree + 1
    # Normal equations: (V^T V) c = V^T y
    vtv = [[0.0] * cols for _ in range(cols)]
    vty = [0.0] * cols

    for i in range(n):
        x = float(sizes[i])
        y = times[i]
        powers = [x**p for p in range(cols)]
        for r in range(cols):
            for c_idx in range(cols):
                vtv[r][c_idx] += powers[r] * powers[c_idx]
            vty[r] += powers[r] * y

    coeffs = _solve_linear_system(vtv, vty)
    if coeffs is None:
        return float("inf")

    residual = 0.0
    for i in range(n):
        x = float(sizes[i])
        predicted = sum(coeffs[p] * (x**p) for p in range(cols))
        residual += (times[i] - predicted) ** 2
    return residual


def _exponential_residual(sizes: list[int], times: list[float]) -> float:
    """Compute sum-of-squares residual for an exponential fit.

    Fits y = a * b^x by taking logarithms and performing linear
    regression on log(y) = log(a) + x * log(b). Data points with
    non-positive times are skipped.

    Args:
        sizes: Problem sizes (x-values).
        times: Observed runtimes (y-values).

    Returns:
        Sum of squared residuals in the original (non-log) space.
    """
    import math

    log_times = []
    valid_sizes = []
    for s, t in zip(sizes, times):
        if t > 0:
            log_times.append(math.log(t))
            valid_sizes.append(float(s))

    if len(log_times) < 2:
        return float("inf")

    n = len(log_times)
    sum_x = sum(valid_sizes)
    sum_y = sum(log_times)
    sum_xx = sum(x * x for x in valid_sizes)
    sum_xy = sum(x * y for x, y in zip(valid_sizes, log_times))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-15:
        return float("inf")

    log_b = (n * sum_xy - sum_x * sum_y) / denom
    log_a = (sum_y - log_b * sum_x) / n

    a = math.exp(log_a)
    b = math.exp(log_b)

    residual = 0.0
    for s, t in zip(sizes, times):
        predicted = a * (b ** float(s))
        residual += (t - predicted) ** 2
    return residual


def _solve_linear_system(
    matrix: list[list[float]], rhs: list[float]
) -> Optional[list[float]]:
    """Solve a linear system Ax = b via Gaussian elimination with pivoting.

    Args:
        matrix: Square coefficient matrix (will be modified in place).
        rhs: Right-hand side vector (will be modified in place).

    Returns:
        Solution vector, or None if the system is singular.
    """
    n = len(rhs)
    # Make copies to avoid mutating caller's data
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
        if max_val < 1e-15:
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
        if abs(a[i][i]) < 1e-15:
            return None
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= a[i][j] * x[j]
        x[i] /= a[i][i]

    return x


def measure_scaling(
    solver: Solver,
    instances: list[Any],
    sizes: list[int],
) -> list[SolverBenchmark]:
    """Benchmark a solver on instances of increasing size.

    Runs the solver on each instance and records timing data.
    The resulting benchmarks can be fed to ``is_polynomial`` for
    empirical scaling analysis.

    Args:
        solver: A solver conforming to the Solver protocol.
        instances: Problem instances ordered by increasing size.
        sizes: The size of each corresponding instance.

    Returns:
        A list of SolverBenchmark records, one per instance.

    Raises:
        ValueError: If instances and sizes have different lengths.
    """
    if len(instances) != len(sizes):
        raise ValueError(
            f"instances and sizes must have the same length, "
            f"got {len(instances)} and {len(sizes)}"
        )

    solver_name = solver.name()
    benchmarks: list[SolverBenchmark] = []

    for instance, size in zip(instances, sizes):
        start = time.perf_counter()
        certificate = solver.solve(instance)
        elapsed = time.perf_counter() - start

        result: Optional[bool] = certificate is not None

        benchmarks.append(
            SolverBenchmark(
                solver_name=solver_name,
                instance_size=size,
                elapsed_seconds=elapsed,
                result=result,
                certificate=certificate,
            )
        )

    return benchmarks
