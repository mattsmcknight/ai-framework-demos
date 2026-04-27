"""Tests for p_equals_np.definitions module.

Covers: DecisionProblem abstraction, ComplexityClass enum,
SolverBenchmark dataclass, is_polynomial heuristic, and measure_scaling.
"""

from __future__ import annotations

import pytest
from typing import Any, Optional

from p_equals_np.definitions import (
    ComplexityClass,
    DecisionProblem,
    PolynomialReduction,
    Solver,
    SolverBenchmark,
    is_polynomial,
    measure_scaling,
)


# ---------------------------------------------------------------------------
# DecisionProblem
# ---------------------------------------------------------------------------


class TestDecisionProblem:
    """Tests for the DecisionProblem abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """DecisionProblem is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            DecisionProblem()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        """A concrete subclass implementing all abstract methods works."""

        class TrivialProblem(DecisionProblem):
            @property
            def name(self) -> str:
                return "Trivial"

            def encode(self, instance: Any) -> str:
                return str(instance)

            def decide(self, instance: Any) -> bool:
                return bool(instance)

            def verify(self, instance: Any, certificate: Any) -> bool:
                return bool(certificate)

        problem = TrivialProblem()
        assert problem.name == "Trivial"
        assert problem.decide(True) is True
        assert problem.decide(False) is False
        assert problem.verify(None, True) is True

    def test_partial_subclass_raises(self) -> None:
        """A subclass missing abstract methods cannot be instantiated."""

        class PartialProblem(DecisionProblem):
            @property
            def name(self) -> str:
                return "Partial"

            def encode(self, instance: Any) -> str:
                return ""

            # Missing decide and verify

        with pytest.raises(TypeError):
            PartialProblem()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# ComplexityClass
# ---------------------------------------------------------------------------


class TestComplexityClass:
    """Tests for the ComplexityClass enum."""

    def test_has_expected_members(self) -> None:
        """All five complexity classes exist."""
        expected = {"P", "NP", "NP_COMPLETE", "NP_HARD", "UNKNOWN"}
        actual = {member.name for member in ComplexityClass}
        assert actual == expected

    def test_p_value(self) -> None:
        """P has value 'P'."""
        assert ComplexityClass.P.value == "P"

    def test_np_complete_value(self) -> None:
        """NP_COMPLETE has value 'NP_COMPLETE'."""
        assert ComplexityClass.NP_COMPLETE.value == "NP_COMPLETE"

    def test_description_not_empty(self) -> None:
        """Every complexity class has a non-empty description."""
        for cls in ComplexityClass:
            assert len(cls.description) > 0, f"{cls.name} has empty description"

    def test_p_description_mentions_key_concept(self) -> None:
        """P description mentions O(n^k) or deterministic Turing machine."""
        desc = ComplexityClass.P.description.lower()
        assert "n^k" in desc or "deterministic" in desc

    def test_np_complete_description_mentions_reducible(self) -> None:
        """NP_COMPLETE description mentions reducibility."""
        assert "reducible" in ComplexityClass.NP_COMPLETE.description.lower()


# ---------------------------------------------------------------------------
# SolverBenchmark
# ---------------------------------------------------------------------------


class TestSolverBenchmark:
    """Tests for the SolverBenchmark frozen dataclass."""

    def test_fields_accessible(self) -> None:
        """All expected fields are accessible."""
        bm = SolverBenchmark(
            solver_name="TestSolver",
            instance_size=10,
            elapsed_seconds=0.5,
            result=True,
            certificate={1: True},
        )
        assert bm.solver_name == "TestSolver"
        assert bm.instance_size == 10
        assert bm.elapsed_seconds == 0.5
        assert bm.result is True
        assert bm.certificate == {1: True}

    def test_frozen(self) -> None:
        """SolverBenchmark is immutable (frozen dataclass)."""
        bm = SolverBenchmark(
            solver_name="Test",
            instance_size=5,
            elapsed_seconds=0.1,
            result=False,
        )
        with pytest.raises(AttributeError):
            bm.solver_name = "Modified"  # type: ignore[misc]

    def test_certificate_defaults_to_none(self) -> None:
        """Certificate defaults to None when not provided."""
        bm = SolverBenchmark(
            solver_name="Test",
            instance_size=5,
            elapsed_seconds=0.1,
            result=None,
        )
        assert bm.certificate is None

    def test_result_can_be_none(self) -> None:
        """Result can be None (timeout / gave up)."""
        bm = SolverBenchmark(
            solver_name="Test",
            instance_size=5,
            elapsed_seconds=30.0,
            result=None,
        )
        assert bm.result is None


# ---------------------------------------------------------------------------
# is_polynomial
# ---------------------------------------------------------------------------


class TestIsPolynomial:
    """Tests for the is_polynomial heuristic."""

    def test_polynomial_data_detected(self) -> None:
        """Data following y = x^2 is classified as polynomial."""
        sizes = [10, 20, 30, 40, 50, 60, 70, 80]
        times = [float(s * s) for s in sizes]

        is_poly, residual, degree = is_polynomial(times, sizes)
        assert is_poly is True
        assert degree <= 6

    def test_exponential_data_detected(self) -> None:
        """Data following y = 2^x is classified as exponential."""
        sizes = [5, 10, 15, 20, 25, 30]
        times = [2.0**s for s in sizes]

        is_poly, residual, degree = is_polynomial(times, sizes)
        assert is_poly is False

    def test_mismatched_lengths_raises(self) -> None:
        """Raises ValueError when times and sizes have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            is_polynomial([1.0, 2.0], [1, 2, 3])

    def test_too_few_points_raises(self) -> None:
        """Raises ValueError with fewer than 3 data points."""
        with pytest.raises(ValueError, match="at least 3"):
            is_polynomial([1.0, 2.0], [1, 2])

    def test_linear_data(self) -> None:
        """Linear data y = 3x + 1 is classified as polynomial."""
        sizes = [10, 20, 30, 40, 50]
        times = [3.0 * s + 1.0 for s in sizes]

        is_poly, _residual, degree = is_polynomial(times, sizes)
        assert is_poly is True

    def test_returns_best_degree(self) -> None:
        """The returned degree matches the shape of the data."""
        sizes = [5, 10, 15, 20, 25, 30, 35, 40]
        # Cubic data: y = x^3
        times = [float(s**3) for s in sizes]

        is_poly, _residual, degree = is_polynomial(times, sizes)
        assert is_poly is True
        # Best-fit degree should be around 3 (allow some tolerance)
        assert degree >= 2


# ---------------------------------------------------------------------------
# measure_scaling
# ---------------------------------------------------------------------------


class TestMeasureScaling:
    """Tests for the measure_scaling function."""

    def test_returns_benchmarks(self) -> None:
        """measure_scaling returns a list of SolverBenchmark objects."""

        class DummySolver:
            def solve(self, instance: Any) -> Optional[Any]:
                return {"answer": True}

            def name(self) -> str:
                return "Dummy"

            def complexity_claim(self) -> str:
                return "O(1)"

        solver = DummySolver()
        instances = ["a", "ab", "abc"]
        sizes = [1, 2, 3]

        benchmarks = measure_scaling(solver, instances, sizes)

        assert len(benchmarks) == 3
        for bm in benchmarks:
            assert isinstance(bm, SolverBenchmark)
            assert bm.solver_name == "Dummy"
            assert bm.result is True  # certificate is not None
            assert bm.elapsed_seconds >= 0.0

    def test_instance_sizes_recorded(self) -> None:
        """Instance sizes match what was passed in."""

        class DummySolver:
            def solve(self, instance: Any) -> Optional[Any]:
                return None

            def name(self) -> str:
                return "Dummy"

            def complexity_claim(self) -> str:
                return "O(1)"

        solver = DummySolver()
        sizes = [5, 10, 15]
        benchmarks = measure_scaling(solver, ["x"] * 3, sizes)

        for bm, expected_size in zip(benchmarks, sizes):
            assert bm.instance_size == expected_size

    def test_mismatched_lengths_raises(self) -> None:
        """Raises ValueError when instances and sizes differ in length."""

        class DummySolver:
            def solve(self, instance: Any) -> Optional[Any]:
                return None

            def name(self) -> str:
                return "Dummy"

            def complexity_claim(self) -> str:
                return "O(1)"

        with pytest.raises(ValueError, match="same length"):
            measure_scaling(DummySolver(), [1, 2], [1])

    def test_unsatisfiable_result_is_false(self) -> None:
        """When solver returns None, result is False."""

        class UnsatSolver:
            def solve(self, instance: Any) -> Optional[Any]:
                return None

            def name(self) -> str:
                return "Unsat"

            def complexity_claim(self) -> str:
                return "O(1)"

        benchmarks = measure_scaling(UnsatSolver(), ["x"], [1])
        assert benchmarks[0].result is False
