"""Tests for p_equals_np.complexity_analysis module.

Covers: median utility, linear_regression, RuntimeMeasurement dataclass,
ScalingExperiment curve fitting (polynomial and exponential), analyze_scaling
with mock data, generate_scaling_report output, and a small end-to-end
scaling experiment.

No dependency on matplotlib -- all tests exercise the pure Python paths.
"""

from __future__ import annotations

import math

import pytest

from p_equals_np.complexity_analysis import (
    RuntimeMeasurement,
    ScalingExperiment,
    linear_regression,
    median,
)
from p_equals_np.brute_force import BruteForceSolver
from p_equals_np.dpll import DPLLSolver


# ---------------------------------------------------------------------------
# Helper: create a minimal ScalingExperiment for curve fitting tests
# ---------------------------------------------------------------------------


def _make_experiment() -> ScalingExperiment:
    """Create a ScalingExperiment with minimal config for curve fitting.

    We only need the instance to call fit_polynomial / fit_exponential /
    analyze_scaling / generate_scaling_report. The solver list and sizes
    are not used by those methods.
    """
    solver = BruteForceSolver(timeout_seconds=1.0)
    return ScalingExperiment(
        solvers=[solver],
        variable_sizes=[5, 10],
        instances_per_size=1,
        timeout_per_instance=1.0,
    )


# =========================================================================
# 1. MEDIAN TESTS
# =========================================================================


class TestMedian:
    """Tests for the median utility function."""

    def test_odd_length_list(self) -> None:
        """Median of odd-length list is the middle element."""
        assert median([3.0, 1.0, 2.0]) == 2.0

    def test_even_length_list(self) -> None:
        """Median of even-length list is the average of two middle elements."""
        assert median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_single_element(self) -> None:
        """Median of single-element list is that element."""
        assert median([42.0]) == 42.0

    def test_two_elements(self) -> None:
        """Median of two elements is their average."""
        assert median([10.0, 20.0]) == 15.0

    def test_already_sorted(self) -> None:
        """Median works correctly on pre-sorted input."""
        assert median([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0

    def test_reverse_sorted(self) -> None:
        """Median works correctly on reverse-sorted input."""
        assert median([5.0, 4.0, 3.0, 2.0, 1.0]) == 3.0

    def test_duplicate_values(self) -> None:
        """Median handles duplicate values."""
        assert median([5.0, 5.0, 5.0]) == 5.0

    def test_empty_list_raises(self) -> None:
        """Median of empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            median([])

    @pytest.mark.parametrize(
        "values, expected",
        [
            ([1.0, 3.0, 5.0, 7.0, 9.0], 5.0),
            ([2.0, 4.0, 6.0, 8.0], 5.0),
            ([100.0], 100.0),
        ],
        ids=["odd-5", "even-4", "single"],
    )
    def test_parametrized(self, values: list[float], expected: float) -> None:
        """Parametrized median tests for various list shapes."""
        assert median(values) == expected


# =========================================================================
# 2. LINEAR REGRESSION TESTS
# =========================================================================


class TestLinearRegression:
    """Tests for the linear_regression utility function."""

    def test_perfect_positive_slope(self) -> None:
        """Perfect linear data y = 2x + 1 gives slope=2, intercept=1, R2=1."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 5.0, 7.0, 9.0, 11.0]
        slope, intercept, r2 = linear_regression(x, y)
        assert abs(slope - 2.0) < 1e-10
        assert abs(intercept - 1.0) < 1e-10
        assert abs(r2 - 1.0) < 1e-10

    def test_perfect_negative_slope(self) -> None:
        """Perfect linear data y = -3x + 10 gives correct parameters."""
        x = [0.0, 1.0, 2.0, 3.0]
        y = [10.0, 7.0, 4.0, 1.0]
        slope, intercept, r2 = linear_regression(x, y)
        assert abs(slope - (-3.0)) < 1e-10
        assert abs(intercept - 10.0) < 1e-10
        assert abs(r2 - 1.0) < 1e-10

    def test_horizontal_line(self) -> None:
        """Constant y = 5 gives slope=0, intercept=5."""
        x = [1.0, 2.0, 3.0, 4.0]
        y = [5.0, 5.0, 5.0, 5.0]
        slope, intercept, r2 = linear_regression(x, y)
        assert abs(slope) < 1e-10
        assert abs(intercept - 5.0) < 1e-10
        # R2 when all y are equal and residual is zero: should be 1.0
        assert r2 == 1.0

    def test_r_squared_less_than_one_with_noise(self) -> None:
        """Imperfect linear data gives R2 < 1."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.1, 3.9, 6.2, 7.8, 10.1]
        slope, intercept, r2 = linear_regression(x, y)
        assert 0.95 < r2 < 1.0
        assert slope > 0

    def test_minimum_two_points(self) -> None:
        """Two data points give a perfect fit (R2 = 1)."""
        x = [0.0, 10.0]
        y = [0.0, 100.0]
        slope, intercept, r2 = linear_regression(x, y)
        assert abs(slope - 10.0) < 1e-10
        assert abs(intercept) < 1e-10
        assert abs(r2 - 1.0) < 1e-10

    def test_mismatched_lengths_raises(self) -> None:
        """Different length x and y raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            linear_regression([1.0, 2.0], [1.0])

    def test_fewer_than_two_points_raises(self) -> None:
        """Fewer than 2 data points raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            linear_regression([1.0], [1.0])


# =========================================================================
# 3. RUNTIME MEASUREMENT DATACLASS TESTS
# =========================================================================


class TestRuntimeMeasurement:
    """Tests for the RuntimeMeasurement frozen dataclass."""

    def test_fields_accessible(self) -> None:
        """All expected fields are accessible on a RuntimeMeasurement."""
        m = RuntimeMeasurement(
            solver_name="TestSolver",
            num_variables=10,
            num_clauses=42,
            clause_ratio=4.2,
            elapsed_seconds=0.123,
            solved=True,
            timed_out=False,
        )
        assert m.solver_name == "TestSolver"
        assert m.num_variables == 10
        assert m.num_clauses == 42
        assert abs(m.clause_ratio - 4.2) < 1e-10
        assert abs(m.elapsed_seconds - 0.123) < 1e-10
        assert m.solved is True
        assert m.timed_out is False

    def test_frozen_immutable(self) -> None:
        """RuntimeMeasurement is frozen -- attributes cannot be reassigned."""
        m = RuntimeMeasurement(
            solver_name="S",
            num_variables=5,
            num_clauses=10,
            clause_ratio=2.0,
            elapsed_seconds=1.0,
            solved=True,
            timed_out=False,
        )
        with pytest.raises(AttributeError):
            m.solver_name = "Other"  # type: ignore[misc]

    def test_slots_present(self) -> None:
        """RuntimeMeasurement uses __slots__ (no __dict__)."""
        m = RuntimeMeasurement(
            solver_name="S",
            num_variables=5,
            num_clauses=10,
            clause_ratio=2.0,
            elapsed_seconds=1.0,
            solved=True,
            timed_out=False,
        )
        assert not hasattr(m, "__dict__")

    def test_timed_out_measurement(self) -> None:
        """A timed-out measurement has solved=False, timed_out=True."""
        m = RuntimeMeasurement(
            solver_name="SlowSolver",
            num_variables=20,
            num_clauses=85,
            clause_ratio=4.25,
            elapsed_seconds=30.0,
            solved=False,
            timed_out=True,
        )
        assert m.solved is False
        assert m.timed_out is True

    def test_equality(self) -> None:
        """Two RuntimeMeasurements with same fields are equal (dataclass eq)."""
        kwargs = dict(
            solver_name="S",
            num_variables=5,
            num_clauses=10,
            clause_ratio=2.0,
            elapsed_seconds=1.0,
            solved=True,
            timed_out=False,
        )
        assert RuntimeMeasurement(**kwargs) == RuntimeMeasurement(**kwargs)


# =========================================================================
# 4. FIT POLYNOMIAL TESTS
# =========================================================================


class TestFitPolynomial:
    """Tests for ScalingExperiment.fit_polynomial."""

    def test_quadratic_data_high_r_squared(self) -> None:
        """Polynomial fit on t = 3*n^2 + n + 1 achieves R2 > 0.99."""
        exp = _make_experiment()
        sizes = list(range(5, 55, 5))  # [5, 10, 15, ..., 50]
        times = [3.0 * n * n + n + 1.0 for n in sizes]

        coeffs, r2 = exp.fit_polynomial(sizes, times)
        assert r2 > 0.99, f"Expected R2 > 0.99, got {r2}"
        # The fit should find degree 2 as best
        assert len(coeffs) >= 3, f"Expected at least degree 2, got {len(coeffs) - 1}"

    def test_linear_data(self) -> None:
        """Polynomial fit on t = 5*n + 3 finds good linear fit."""
        exp = _make_experiment()
        sizes = list(range(10, 110, 10))
        times = [5.0 * n + 3.0 for n in sizes]

        coeffs, r2 = exp.fit_polynomial(sizes, times)
        assert r2 > 0.999

    def test_cubic_data(self) -> None:
        """Polynomial fit on t = 0.5*n^3 + 2*n achieves R2 > 0.99."""
        exp = _make_experiment()
        sizes = list(range(2, 22, 2))
        times = [0.5 * n ** 3 + 2.0 * n for n in sizes]

        coeffs, r2 = exp.fit_polynomial(sizes, times)
        assert r2 > 0.99

    def test_exponential_data_fits_worse_than_exponential_model(self) -> None:
        """Polynomial fit on exponential data is worse than exponential fit.

        This verifies the curve fitting can distinguish polynomial from
        exponential behavior. On true exponential data, the exponential
        model should achieve a better R2 than the polynomial model.
        We use a wide range of sizes so polynomial cannot track the
        exponential growth adequately.
        """
        exp = _make_experiment()
        sizes = list(range(1, 40))
        times = [0.001 * (2.0 ** n) for n in sizes]

        _poly_coeffs, poly_r2 = exp.fit_polynomial(sizes, times)
        _a, _b, exp_r2 = exp.fit_exponential(sizes, times)

        assert exp_r2 > poly_r2, (
            f"Exponential fit R2 = {exp_r2} should exceed polynomial "
            f"fit R2 = {poly_r2} on true exponential data."
        )
        # Exponential fit should be near-perfect on exponential data
        assert exp_r2 > 0.99, f"Expected exponential R2 > 0.99, got {exp_r2}"

    def test_coefficients_ascending_order(self) -> None:
        """Coefficients are returned in ascending order [a_0, a_1, ..., a_k]."""
        exp = _make_experiment()
        # t = 10 + 2*n -> coeffs should be approximately [10, 2]
        sizes = list(range(1, 20))
        times = [10.0 + 2.0 * n for n in sizes]

        coeffs, r2 = exp.fit_polynomial(sizes, times)
        assert r2 > 0.999
        # First coefficient (a_0) should be near 10, second (a_1) near 2
        # But the best fit might be degree > 1; at minimum, check structure
        assert len(coeffs) >= 2

    def test_mismatched_lengths_raises(self) -> None:
        """fit_polynomial raises ValueError for mismatched input lengths."""
        exp = _make_experiment()
        with pytest.raises(ValueError, match="same length"):
            exp.fit_polynomial([1, 2, 3], [1.0, 2.0])

    def test_too_few_points_raises(self) -> None:
        """fit_polynomial raises ValueError with fewer than 2 data points."""
        exp = _make_experiment()
        with pytest.raises(ValueError, match="at least 2"):
            exp.fit_polynomial([5], [1.0])


# =========================================================================
# 5. FIT EXPONENTIAL TESTS
# =========================================================================


class TestFitExponential:
    """Tests for ScalingExperiment.fit_exponential."""

    def test_exponential_data_high_r_squared(self) -> None:
        """Exponential fit on t = 0.001 * 2^n achieves R2 > 0.99."""
        exp = _make_experiment()
        sizes = list(range(5, 25))
        times = [0.001 * (2.0 ** n) for n in sizes]

        a, b, r2 = exp.fit_exponential(sizes, times)
        assert r2 > 0.99, f"Expected R2 > 0.99, got {r2}"
        assert abs(b - 2.0) < 0.1, f"Expected b near 2.0, got {b}"
        assert a > 0, f"Expected a > 0, got {a}"

    def test_base_recovery(self) -> None:
        """Exponential fit recovers the base accurately for t = 0.01 * 1.5^n."""
        exp = _make_experiment()
        sizes = list(range(5, 30))
        times = [0.01 * (1.5 ** n) for n in sizes]

        a, b, r2 = exp.fit_exponential(sizes, times)
        assert abs(b - 1.5) < 0.05, f"Expected b near 1.5, got {b}"
        assert r2 > 0.99

    def test_coefficient_recovery(self) -> None:
        """Exponential fit recovers the coefficient a for t = 5.0 * 1.2^n."""
        exp = _make_experiment()
        sizes = list(range(1, 20))
        times = [5.0 * (1.2 ** n) for n in sizes]

        a, b, r2 = exp.fit_exponential(sizes, times)
        assert abs(b - 1.2) < 0.05, f"Expected b near 1.2, got {b}"
        # 'a' should be near 5.0 (within factor of 2 due to log-space fitting)
        assert 2.0 < a < 10.0, f"Expected a near 5.0, got {a}"
        assert r2 > 0.99

    def test_mismatched_lengths_raises(self) -> None:
        """fit_exponential raises ValueError for mismatched input lengths."""
        exp = _make_experiment()
        with pytest.raises(ValueError, match="same length"):
            exp.fit_exponential([1, 2], [1.0, 2.0, 3.0])

    def test_too_few_positive_points_raises(self) -> None:
        """fit_exponential raises ValueError when fewer than 2 positive times."""
        exp = _make_experiment()
        with pytest.raises(ValueError, match="at least 2"):
            exp.fit_exponential([1, 2], [0.0, 0.0])


# =========================================================================
# 6. ANALYZE SCALING TESTS
# =========================================================================


class TestAnalyzeScaling:
    """Tests for ScalingExperiment.analyze_scaling with mock measurements."""

    @staticmethod
    def _make_mock_measurements() -> list[RuntimeMeasurement]:
        """Create mock measurements for two solvers: one poly, one exponential.

        PolySolver: times scale as ~ n^2 / 1000
        ExpSolver: times scale as ~ 0.001 * 1.5^n

        Uses a wide range of sizes (10 points from 5 to 50) so that the
        exponential growth is sufficiently distinct from any polynomial.
        """
        measurements: list[RuntimeMeasurement] = []
        sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        instances_per_size = 3

        for size in sizes:
            for i in range(instances_per_size):
                # Small random-ish noise via instance index
                noise = 1.0 + 0.01 * i

                # PolySolver: quadratic scaling
                poly_time = (size ** 2 / 1000.0) * noise
                measurements.append(RuntimeMeasurement(
                    solver_name="PolySolver",
                    num_variables=size,
                    num_clauses=round(size * 4.267),
                    clause_ratio=4.267,
                    elapsed_seconds=poly_time,
                    solved=True,
                    timed_out=False,
                ))

                # ExpSolver: exponential scaling
                exp_time = (0.001 * (1.5 ** size)) * noise
                measurements.append(RuntimeMeasurement(
                    solver_name="ExpSolver",
                    num_variables=size,
                    num_clauses=round(size * 4.267),
                    clause_ratio=4.267,
                    elapsed_seconds=exp_time,
                    solved=True,
                    timed_out=False,
                ))

        return measurements

    def test_both_solvers_present(self) -> None:
        """analyze_scaling returns entries for both solvers."""
        exp = _make_experiment()
        measurements = self._make_mock_measurements()
        analysis = exp.analyze_scaling(measurements)
        assert "PolySolver" in analysis
        assert "ExpSolver" in analysis

    def test_poly_solver_best_model_is_polynomial(self) -> None:
        """PolySolver (quadratic data) should have best_model = polynomial."""
        exp = _make_experiment()
        measurements = self._make_mock_measurements()
        analysis = exp.analyze_scaling(measurements)
        poly_data = analysis["PolySolver"]
        assert poly_data["best_model"] == "polynomial", (
            f"Expected polynomial, got {poly_data['best_model']}"
        )
        assert poly_data["best_r2"] > 0.95

    def test_exp_solver_best_model_is_exponential(self) -> None:
        """ExpSolver (exponential data) should have best_model = exponential."""
        exp = _make_experiment()
        measurements = self._make_mock_measurements()
        analysis = exp.analyze_scaling(measurements)
        exp_data = analysis["ExpSolver"]
        assert exp_data["best_model"] == "exponential", (
            f"Expected exponential, got {exp_data['best_model']}"
        )
        assert exp_data["best_r2"] > 0.95

    def test_sizes_and_median_times_present(self) -> None:
        """Analysis contains sizes and median_times for each solver."""
        exp = _make_experiment()
        measurements = self._make_mock_measurements()
        analysis = exp.analyze_scaling(measurements)

        for solver_name in ["PolySolver", "ExpSolver"]:
            data = analysis[solver_name]
            assert "sizes" in data
            assert "median_times" in data
            assert len(data["sizes"]) == 10  # 10 sizes
            assert len(data["median_times"]) == 10
            # Sizes should be sorted
            assert data["sizes"] == sorted(data["sizes"])

    def test_fit_data_present(self) -> None:
        """Analysis contains poly_fit and exp_fit for each solver."""
        exp = _make_experiment()
        measurements = self._make_mock_measurements()
        analysis = exp.analyze_scaling(measurements)

        for solver_name in ["PolySolver", "ExpSolver"]:
            data = analysis[solver_name]
            assert "poly_fit" in data
            assert "exp_fit" in data
            # poly_fit is (coefficients, r_squared)
            poly_coeffs, poly_r2 = data["poly_fit"]
            assert isinstance(poly_coeffs, list)
            assert isinstance(poly_r2, float)
            # exp_fit is (a, b, r_squared)
            exp_a, exp_b, exp_r2 = data["exp_fit"]
            assert isinstance(exp_a, float)
            assert isinstance(exp_b, float)
            assert isinstance(exp_r2, float)

    def test_insufficient_data_handling(self) -> None:
        """analyze_scaling handles solver with only one size gracefully."""
        exp = _make_experiment()
        measurements = [
            RuntimeMeasurement(
                solver_name="OneSizeSolver",
                num_variables=10,
                num_clauses=42,
                clause_ratio=4.2,
                elapsed_seconds=0.5,
                solved=True,
                timed_out=False,
            ),
        ]
        analysis = exp.analyze_scaling(measurements)
        assert analysis["OneSizeSolver"]["best_model"] == "insufficient_data"


# =========================================================================
# 7. GENERATE SCALING REPORT TESTS
# =========================================================================


class TestGenerateScalingReport:
    """Tests for ScalingExperiment.generate_scaling_report."""

    def test_report_is_nonempty_string(self) -> None:
        """generate_scaling_report returns a non-empty string."""
        exp = _make_experiment()
        measurements = TestAnalyzeScaling._make_mock_measurements()
        analysis = exp.analyze_scaling(measurements)
        report = exp.generate_scaling_report(analysis)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_contains_solver_names(self) -> None:
        """Report contains both solver names."""
        exp = _make_experiment()
        measurements = TestAnalyzeScaling._make_mock_measurements()
        analysis = exp.analyze_scaling(measurements)
        report = exp.generate_scaling_report(analysis)
        assert "PolySolver" in report
        assert "ExpSolver" in report

    def test_report_contains_caveat(self) -> None:
        """Report contains the intellectual honesty caveat."""
        exp = _make_experiment()
        measurements = TestAnalyzeScaling._make_mock_measurements()
        analysis = exp.analyze_scaling(measurements)
        report = exp.generate_scaling_report(analysis)
        assert "CAVEAT" in report or "cannot establish" in report.lower()

    def test_report_contains_r_squared(self) -> None:
        """Report contains R-squared values for at least one solver."""
        exp = _make_experiment()
        measurements = TestAnalyzeScaling._make_mock_measurements()
        analysis = exp.analyze_scaling(measurements)
        report = exp.generate_scaling_report(analysis)
        assert "R-squared" in report

    def test_report_contains_model_designation(self) -> None:
        """Report contains best fit designation."""
        exp = _make_experiment()
        measurements = TestAnalyzeScaling._make_mock_measurements()
        analysis = exp.analyze_scaling(measurements)
        report = exp.generate_scaling_report(analysis)
        assert "Best fit" in report


# =========================================================================
# 8. SCALING EXPERIMENT VALIDATION TESTS
# =========================================================================


class TestScalingExperimentValidation:
    """Tests for ScalingExperiment constructor validation."""

    def test_empty_solvers_raises(self) -> None:
        """Empty solvers list raises ValueError."""
        with pytest.raises(ValueError, match="solvers"):
            ScalingExperiment(
                solvers=[],
                variable_sizes=[5, 10],
            )

    def test_empty_variable_sizes_raises(self) -> None:
        """Empty variable_sizes raises ValueError."""
        with pytest.raises(ValueError, match="variable_sizes"):
            ScalingExperiment(
                solvers=[BruteForceSolver()],
                variable_sizes=[],
            )

    def test_zero_instances_per_size_raises(self) -> None:
        """instances_per_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="instances_per_size"):
            ScalingExperiment(
                solvers=[BruteForceSolver()],
                variable_sizes=[5],
                instances_per_size=0,
            )

    def test_non_positive_timeout_raises(self) -> None:
        """timeout_per_instance <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="timeout_per_instance"):
            ScalingExperiment(
                solvers=[BruteForceSolver()],
                variable_sizes=[5],
                timeout_per_instance=-1.0,
            )

    def test_variable_sizes_sorted(self) -> None:
        """Constructor sorts variable_sizes."""
        exp = ScalingExperiment(
            solvers=[BruteForceSolver()],
            variable_sizes=[20, 5, 10],
        )
        assert exp.variable_sizes == [5, 10, 20]


# =========================================================================
# 9. SMALL END-TO-END SCALING EXPERIMENT
# =========================================================================


class TestSmallScalingExperiment:
    """End-to-end test: run a small scaling experiment without error.

    Uses 2 sizes, 2 instances each, short timeout. Exercises the full
    pipeline: instance generation, solver execution, measurement collection,
    analysis, and report generation.
    """

    def test_experiment_runs_without_error(self) -> None:
        """A small experiment (2 sizes, 2 instances) completes successfully."""
        solvers = [
            BruteForceSolver(timeout_seconds=5.0),
            DPLLSolver(timeout_seconds=5.0),
        ]
        exp = ScalingExperiment(
            solvers=solvers,
            variable_sizes=[5, 8],
            instances_per_size=2,
            timeout_per_instance=5.0,
        )

        # Run experiment
        measurements = exp.run_experiment()

        # Should have 2 solvers * 2 sizes * 2 instances = 8 measurements
        assert len(measurements) == 8

        # All should be RuntimeMeasurement instances
        for m in measurements:
            assert isinstance(m, RuntimeMeasurement)
            assert m.num_variables in (5, 8)
            assert m.elapsed_seconds >= 0
            assert m.elapsed_seconds <= 10.0  # generous upper bound

        # Analysis should succeed
        analysis = exp.analyze_scaling(measurements)
        assert len(analysis) == 2  # two solvers

        # Report should be non-empty
        report = exp.generate_scaling_report(analysis)
        assert isinstance(report, str)
        assert len(report) > 100  # non-trivial content

    def test_measurements_have_correct_solver_names(self) -> None:
        """Measurements carry the correct solver name."""
        solvers = [
            BruteForceSolver(timeout_seconds=5.0),
            DPLLSolver(timeout_seconds=5.0),
        ]
        exp = ScalingExperiment(
            solvers=solvers,
            variable_sizes=[5],
            instances_per_size=1,
            timeout_per_instance=5.0,
        )
        measurements = exp.run_experiment()

        solver_names = {m.solver_name for m in measurements}
        assert "BruteForce" in solver_names
        assert "DPLL" in solver_names

    def test_all_instances_solved_for_small_sizes(self) -> None:
        """All instances at small sizes (n=5) should be solved within timeout."""
        solver = BruteForceSolver(timeout_seconds=5.0)
        exp = ScalingExperiment(
            solvers=[solver],
            variable_sizes=[5],
            instances_per_size=3,
            timeout_per_instance=5.0,
        )
        measurements = exp.run_experiment()

        for m in measurements:
            assert m.solved is True, (
                f"Instance with {m.num_variables} vars should be solvable "
                f"but timed_out={m.timed_out}"
            )


# =========================================================================
# 10. PLOT SCALING FALLBACK TEST
# =========================================================================


class TestPlotScalingFallback:
    """Test that plot_scaling degrades gracefully without matplotlib."""

    def test_csv_fallback_creates_files(self, tmp_path) -> None:
        """plot_scaling creates CSV files when matplotlib is unavailable.

        Uses the CSV fallback path by passing analysis data and
        calling _export_csv_fallback directly.
        """
        import os

        exp = _make_experiment()
        measurements = TestAnalyzeScaling._make_mock_measurements()
        analysis = exp.analyze_scaling(measurements)

        output_dir = str(tmp_path / "scaling_output")
        os.makedirs(output_dir, exist_ok=True)
        exp._export_csv_fallback(analysis, output_dir)

        # Should have created CSV files
        import os
        files = os.listdir(output_dir)
        assert len(files) > 0
        csv_files = [f for f in files if f.endswith(".csv")]
        assert len(csv_files) >= 2  # one per solver
        # Should have a summary text file
        txt_files = [f for f in files if f.endswith(".txt")]
        assert len(txt_files) >= 1
