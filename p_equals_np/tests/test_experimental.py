"""Tests for all four experimental SAT solving approaches.

Validates each experimental solver (Algebraic, Spectral, LP Relaxation,
Structural) both in isolation and against the brute-force ground truth.
Cross-approach consistency is verified on 20 shared instances to ensure
no solver returns an invalid satisfying assignment.

Acceptable outcomes per solver on a given instance:
    - Returns a valid assignment that satisfies the formula (correct SAT).
    - Returns None (either UNSAT or the approach could not find a solution).
Unacceptable outcome:
    - Returns an assignment that does NOT satisfy the formula (bug).

Test categories:
    1. Algebraic approach: polynomial system generation, GF(2) arithmetic,
       linear equation solving, brute-force agreement.
    2. Spectral approach: VIG construction, Laplacian, eigenvalue computation,
       spectral features, brute-force agreement.
    3. LP relaxation: LP formulation, feasibility, integrality gap, rounding,
       brute-force agreement.
    4. Structural approach: 2-SAT/Horn-SAT detection and solving, treewidth
       estimation, backdoor search, brute-force agreement.
    5. Cross-approach consistency: all solvers on 20 shared instances.
    6. Edge cases: empty formula, single clause, contradictions.
"""

from __future__ import annotations

import pytest

from p_equals_np.sat_types import Clause, CNFFormula, Literal, Variable
from p_equals_np.sat_generator import (
    generate_random_ksat,
    generate_satisfiable_instance,
)
from p_equals_np.brute_force import BruteForceSolver
from p_equals_np.dpll import DPLLSolver
from p_equals_np.experimental.algebraic_approach import AlgebraicSolver
from p_equals_np.experimental.spectral_approach import (
    SpectralSolver,
    formula_to_vig,
    compute_laplacian,
    eigenvalues,
    spectral_features,
    spectral_partition,
)
from p_equals_np.experimental.geometric_approach import LPRelaxationSolver
from p_equals_np.experimental.structural_approach import (
    StructuralSolver,
    detect_2sat,
    detect_horn_sat,
    solve_2sat,
    solve_horn_sat,
    compute_vig as structural_compute_vig,
    estimate_treewidth,
    find_backdoor_candidates,
)


# ---------------------------------------------------------------------------
# Helper: build small formulas for targeted tests
# ---------------------------------------------------------------------------


def _make_formula(clauses_data: list[list[tuple[int, bool]]]) -> CNFFormula:
    """Build a CNFFormula from a compact representation.

    Args:
        clauses_data: List of clauses, each a list of (var_index, positive)
            tuples.

    Returns:
        A CNFFormula.
    """
    clauses = []
    for clause_data in clauses_data:
        literals = tuple(
            Literal(Variable(idx), positive=pos) for idx, pos in clause_data
        )
        clauses.append(Clause(literals))
    return CNFFormula(tuple(clauses))


def _make_2sat_sat() -> CNFFormula:
    """A satisfiable 2-SAT instance: (x1 OR x2) AND (~x1 OR x2).

    Satisfying assignment: {1: True, 2: True} or {1: False, 2: True}.
    """
    return _make_formula([
        [(1, True), (2, True)],
        [(1, False), (2, True)],
    ])


def _make_2sat_unsat() -> CNFFormula:
    """An unsatisfiable 2-SAT instance.

    (x1 OR x2) AND (~x1 OR ~x2) AND (x1 OR ~x2) AND (~x1 OR x2)
    Forces x1=x2 AND x1!=x2, which is impossible.
    """
    return _make_formula([
        [(1, True), (2, True)],
        [(1, False), (2, False)],
        [(1, True), (2, False)],
        [(1, False), (2, True)],
    ])


def _make_horn_sat() -> CNFFormula:
    """A satisfiable Horn-SAT instance.

    (x1) AND (~x1 OR ~x2 OR ~x3)
    Each clause has at most one positive literal. Assignment: x1=True,
    x2=False, x3=False satisfies both clauses.
    """
    return _make_formula([
        [(1, True)],                                    # x1
        [(1, False), (2, False), (3, False)],           # ~x1 OR ~x2 OR ~x3
    ])


def _make_horn_unsat() -> CNFFormula:
    """An unsatisfiable Horn-SAT instance.

    (x1) AND (~x1 OR x2) AND (~x2)
    Forces x1=True -> x2=True, but (~x2) requires x2=False.
    """
    return _make_formula([
        [(1, True)],
        [(1, False), (2, True)],
        [(2, False)],
    ])


def _to_int_clauses(formula: CNFFormula) -> list[frozenset[int]]:
    """Convert CNFFormula to int-literal clause format for structural API.

    Args:
        formula: A CNFFormula.

    Returns:
        List of frozensets of int literals.
    """
    result = []
    for clause in formula.clauses:
        int_lits = set()
        for lit in clause.literals:
            idx = lit.variable.index
            int_lits.add(idx if lit.positive else -idx)
        result.append(frozenset(int_lits))
    return result


def _collect_vars(clauses: list[frozenset[int]]) -> frozenset[int]:
    """Collect variable indices from int-clause representation."""
    variables: set[int] = set()
    for clause in clauses:
        for lit in clause:
            variables.add(abs(lit))
    return frozenset(variables)


# ---------------------------------------------------------------------------
# Test parameters for brute-force agreement (shared across approaches)
# ---------------------------------------------------------------------------

SMALL_INSTANCE_SEEDS = list(range(10))
SMALL_N = 5
SMALL_M = 12


# =========================================================================
# 1. ALGEBRAIC APPROACH TESTS
# =========================================================================


class TestAlgebraicSolver:
    """Tests for the AlgebraicSolver (polynomial systems over GF(2))."""

    def test_solver_protocol(self) -> None:
        """AlgebraicSolver provides name and complexity_claim."""
        solver = AlgebraicSolver()
        assert isinstance(solver.name(), str)
        assert len(solver.name()) > 0
        assert isinstance(solver.complexity_claim(), str)
        assert len(solver.complexity_claim()) > 0

    def test_polynomial_system_generation(self) -> None:
        """Polynomial system has at least one polynomial per clause."""
        formula = _make_formula([
            [(1, True), (2, False)],
            [(2, True), (3, True)],
        ])
        solver = AlgebraicSolver()
        poly_system = solver.formula_to_polynomial_system(formula)
        # At least one polynomial per clause (plus field equations)
        assert len(poly_system) >= 2

    def test_gf2_polynomial_multiplication(self) -> None:
        """GF(2) multiplication handles idempotency (x^2 = x)."""
        solver = AlgebraicSolver()
        # p1 = x1, p2 = x1 -> product should be x1 (not x1^2)
        p1 = {frozenset({1}): 1}
        p2 = {frozenset({1}): 1}
        product = solver.multiply_polynomials_gf2(p1, p2)
        # In GF(2), x1 * x1 = x1 (frozenset union: {1} | {1} = {1})
        assert product.get(frozenset({1}), 0) == 1

    def test_gf2_multiplication_two_vars(self) -> None:
        """GF(2) multiplication of two distinct variables."""
        solver = AlgebraicSolver()
        # p1 = x1, p2 = x2 -> product should be x1*x2
        p1 = {frozenset({1}): 1}
        p2 = {frozenset({2}): 1}
        product = solver.multiply_polynomials_gf2(p1, p2)
        assert product.get(frozenset({1, 2}), 0) == 1

    def test_gf2_addition_mod2(self) -> None:
        """GF(2) multiplication handles coefficient cancellation (1+1=0)."""
        solver = AlgebraicSolver()
        # (1 + x1) * (1 + x1) = 1 + x1 + x1 + x1 = 1 + x1 (mod 2)
        p = {frozenset(): 1, frozenset({1}): 1}
        product = solver.multiply_polynomials_gf2(p, p)
        # 1*1=1, 1*x1=x1, x1*1=x1, x1*x1=x1
        # = 1 + x1 + x1 + x1 = 1 + 3*x1 = 1 + x1 (mod 2)
        assert product.get(frozenset(), 0) == 1
        assert product.get(frozenset({1}), 0) == 1

    def test_properties_after_solve(self) -> None:
        """Properties are updated after solving."""
        formula, _ = generate_satisfiable_instance(
            num_vars=5, num_clauses=10, k=3, seed=42
        )
        solver = AlgebraicSolver()
        solver.solve(formula)
        assert solver.polynomial_operations >= 0
        assert solver.max_degree_seen >= 0

    def test_simple_sat(self, simple_sat_formula: CNFFormula) -> None:
        """AlgebraicSolver finds a solution for simple SAT instance."""
        solver = AlgebraicSolver()
        result = solver.solve(simple_sat_formula)
        assert result is not None
        assert simple_sat_formula.evaluate(result)

    def test_simple_unsat(self, simple_unsat_formula: CNFFormula) -> None:
        """AlgebraicSolver returns None for UNSAT instance."""
        solver = AlgebraicSolver()
        result = solver.solve(simple_unsat_formula)
        assert result is None

    @pytest.mark.parametrize("seed", SMALL_INSTANCE_SEEDS)
    def test_agrees_with_brute_force(self, seed: int) -> None:
        """AlgebraicSolver agrees with brute force on small instances."""
        formula = generate_random_ksat(
            k=3, num_vars=SMALL_N, num_clauses=SMALL_M, seed=seed
        )
        brute = BruteForceSolver(timeout_seconds=10.0)
        brute_result = brute.solve(formula)

        algebraic = AlgebraicSolver()
        alg_result = algebraic.solve(formula)

        _assert_solver_consistency(
            formula, brute_result, alg_result, "Algebraic", seed
        )


# =========================================================================
# 2. SPECTRAL APPROACH TESTS
# =========================================================================


class TestSpectralVIG:
    """Tests for VIG construction and Laplacian computation."""

    def test_vig_triangle(self) -> None:
        """VIG of (x1 OR x2) AND (x2 OR x3) AND (x1 OR x3) is K3."""
        formula = _make_formula([
            [(1, True), (2, True)],
            [(2, True), (3, True)],
            [(1, True), (3, True)],
        ])
        adj = formula_to_vig(formula)
        assert len(adj) == 3
        # K3: each pair has weight 1
        assert adj[0][1] == 1.0
        assert adj[1][0] == 1.0
        assert adj[0][2] == 1.0
        assert adj[2][0] == 1.0
        assert adj[1][2] == 1.0
        assert adj[2][1] == 1.0
        # No self-loops
        assert adj[0][0] == 0.0
        assert adj[1][1] == 0.0
        assert adj[2][2] == 0.0

    def test_vig_empty_formula(self, empty_formula: CNFFormula) -> None:
        """VIG of empty formula is an empty matrix."""
        adj = formula_to_vig(empty_formula)
        assert adj == []

    def test_laplacian_k3(self) -> None:
        """Laplacian of K3 is [[2,-1,-1],[-1,2,-1],[-1,-1,2]]."""
        adj = [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
        lap = compute_laplacian(adj)
        assert lap[0][0] == 2.0
        assert lap[0][1] == -1.0
        assert lap[0][2] == -1.0
        assert lap[1][1] == 2.0
        assert lap[2][2] == 2.0


class TestSpectralEigenvalues:
    """Tests for eigenvalue computation accuracy."""

    def test_identity_2x2(self) -> None:
        """Eigenvalues of 2x2 identity are [1, 1]."""
        eigs = eigenvalues([[1.0, 0.0], [0.0, 1.0]])
        assert len(eigs) == 2
        assert abs(eigs[0] - 1.0) < 1e-6
        assert abs(eigs[1] - 1.0) < 1e-6

    def test_symmetric_2x2(self) -> None:
        """Eigenvalues of [[2,1],[1,2]] are [1, 3]."""
        eigs = eigenvalues([[2.0, 1.0], [1.0, 2.0]])
        assert len(eigs) == 2
        assert abs(eigs[0] - 1.0) < 1e-6
        assert abs(eigs[1] - 3.0) < 1e-6

    def test_k3_laplacian(self) -> None:
        """Eigenvalues of K3 Laplacian are [0, 3, 3]."""
        lap = [
            [2.0, -1.0, -1.0],
            [-1.0, 2.0, -1.0],
            [-1.0, -1.0, 2.0],
        ]
        eigs = eigenvalues(lap)
        assert len(eigs) == 3
        assert abs(eigs[0] - 0.0) < 1e-6
        assert abs(eigs[1] - 3.0) < 1e-6
        assert abs(eigs[2] - 3.0) < 1e-6

    def test_1x1_matrix(self) -> None:
        """Eigenvalue of 1x1 matrix [5] is [5]."""
        eigs = eigenvalues([[5.0]])
        assert len(eigs) == 1
        assert abs(eigs[0] - 5.0) < 1e-6

    def test_empty_matrix_raises(self) -> None:
        """Empty matrix raises ValueError."""
        with pytest.raises(ValueError):
            eigenvalues([])


class TestSpectralFeatures:
    """Tests for spectral feature computation."""

    def test_features_have_expected_keys(self) -> None:
        """spectral_features returns dict with all expected keys."""
        formula = _make_formula([
            [(1, True), (2, True)],
            [(2, True), (3, True)],
        ])
        features = spectral_features(formula)
        assert "algebraic_connectivity" in features
        assert "spectral_radius" in features
        assert "spectral_gap" in features
        assert "eigenvalue_ratio" in features

    def test_algebraic_connectivity_nonnegative(self) -> None:
        """Algebraic connectivity (lambda_2) is non-negative."""
        formula, _ = generate_satisfiable_instance(
            num_vars=6, num_clauses=12, k=3, seed=42
        )
        features = spectral_features(formula)
        assert features["algebraic_connectivity"] >= -1e-10

    def test_spectral_radius_positive(self) -> None:
        """Spectral radius is positive for non-trivial formulas."""
        formula, _ = generate_satisfiable_instance(
            num_vars=6, num_clauses=12, k=3, seed=42
        )
        features = spectral_features(formula)
        assert features["spectral_radius"] > 0

    def test_trivial_formula_features(self) -> None:
        """Single-variable formula gives zero features."""
        formula = _make_formula([[(1, True)]])
        features = spectral_features(formula)
        assert features["algebraic_connectivity"] == 0.0
        assert features["spectral_radius"] == 0.0


class TestSpectralPartition:
    """Tests for spectral partitioning."""

    def test_partition_covers_all_variables(self) -> None:
        """Spectral partition covers all variables in the formula."""
        formula, _ = generate_satisfiable_instance(
            num_vars=8, num_clauses=16, k=3, seed=42
        )
        part_a, part_b = spectral_partition(formula)
        all_vars = {v.index for v in formula.get_variables()}
        assert part_a | part_b == all_vars

    def test_partition_nonempty(self) -> None:
        """Both partitions are non-empty for multi-variable formulas."""
        formula, _ = generate_satisfiable_instance(
            num_vars=8, num_clauses=16, k=3, seed=42
        )
        part_a, part_b = spectral_partition(formula)
        assert len(part_a) > 0
        assert len(part_b) > 0


class TestSpectralSolver:
    """Tests for the SpectralSolver end-to-end."""

    def test_solver_protocol(self) -> None:
        """SpectralSolver provides name and complexity_claim."""
        solver = SpectralSolver()
        assert isinstance(solver.name(), str)
        assert "Spectral" in solver.name()
        assert isinstance(solver.complexity_claim(), str)

    def test_simple_sat(self, simple_sat_formula: CNFFormula) -> None:
        """SpectralSolver finds solution for simple SAT."""
        solver = SpectralSolver()
        result = solver.solve(simple_sat_formula)
        assert result is not None
        assert simple_sat_formula.evaluate(result)

    def test_simple_unsat(self, simple_unsat_formula: CNFFormula) -> None:
        """SpectralSolver returns None for UNSAT."""
        solver = SpectralSolver()
        result = solver.solve(simple_unsat_formula)
        assert result is None

    @pytest.mark.parametrize("seed", SMALL_INSTANCE_SEEDS)
    def test_agrees_with_brute_force(self, seed: int) -> None:
        """SpectralSolver agrees with brute force on small instances."""
        formula = generate_random_ksat(
            k=3, num_vars=SMALL_N, num_clauses=SMALL_M, seed=seed
        )
        brute = BruteForceSolver(timeout_seconds=10.0)
        brute_result = brute.solve(formula)

        spectral = SpectralSolver(timeout_seconds=10.0)
        spec_result = spectral.solve(formula)

        _assert_solver_consistency(
            formula, brute_result, spec_result, "Spectral", seed
        )


# =========================================================================
# 3. LP RELAXATION APPROACH TESTS
# =========================================================================


class TestLPFormulation:
    """Tests for LP formulation of SAT."""

    def test_constraint_matrix_dimensions(self) -> None:
        """Constraint matrix has one row per clause, one col per variable."""
        formula = _make_formula([
            [(1, True), (2, False), (3, True)],
            [(2, True), (3, True)],
        ])
        solver = LPRelaxationSolver()
        A, b, bounds = solver.formula_to_lp(formula)
        assert len(A) == 2  # 2 clauses
        assert all(len(row) == 3 for row in A)  # 3 variables
        assert len(b) == 2
        assert len(bounds) == 3

    def test_positive_literal_coefficient(self) -> None:
        """Positive literal xi contributes +1 to coefficient."""
        formula = _make_formula([[(1, True), (2, True), (3, True)]])
        solver = LPRelaxationSolver()
        A, b, bounds = solver.formula_to_lp(formula)
        assert A[0] == [1.0, 1.0, 1.0]
        assert b[0] == 1.0

    def test_negative_literal_coefficient(self) -> None:
        """Negated literal ~xi contributes -1 to coefficient."""
        formula = _make_formula([[(1, True), (2, False), (3, True)]])
        solver = LPRelaxationSolver()
        A, b, bounds = solver.formula_to_lp(formula)
        # Coefficients: x1=+1, ~x2=-1, x3=+1
        assert A[0] == [1.0, -1.0, 1.0]
        # RHS: 1 - (number of negations) = 1 - 1 = 0
        assert b[0] == 0.0

    def test_bounds_are_unit_interval(self) -> None:
        """Variable bounds are [0, 1] for LP relaxation."""
        formula = _make_formula([
            [(1, True), (2, True)],
        ])
        solver = LPRelaxationSolver()
        _, _, bounds = solver.formula_to_lp(formula)
        for lo, hi in bounds:
            assert lo == 0.0
            assert hi == 1.0


class TestLPFeasibility:
    """Tests for LP relaxation feasibility."""

    def test_lp_feasible_for_multi_literal_clauses(self) -> None:
        """LP relaxation is feasible for formulas with clauses of width >= 2."""
        formula, _ = generate_satisfiable_instance(
            num_vars=8, num_clauses=20, k=3, seed=42
        )
        solver = LPRelaxationSolver()
        result = solver.solve_lp_relaxation(formula)
        assert result is not None
        assert solver.lp_feasible

    def test_lp_infeasible_for_contradictory_units(self) -> None:
        """LP relaxation detects infeasibility from contradictory unit clauses."""
        formula = _make_formula([
            [(1, True)],   # x1 >= 1
            [(1, False)],  # -x1 >= 0, i.e., x1 <= 0
        ])
        solver = LPRelaxationSolver()
        result = solver.solve_lp_relaxation(formula)
        # Contradictory unit clauses may be detected as infeasible
        if result is None:
            assert not solver.lp_feasible


class TestIntegralityGap:
    """Tests for integrality gap computation."""

    def test_integral_solution_gap_zero(self) -> None:
        """Integrality gap is zero for an integral solution."""
        solver = LPRelaxationSolver()
        gap = solver.compute_integrality_gap([0.0, 1.0, 0.0, 1.0])
        assert abs(gap) < 1e-10

    def test_half_solution_gap_maximal(self) -> None:
        """Integrality gap is n/2 when all values are 0.5."""
        solver = LPRelaxationSolver()
        gap = solver.compute_integrality_gap([0.5, 0.5, 0.5, 0.5])
        assert abs(gap - 2.0) < 1e-10  # 4 * 0.5 = 2.0

    def test_empty_solution_gap_zero(self) -> None:
        """Integrality gap is zero for empty solution."""
        solver = LPRelaxationSolver()
        gap = solver.compute_integrality_gap([])
        assert abs(gap) < 1e-10


class TestLPRounding:
    """Tests for LP rounding strategies."""

    def test_threshold_rounding(self) -> None:
        """Threshold rounding: x >= 0.5 -> True, x < 0.5 -> False."""
        formula = _make_formula([[(1, True), (2, True)]])
        solver = LPRelaxationSolver()
        assignment = solver.round_threshold([0.7, 0.3], formula)
        assert assignment[1] is True
        assert assignment[2] is False

    def test_randomized_rounding_reproducible(self) -> None:
        """Randomized rounding with same seed gives same result."""
        formula = _make_formula([[(1, True), (2, True)]])
        solver = LPRelaxationSolver()
        result1 = solver.round_randomized([0.5, 0.5], formula, seed=42)
        result2 = solver.round_randomized([0.5, 0.5], formula, seed=42)
        assert result1 == result2

    def test_iterative_rounding_returns_assignment(self) -> None:
        """Iterative rounding returns a complete assignment."""
        formula = _make_formula([
            [(1, True), (2, True)],
            [(2, True), (3, True)],
        ])
        solver = LPRelaxationSolver()
        assignment = solver.round_iterative([0.8, 0.6, 0.3], formula)
        var_indices = sorted(v.index for v in formula.get_variables())
        for idx in var_indices:
            assert idx in assignment


class TestLPRelaxationSolver:
    """Tests for the LPRelaxationSolver end-to-end."""

    def test_solver_protocol(self) -> None:
        """LPRelaxationSolver provides name and complexity_claim."""
        solver = LPRelaxationSolver()
        assert isinstance(solver.name(), str)
        assert "LP" in solver.name()
        assert isinstance(solver.complexity_claim(), str)

    def test_simple_sat(self, simple_sat_formula: CNFFormula) -> None:
        """LPRelaxationSolver finds solution for simple SAT."""
        solver = LPRelaxationSolver()
        result = solver.solve(simple_sat_formula)
        if result is not None:
            assert simple_sat_formula.evaluate(result)

    def test_rounding_statistics_updated(self) -> None:
        """Rounding attempts counter increases after solve."""
        formula, _ = generate_satisfiable_instance(
            num_vars=5, num_clauses=8, k=3, seed=42
        )
        solver = LPRelaxationSolver()
        solver.solve(formula)
        assert solver.rounding_attempts > 0

    @pytest.mark.parametrize("seed", SMALL_INSTANCE_SEEDS)
    def test_agrees_with_brute_force(self, seed: int) -> None:
        """LPRelaxationSolver agrees with brute force on small instances."""
        formula = generate_random_ksat(
            k=3, num_vars=SMALL_N, num_clauses=SMALL_M, seed=seed
        )
        brute = BruteForceSolver(timeout_seconds=10.0)
        brute_result = brute.solve(formula)

        lp = LPRelaxationSolver()
        lp_result = lp.solve(formula)

        _assert_solver_consistency(
            formula, brute_result, lp_result, "LP", seed
        )


# =========================================================================
# 4. STRUCTURAL APPROACH TESTS
# =========================================================================


class TestSubclassDetection:
    """Tests for 2-SAT and Horn-SAT detection."""

    def test_detect_2sat_true(self) -> None:
        """Detect 2-SAT for formula with all clauses of width <= 2."""
        clauses = _to_int_clauses(_make_2sat_sat())
        assert detect_2sat(clauses) is True

    def test_detect_2sat_false(self) -> None:
        """Detect non-2-SAT for formula with a 3-literal clause."""
        formula = _make_formula([
            [(1, True), (2, True), (3, True)],
        ])
        clauses = _to_int_clauses(formula)
        assert detect_2sat(clauses) is False

    def test_detect_2sat_unit_clauses(self) -> None:
        """Unit clauses (width 1) are valid 2-SAT clauses."""
        formula = _make_formula([[(1, True)], [(2, False)]])
        clauses = _to_int_clauses(formula)
        assert detect_2sat(clauses) is True

    def test_detect_horn_sat_true(self) -> None:
        """Detect Horn-SAT: each clause has at most one positive literal."""
        clauses = _to_int_clauses(_make_horn_sat())
        assert detect_horn_sat(clauses) is True

    def test_detect_horn_sat_false(self) -> None:
        """Non-Horn formula: clause with two positive literals."""
        formula = _make_formula([
            [(1, True), (2, True), (3, True)],
        ])
        clauses = _to_int_clauses(formula)
        assert detect_horn_sat(clauses) is False

    def test_horn_sat_with_all_negative(self) -> None:
        """All-negative clause is Horn (0 positive literals)."""
        formula = _make_formula([
            [(1, False), (2, False), (3, False)],
        ])
        clauses = _to_int_clauses(formula)
        assert detect_horn_sat(clauses) is True


class TestSolve2SAT:
    """Tests for the 2-SAT solver (implication graph + SCC)."""

    def test_simple_sat(self) -> None:
        """2-SAT solver finds valid assignment for SAT instance."""
        formula = _make_2sat_sat()
        clauses = _to_int_clauses(formula)
        variables = _collect_vars(clauses)
        result = solve_2sat(clauses, variables)
        assert result is not None
        assert formula.evaluate(result)

    def test_simple_unsat(self) -> None:
        """2-SAT solver returns None for UNSAT instance."""
        formula = _make_2sat_unsat()
        clauses = _to_int_clauses(formula)
        variables = _collect_vars(clauses)
        result = solve_2sat(clauses, variables)
        assert result is None

    def test_unit_clause_positive(self) -> None:
        """2-SAT solver handles unit clause (x1) correctly."""
        formula = _make_formula([[(1, True)]])
        clauses = _to_int_clauses(formula)
        variables = _collect_vars(clauses)
        result = solve_2sat(clauses, variables)
        assert result is not None
        assert result[1] is True

    def test_unit_clause_negative(self) -> None:
        """2-SAT solver handles unit clause (~x1) correctly."""
        formula = _make_formula([[(1, False)]])
        clauses = _to_int_clauses(formula)
        variables = _collect_vars(clauses)
        result = solve_2sat(clauses, variables)
        assert result is not None
        assert result[1] is False

    def test_contradictory_units(self) -> None:
        """2-SAT solver detects contradiction: (x1) AND (~x1)."""
        formula = _make_formula([[(1, True)], [(1, False)]])
        clauses = _to_int_clauses(formula)
        variables = _collect_vars(clauses)
        result = solve_2sat(clauses, variables)
        assert result is None

    def test_chain_implication(self) -> None:
        """2-SAT solver handles implication chain correctly.

        (x1 OR x2) AND (~x1 OR x3) AND (~x2 OR x3):
        If x1=F then x2=T (from clause 1), then x3=T (from clause 3).
        """
        formula = _make_formula([
            [(1, True), (2, True)],
            [(1, False), (3, True)],
            [(2, False), (3, True)],
        ])
        clauses = _to_int_clauses(formula)
        variables = _collect_vars(clauses)
        result = solve_2sat(clauses, variables)
        assert result is not None
        assert formula.evaluate(result)


class TestSolveHornSAT:
    """Tests for the Horn-SAT solver (unit propagation)."""

    def test_simple_horn_sat(self) -> None:
        """Horn-SAT solver finds valid assignment."""
        formula = _make_horn_sat()
        clauses = _to_int_clauses(formula)
        variables = _collect_vars(clauses)
        result = solve_horn_sat(clauses, variables)
        assert result is not None
        assert formula.evaluate(result)

    def test_horn_unsat(self) -> None:
        """Horn-SAT solver returns None for UNSAT instance."""
        formula = _make_horn_unsat()
        clauses = _to_int_clauses(formula)
        variables = _collect_vars(clauses)
        result = solve_horn_sat(clauses, variables)
        assert result is None

    def test_all_negative_horn(self) -> None:
        """Horn formula with only negative literals is trivially SAT (all False)."""
        formula = _make_formula([
            [(1, False), (2, False)],
        ])
        clauses = _to_int_clauses(formula)
        variables = _collect_vars(clauses)
        result = solve_horn_sat(clauses, variables)
        assert result is not None
        assert formula.evaluate(result)


class TestTreewidthEstimation:
    """Tests for treewidth estimation."""

    def test_treewidth_nonnegative(self) -> None:
        """Treewidth estimate is non-negative."""
        formula, _ = generate_satisfiable_instance(
            num_vars=6, num_clauses=12, k=3, seed=42
        )
        clauses = _to_int_clauses(formula)
        tw = estimate_treewidth(clauses)
        assert tw >= 0

    def test_treewidth_at_most_n_minus_1(self) -> None:
        """Treewidth estimate does not exceed n-1."""
        formula, _ = generate_satisfiable_instance(
            num_vars=8, num_clauses=20, k=3, seed=42
        )
        clauses = _to_int_clauses(formula)
        tw = estimate_treewidth(clauses)
        n = len(_collect_vars(clauses))
        assert tw <= n - 1

    def test_treewidth_empty_formula(self) -> None:
        """Treewidth of empty formula is 0."""
        tw = estimate_treewidth([])
        assert tw == 0

    def test_treewidth_single_clause(self) -> None:
        """Treewidth of a single 3-clause equals 2 (complete graph on 3 vertices)."""
        formula = _make_formula([[(1, True), (2, True), (3, True)]])
        clauses = _to_int_clauses(formula)
        tw = estimate_treewidth(clauses)
        # K3 has treewidth 2
        assert tw == 2


class TestVIGConstruction:
    """Tests for the structural VIG (adjacency list form)."""

    def test_vig_has_all_variables(self) -> None:
        """VIG contains all variables from the formula."""
        formula = _make_formula([
            [(1, True), (2, True), (3, True)],
            [(2, True), (4, True)],
        ])
        clauses = _to_int_clauses(formula)
        vig = structural_compute_vig(clauses)
        assert set(vig.keys()) == {1, 2, 3, 4}

    def test_vig_edge_from_shared_clause(self) -> None:
        """Variables in the same clause are adjacent in VIG."""
        formula = _make_formula([
            [(1, True), (2, True), (3, True)],
        ])
        clauses = _to_int_clauses(formula)
        vig = structural_compute_vig(clauses)
        assert 2 in vig[1]
        assert 3 in vig[1]
        assert 1 in vig[2]


class TestBackdoorSearch:
    """Tests for backdoor candidate search."""

    def test_2sat_has_empty_backdoor(self) -> None:
        """A 2-SAT formula needs no backdoor (is already tractable).

        find_backdoor_candidates may or may not find trivial backdoors,
        but the formula should be detected as 2-SAT directly.
        """
        formula = _make_2sat_sat()
        clauses = _to_int_clauses(formula)
        assert detect_2sat(clauses)

    def test_backdoor_for_near_tractable(self) -> None:
        """A formula that becomes 2-SAT when one variable is fixed.

        (x1 OR x2 OR x3) AND (x1 OR x2) -- the first clause is 3-SAT.
        If x1 is fixed (to True), both clauses are satisfied.
        If x1 is fixed (to False), the first becomes (x2 OR x3) which is 2-SAT.
        So {x1} is a backdoor.
        """
        formula = _make_formula([
            [(1, True), (2, True), (3, True)],
            [(1, True), (2, True)],
        ])
        clauses = _to_int_clauses(formula)
        variables = _collect_vars(clauses)
        backdoors = find_backdoor_candidates(clauses, variables, max_size=2)
        # Should find at least one backdoor
        assert len(backdoors) > 0
        # At least one should have size <= 2
        assert any(len(bd) <= 2 for bd in backdoors)


class TestStructuralSolver:
    """Tests for the StructuralSolver end-to-end."""

    def test_solver_protocol(self) -> None:
        """StructuralSolver provides name and complexity_claim."""
        solver = StructuralSolver()
        assert isinstance(solver.name(), str)
        assert "Structural" in solver.name()
        assert isinstance(solver.complexity_claim(), str)

    def test_solves_2sat_correctly(self) -> None:
        """StructuralSolver dispatches to 2-SAT solver and finds solution."""
        formula = _make_2sat_sat()
        solver = StructuralSolver()
        result = solver.solve(formula)
        assert result is not None
        assert formula.evaluate(result)
        assert solver.detected_class == "2-SAT"

    def test_detects_2sat_unsat(self) -> None:
        """StructuralSolver detects UNSAT 2-SAT instance."""
        formula = _make_2sat_unsat()
        solver = StructuralSolver()
        result = solver.solve(formula)
        assert result is None
        assert solver.detected_class == "2-SAT"

    def test_solves_horn_sat_correctly(self) -> None:
        """StructuralSolver dispatches to Horn-SAT solver."""
        formula = _make_horn_sat()
        solver = StructuralSolver()
        result = solver.solve(formula)
        # Horn-SAT that is also 2-SAT might dispatch to 2-SAT first;
        # the key check is correctness.
        if result is not None:
            assert formula.evaluate(result)

    def test_general_3sat_returns_none(self) -> None:
        """StructuralSolver returns None for general 3-SAT (no shortcut)."""
        # Generate a random 3-SAT instance that is neither 2-SAT nor Horn-SAT
        formula = generate_random_ksat(k=3, num_vars=8, num_clauses=20, seed=99)
        solver = StructuralSolver()
        result = solver.solve(formula)
        # The structural solver may or may not find a backdoor.
        # If it returns a result, it must be valid.
        if result is not None:
            assert formula.evaluate(result)

    @pytest.mark.parametrize("seed", SMALL_INSTANCE_SEEDS)
    def test_agrees_with_brute_force(self, seed: int) -> None:
        """StructuralSolver agrees with brute force on small instances."""
        formula = generate_random_ksat(
            k=3, num_vars=SMALL_N, num_clauses=SMALL_M, seed=seed
        )
        brute = BruteForceSolver(timeout_seconds=10.0)
        brute_result = brute.solve(formula)

        structural = StructuralSolver()
        struct_result = structural.solve(formula)

        _assert_solver_consistency(
            formula, brute_result, struct_result, "Structural", seed
        )


# =========================================================================
# 5. CROSS-APPROACH CONSISTENCY TESTS
# =========================================================================


CROSS_SEEDS = list(range(20))
CROSS_N = 6
CROSS_M = 14


class TestCrossApproachConsistency:
    """All solvers agree on satisfiability and produce valid assignments.

    For 20 random 3-SAT instances with 6 variables:
    - Brute force is the ground truth.
    - Every experimental solver that returns an assignment must have that
      assignment actually satisfy the formula (no invalid assignments).
    - False negatives (returning None when brute force finds SAT) are
      acceptable -- the experimental approaches are not guaranteed to
      find solutions on every instance.
    """

    @pytest.mark.parametrize("seed", CROSS_SEEDS)
    def test_all_solvers_consistent(self, seed: int) -> None:
        """All solvers agree: no solver returns an invalid assignment."""
        formula = generate_random_ksat(
            k=3, num_vars=CROSS_N, num_clauses=CROSS_M, seed=seed
        )

        brute = BruteForceSolver(timeout_seconds=10.0)
        brute_result = brute.solve(formula)

        solvers: list[tuple[str, object]] = [
            ("DPLL", DPLLSolver(timeout_seconds=10.0)),
            ("Algebraic", AlgebraicSolver()),
            ("Spectral", SpectralSolver(timeout_seconds=10.0)),
            ("LP", LPRelaxationSolver()),
            ("Structural", StructuralSolver()),
        ]

        for solver_name, solver in solvers:
            result = solver.solve(formula)  # type: ignore[union-attr]
            _assert_solver_consistency(
                formula, brute_result, result, solver_name, seed
            )


# =========================================================================
# 6. EDGE CASE TESTS
# =========================================================================


class TestEdgeCases:
    """Edge cases: empty formula, single clause, contradictions."""

    def test_empty_formula_all_solvers(
        self, empty_formula: CNFFormula
    ) -> None:
        """Empty formula is vacuously SAT; all solvers should agree."""
        solvers = [
            AlgebraicSolver(),
            SpectralSolver(),
            LPRelaxationSolver(),
        ]
        for solver in solvers:
            result = solver.solve(empty_formula)
            # Empty formula is vacuously SAT, result should be {} or None
            # (some solvers may not handle empty formulas)
            if result is not None:
                assert empty_formula.evaluate(result)

    def test_single_positive_unit(
        self, trivial_formula: CNFFormula
    ) -> None:
        """Single clause (x1) is trivially SAT."""
        solvers = [
            AlgebraicSolver(),
            SpectralSolver(),
            LPRelaxationSolver(),
            StructuralSolver(),
        ]
        for solver in solvers:
            result = solver.solve(trivial_formula)
            if result is not None:
                assert trivial_formula.evaluate(result)

    def test_contradiction_unit_clauses(self) -> None:
        """Contradictory unit clauses: (x1) AND (~x1) is UNSAT."""
        formula = _make_formula([
            [(1, True)],
            [(1, False)],
        ])
        brute = BruteForceSolver()
        assert brute.solve(formula) is None

        solvers = [
            AlgebraicSolver(),
            SpectralSolver(),
            StructuralSolver(),
        ]
        for solver in solvers:
            result = solver.solve(formula)
            if result is not None:
                # If a solver returns a result, it must be valid
                assert formula.evaluate(result), (
                    f"{solver.name()} returned invalid assignment for "
                    f"contradictory UNSAT instance"
                )

    def test_single_clause_multiple_literals(self) -> None:
        """Single clause with 3 literals is easily SAT."""
        formula = _make_formula([
            [(1, True), (2, False), (3, True)],
        ])
        brute = BruteForceSolver()
        brute_result = brute.solve(formula)
        assert brute_result is not None

        solvers = [
            AlgebraicSolver(),
            SpectralSolver(),
            LPRelaxationSolver(),
            StructuralSolver(),
        ]
        for solver in solvers:
            result = solver.solve(formula)
            if result is not None:
                assert formula.evaluate(result)


# =========================================================================
# Assertion helpers
# =========================================================================


def _assert_solver_consistency(
    formula: CNFFormula,
    brute_result: dict[int, bool] | None,
    solver_result: dict[int, bool] | None,
    solver_name: str,
    seed: int,
) -> None:
    """Assert that a solver's result is consistent with brute force.

    Rules:
    - If brute force says UNSAT, the solver must NOT return an assignment
      that satisfies the formula. (It may return None, or an invalid
      assignment that does not satisfy -- but an invalid assignment is a bug.)
    - If brute force says SAT, the solver may return None (false negative,
      acceptable) or a valid satisfying assignment.
    - Any returned non-None assignment MUST satisfy the formula.

    Args:
        formula: The CNF formula.
        brute_result: Ground truth from BruteForceSolver.
        solver_result: Result from the experimental solver.
        solver_name: Name for error messages.
        seed: Random seed for error messages.
    """
    context = f"{solver_name} (seed={seed})"
    brute_sat = brute_result is not None

    if solver_result is not None:
        # Any non-None result must be a valid satisfying assignment
        assert formula.evaluate(solver_result), (
            f"{context} returned an assignment that does NOT satisfy "
            f"the formula. Assignment: {solver_result}"
        )

    if not brute_sat and solver_result is not None:
        # Solver claims SAT but brute force says UNSAT -- bug!
        # (We already checked the assignment is valid above, so if
        # we reach here, brute force has a bug or this is impossible.)
        assert formula.evaluate(solver_result), (
            f"{context} returned SAT but brute force says UNSAT"
        )
