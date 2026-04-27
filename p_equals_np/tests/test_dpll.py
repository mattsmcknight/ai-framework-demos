"""Tests for p_equals_np.dpll module.

Covers: DPLLSolver on all fixture types (SAT, UNSAT, trivial, empty),
unit propagation, pure literal elimination, agreement with BruteForceSolver
on random instances, statistics counters, and performance comparison.
"""

from __future__ import annotations

import time

import pytest

from p_equals_np.brute_force import BruteForceSolver
from p_equals_np.dpll import DPLLSolver
from p_equals_np.sat_types import (
    Clause,
    CNFFormula,
    Literal,
    Variable,
)
from p_equals_np.sat_generator import (
    generate_random_ksat,
    generate_satisfiable_instance,
    generate_unsatisfiable_instance,
)
from p_equals_np.definitions import Solver


# ---------------------------------------------------------------------------
# Basic correctness on fixtures
# ---------------------------------------------------------------------------


class TestDPLLCorrectness:
    """Tests for DPLLSolver correctness on all fixture types."""

    def test_sat_returns_assignment(self, simple_sat_formula) -> None:
        """Solver returns a satisfying assignment for SAT formula."""
        solver = DPLLSolver(timeout_seconds=5.0)
        result = solver.solve(simple_sat_formula)
        assert result is not None
        assert simple_sat_formula.evaluate(result) is True

    def test_unsat_returns_none(self, simple_unsat_formula) -> None:
        """Solver returns None for UNSAT formula."""
        solver = DPLLSolver(timeout_seconds=5.0)
        result = solver.solve(simple_unsat_formula)
        assert result is None

    def test_trivial_formula(self, trivial_formula) -> None:
        """Solver handles a single-clause, single-variable formula."""
        solver = DPLLSolver(timeout_seconds=5.0)
        result = solver.solve(trivial_formula)
        assert result is not None
        assert result[1] is True

    def test_empty_formula(self, empty_formula) -> None:
        """Solver returns empty assignment for empty formula."""
        solver = DPLLSolver(timeout_seconds=5.0)
        result = solver.solve(empty_formula)
        assert result is not None
        assert result == {}

    def test_medium_sat(self, medium_sat_formula) -> None:
        """Solver finds a solution for a medium planted instance."""
        solver = DPLLSolver(timeout_seconds=10.0)
        result = solver.solve(medium_sat_formula)
        assert result is not None
        assert medium_sat_formula.evaluate(result) is True

    def test_hard_sat(self, hard_sat_formula) -> None:
        """Solver handles a hard phase-transition instance."""
        solver = DPLLSolver(timeout_seconds=10.0)
        result = solver.solve(hard_sat_formula)
        # May be SAT or UNSAT; if SAT, assignment must be valid
        if result is not None:
            assert hard_sat_formula.evaluate(result) is True

    def test_solver_protocol_conformance(self) -> None:
        """DPLLSolver conforms to the Solver protocol."""
        solver = DPLLSolver()
        assert isinstance(solver, Solver)

    def test_name(self) -> None:
        """name() returns 'DPLL'."""
        solver = DPLLSolver()
        assert solver.name() == "DPLL"

    def test_complexity_claim(self) -> None:
        """complexity_claim() mentions worst case."""
        solver = DPLLSolver()
        assert "worst case" in solver.complexity_claim()


# ---------------------------------------------------------------------------
# Unit propagation
# ---------------------------------------------------------------------------


class TestUnitPropagation:
    """Tests for unit propagation behavior."""

    def test_formula_with_unit_clause(self) -> None:
        """Unit propagation forces the single literal in a unit clause.

        Formula: (x1) AND (x1 OR x2)
        Unit propagation should set x1=True, satisfying both clauses.
        """
        x1 = Variable(1)
        x2 = Variable(2)
        formula = CNFFormula((
            Clause((Literal(x1, True),)),  # unit clause: x1
            Clause((Literal(x1, True), Literal(x2, True))),
        ))
        solver = DPLLSolver(timeout_seconds=5.0)
        result = solver.solve(formula)
        assert result is not None
        assert result[1] is True  # x1 must be True due to unit propagation
        assert solver.propagations > 0

    def test_cascading_unit_propagation(self) -> None:
        """Cascading unit propagation resolves the entire formula.

        Formula: (x1) AND (~x1 OR x2) AND (~x2 OR x3)
        Unit prop: x1=True -> x2=True -> x3=True
        """
        x1, x2, x3 = Variable(1), Variable(2), Variable(3)
        formula = CNFFormula((
            Clause((Literal(x1, True),)),
            Clause((Literal(x1, False), Literal(x2, True))),
            Clause((Literal(x2, False), Literal(x3, True))),
        ))
        solver = DPLLSolver(timeout_seconds=5.0)
        result = solver.solve(formula)
        assert result is not None
        assert result[1] is True
        assert result[2] is True
        assert result[3] is True
        # Should need no decisions (all forced by unit propagation)
        assert solver.decisions == 0


# ---------------------------------------------------------------------------
# Pure literal elimination
# ---------------------------------------------------------------------------


class TestPureLiteralElimination:
    """Tests for pure literal elimination behavior."""

    def test_pure_literal_assigned(self) -> None:
        """Pure literal is assigned to satisfy its clauses.

        Formula: (x1 OR x2) AND (x1 OR x3)
        x1 appears only positively -> pure literal, set x1=True.
        """
        x1, x2, x3 = Variable(1), Variable(2), Variable(3)
        formula = CNFFormula((
            Clause((Literal(x1, True), Literal(x2, True))),
            Clause((Literal(x1, True), Literal(x3, True))),
        ))
        solver = DPLLSolver(timeout_seconds=5.0)
        result = solver.solve(formula)
        assert result is not None
        assert formula.evaluate(result) is True
        # Propagations should be positive (pure literal counts as propagation)
        assert solver.propagations > 0


# ---------------------------------------------------------------------------
# Agreement with BruteForceSolver
# ---------------------------------------------------------------------------


class TestAgreementWithBruteForce:
    """DPLL agrees with BruteForce on random instances (seeds 0-19)."""

    @pytest.mark.parametrize("seed", range(20))
    def test_agreement_random_3sat(self, seed: int) -> None:
        """DPLL and BruteForce agree on random 3-SAT (8 vars, 25 clauses)."""
        formula = generate_random_ksat(k=3, num_vars=8, num_clauses=25, seed=seed)
        brute = BruteForceSolver(timeout_seconds=10.0)
        dpll = DPLLSolver(timeout_seconds=10.0)

        brute_result = brute.solve(formula)
        dpll_result = dpll.solve(formula)

        if brute_result is not None:
            # BruteForce says SAT -> DPLL must also say SAT
            assert dpll_result is not None, (
                f"seed={seed}: BruteForce SAT but DPLL UNSAT"
            )
            assert formula.evaluate(dpll_result), (
                f"seed={seed}: DPLL assignment invalid"
            )
        else:
            # BruteForce says UNSAT -> DPLL must also say UNSAT
            assert dpll_result is None, (
                f"seed={seed}: BruteForce UNSAT but DPLL returned assignment"
            )


# ---------------------------------------------------------------------------
# Statistics counters
# ---------------------------------------------------------------------------


class TestCounters:
    """Tests for DPLL statistics counters."""

    def test_counters_positive_after_solving(self, medium_sat_formula) -> None:
        """Counters are positive after solving a non-trivial formula."""
        solver = DPLLSolver(timeout_seconds=10.0)
        solver.solve(medium_sat_formula)
        # At least one of decisions or propagations should be positive
        assert (solver.decisions + solver.propagations) > 0

    def test_counters_are_integers(self, simple_sat_formula) -> None:
        """All counters are integers."""
        solver = DPLLSolver(timeout_seconds=5.0)
        solver.solve(simple_sat_formula)
        assert isinstance(solver.decisions, int)
        assert isinstance(solver.propagations, int)
        assert isinstance(solver.backtracks, int)

    def test_counters_reset_between_solves(self) -> None:
        """Counters reset at the start of each solve call."""
        formula = generate_unsatisfiable_instance(num_vars=4, seed=0)
        solver = DPLLSolver(timeout_seconds=5.0)

        solver.solve(formula)
        first_propagations = solver.propagations

        solver.solve(formula)
        second_propagations = solver.propagations

        # Counters should be the same for the same formula
        assert first_propagations == second_propagations

    def test_backtracks_on_unsat(self) -> None:
        """Backtracks are positive when solving an UNSAT formula with branching."""
        # Use a formula that requires branching (not just unit propagation)
        formula = generate_random_ksat(k=3, num_vars=6, num_clauses=30, seed=5)
        solver = DPLLSolver(timeout_seconds=10.0)
        result = solver.solve(formula)
        # If UNSAT, there must have been backtracks
        if result is None:
            assert solver.backtracks > 0


# ---------------------------------------------------------------------------
# Performance comparison
# ---------------------------------------------------------------------------


class TestPerformance:
    """DPLL should be faster than BruteForce on medium instances."""

    def test_dpll_faster_on_medium_sat(self) -> None:
        """DPLL is faster than BruteForce on multiple medium planted instances.

        We time both solvers on 5 planted instances at 12 variables.
        DPLL total time should be strictly less than BruteForce total time.
        """
        brute = BruteForceSolver(timeout_seconds=30.0)
        dpll = DPLLSolver(timeout_seconds=30.0)

        brute_total = 0.0
        dpll_total = 0.0

        for seed in range(5):
            formula, _ = generate_satisfiable_instance(
                num_vars=12, num_clauses=40, k=3, seed=seed + 100
            )

            start = time.perf_counter()
            brute.solve(formula)
            brute_total += time.perf_counter() - start

            start = time.perf_counter()
            dpll.solve(formula)
            dpll_total += time.perf_counter() - start

        assert dpll_total < brute_total, (
            f"DPLL ({dpll_total:.4f}s) not faster than BruteForce ({brute_total:.4f}s)"
        )

    def test_dpll_faster_on_unsat(self) -> None:
        """DPLL is faster than BruteForce on UNSAT instances.

        Unit propagation detects contradictions quickly.
        """
        brute = BruteForceSolver(timeout_seconds=30.0)
        dpll = DPLLSolver(timeout_seconds=30.0)

        brute_total = 0.0
        dpll_total = 0.0

        for n in range(3, 9):
            formula = generate_unsatisfiable_instance(num_vars=n, seed=0)

            start = time.perf_counter()
            brute.solve(formula)
            brute_total += time.perf_counter() - start

            start = time.perf_counter()
            dpll.solve(formula)
            dpll_total += time.perf_counter() - start

        assert dpll_total < brute_total, (
            f"DPLL ({dpll_total:.4f}s) not faster than BruteForce ({brute_total:.4f}s)"
        )
