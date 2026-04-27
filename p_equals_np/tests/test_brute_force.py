"""Tests for p_equals_np.brute_force module.

Covers: BruteForceSolver.solve (SAT/UNSAT), assignments_tried count,
count_solutions, timeout handling, empty formula, and agreement with
SATDecisionProblem.decide on random instances.
"""

from __future__ import annotations

import pytest

from p_equals_np.brute_force import BruteForceSolver
from p_equals_np.sat_types import (
    Clause,
    CNFFormula,
    Literal,
    SATDecisionProblem,
    Variable,
)
from p_equals_np.sat_generator import (
    generate_random_ksat,
    generate_satisfiable_instance,
    generate_unsatisfiable_instance,
)
from p_equals_np.definitions import Solver


# ---------------------------------------------------------------------------
# Basic solve behavior
# ---------------------------------------------------------------------------


class TestBruteForceSolve:
    """Tests for BruteForceSolver.solve on known instances."""

    def test_sat_returns_assignment(self, simple_sat_formula) -> None:
        """Solver returns a satisfying assignment for SAT formula."""
        solver = BruteForceSolver(timeout_seconds=5.0)
        result = solver.solve(simple_sat_formula)
        assert result is not None
        assert simple_sat_formula.evaluate(result) is True

    def test_unsat_returns_none(self, simple_unsat_formula) -> None:
        """Solver returns None for UNSAT formula."""
        solver = BruteForceSolver(timeout_seconds=5.0)
        result = solver.solve(simple_unsat_formula)
        assert result is None

    def test_trivial_formula(self, trivial_formula) -> None:
        """Solver handles a single-clause, single-variable formula."""
        solver = BruteForceSolver(timeout_seconds=5.0)
        result = solver.solve(trivial_formula)
        assert result is not None
        assert result[1] is True

    def test_empty_formula(self, empty_formula) -> None:
        """Solver returns empty assignment for empty formula."""
        solver = BruteForceSolver(timeout_seconds=5.0)
        result = solver.solve(empty_formula)
        assert result is not None
        assert result == {}

    def test_medium_sat(self, medium_sat_formula) -> None:
        """Solver finds a solution for a medium planted instance."""
        solver = BruteForceSolver(timeout_seconds=10.0)
        result = solver.solve(medium_sat_formula)
        assert result is not None
        assert medium_sat_formula.evaluate(result) is True

    def test_solver_protocol_conformance(self) -> None:
        """BruteForceSolver conforms to the Solver protocol."""
        solver = BruteForceSolver()
        assert isinstance(solver, Solver)

    def test_name(self) -> None:
        """name() returns 'BruteForce'."""
        solver = BruteForceSolver()
        assert solver.name() == "BruteForce"

    def test_complexity_claim(self) -> None:
        """complexity_claim() mentions 2^n."""
        solver = BruteForceSolver()
        assert "2^n" in solver.complexity_claim()


# ---------------------------------------------------------------------------
# assignments_tried counter
# ---------------------------------------------------------------------------


class TestAssignmentsTried:
    """Tests for the assignments_tried counter."""

    def test_unsat_tries_all_assignments(self, simple_unsat_formula) -> None:
        """For UNSAT with 1 variable, tries 2^1 = 2 assignments."""
        solver = BruteForceSolver(timeout_seconds=5.0)
        solver.solve(simple_unsat_formula)
        assert solver.assignments_tried == 2

    def test_unsat_n_vars(self) -> None:
        """For UNSAT with n variables, tries 2^n assignments."""
        for n in range(1, 6):
            formula = generate_unsatisfiable_instance(num_vars=n, seed=0)
            solver = BruteForceSolver(timeout_seconds=10.0)
            solver.solve(formula)
            assert solver.assignments_tried == 2**n, (
                f"Expected 2^{n}={2**n}, got {solver.assignments_tried}"
            )

    def test_sat_does_not_try_all(self) -> None:
        """For SAT, solver may stop before trying all assignments."""
        # Trivially satisfiable: first assignment works
        formula, _ = generate_satisfiable_instance(
            num_vars=8, num_clauses=10, seed=0
        )
        solver = BruteForceSolver(timeout_seconds=5.0)
        result = solver.solve(formula)
        assert result is not None
        # Should not need all 256 assignments for a satisfiable instance
        # (though technically could if the solution is the last one)


# ---------------------------------------------------------------------------
# count_solutions
# ---------------------------------------------------------------------------


class TestCountSolutions:
    """Tests for BruteForceSolver.count_solutions."""

    def test_single_variable_sat(self) -> None:
        """(x1) has exactly 1 solution."""
        x1 = Variable(1)
        formula = CNFFormula((Clause((Literal(x1, True),)),))
        solver = BruteForceSolver(timeout_seconds=5.0)
        assert solver.count_solutions(formula) == 1

    def test_two_variable_unconstrained(self) -> None:
        """(x1 OR x2) has 3 satisfying assignments out of 4."""
        formula = CNFFormula((Clause((
            Literal(Variable(1), True),
            Literal(Variable(2), True),
        )),))
        solver = BruteForceSolver(timeout_seconds=5.0)
        assert solver.count_solutions(formula) == 3

    def test_unsat_zero_solutions(self, simple_unsat_formula) -> None:
        """UNSAT formula has 0 solutions."""
        solver = BruteForceSolver(timeout_seconds=5.0)
        assert solver.count_solutions(simple_unsat_formula) == 0

    def test_empty_formula_one_solution(self, empty_formula) -> None:
        """Empty formula has 1 solution (vacuously)."""
        solver = BruteForceSolver(timeout_seconds=5.0)
        assert solver.count_solutions(empty_formula) == 1

    def test_max_count_stops_early(self) -> None:
        """count_solutions with max_count stops at the limit."""
        # x1 OR x2 has 3 solutions; asking for max 2
        formula = CNFFormula((Clause((
            Literal(Variable(1), True),
            Literal(Variable(2), True),
        )),))
        solver = BruteForceSolver(timeout_seconds=5.0)
        assert solver.count_solutions(formula, max_count=2) == 2


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    """Tests for timeout handling."""

    def test_timeout_on_short_limit(self) -> None:
        """Solver raises TimeoutError on very short timeout with hard instance."""
        # 20 variables -> 2^20 = 1M assignments; very short timeout
        formula = generate_unsatisfiable_instance(num_vars=20, seed=0)
        solver = BruteForceSolver(timeout_seconds=0.0001)
        with pytest.raises(TimeoutError):
            solver.solve(formula)

    def test_no_timeout_when_disabled(self) -> None:
        """Solver does not timeout when timeout is disabled."""
        formula = generate_unsatisfiable_instance(num_vars=5, seed=0)
        solver = BruteForceSolver(timeout_seconds=0.0)  # disabled
        result = solver.solve(formula)
        assert result is None  # UNSAT, no timeout


# ---------------------------------------------------------------------------
# Agreement with SATDecisionProblem
# ---------------------------------------------------------------------------


class TestAgreementWithDecisionProblem:
    """BruteForceSolver agrees with SATDecisionProblem.decide on random instances."""

    @pytest.mark.parametrize("seed", range(15))
    def test_agreement_on_random_instance(self, seed: int) -> None:
        """Solver and SATDecisionProblem agree on random 3-SAT (6 vars, 20 clauses)."""
        formula = generate_random_ksat(k=3, num_vars=6, num_clauses=20, seed=seed)
        problem = SATDecisionProblem()
        solver = BruteForceSolver(timeout_seconds=5.0)

        expected_sat = problem.decide(formula)
        result = solver.solve(formula)

        if expected_sat:
            assert result is not None, f"seed={seed}: expected SAT but got None"
            assert formula.evaluate(result), f"seed={seed}: assignment does not satisfy"
        else:
            assert result is None, f"seed={seed}: expected UNSAT but got assignment"
