"""Cross-validation tests: BruteForceSolver vs DPLLSolver on random 3-SAT.

Generates 54 random 3-SAT instances with 8 variables at six different
clause-to-variable ratios spanning the satisfiability spectrum from
under-constrained (ratio 2.0) through the phase transition (4.267)
to over-constrained (6.0). For each instance, both solvers are run
and their results are compared.

Assertions:
    - Both solvers agree on satisfiability (both SAT or both UNSAT).
    - For SAT instances, both solutions are verified via CNFFormula.evaluate().
    - For SAT instances, both solutions are verified via SATDecisionProblem.verify().

These tests serve as a mutual consistency check: if the solvers disagree
on any instance, at least one of them has a bug.

The full suite runs in well under 60 seconds (8-variable instances are
small enough for brute-force but large enough to exercise DPLL's pruning).
"""

from __future__ import annotations

import pytest

from p_equals_np.brute_force import BruteForceSolver
from p_equals_np.dpll import DPLLSolver
from p_equals_np.sat_types import CNFFormula, SATDecisionProblem
from p_equals_np.sat_generator import generate_random_ksat


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

NUM_VARS = 8
K = 3
TIMEOUT = 30.0

# Clause-to-variable ratios to test, spanning the full satisfiability range:
#   2.0 - Under-constrained (almost always SAT)
#   3.0 - Moderately constrained (usually SAT)
#   4.0 - Near threshold (mix of SAT and UNSAT)
#   4.267 - Phase transition threshold (hardest instances)
#   5.0 - Over-constrained (usually UNSAT)
#   6.0 - Heavily over-constrained (almost always UNSAT)
CLAUSE_RATIOS = [2.0, 3.0, 4.0, 4.267, 5.0, 6.0]

# 9 seeds per ratio = 54 total parametrized test cases
SEEDS_PER_RATIO = 9


def _build_test_params() -> list[tuple[float, int, int]]:
    """Build parametrized test cases as (ratio, num_clauses, seed) tuples.

    Returns:
        A list of tuples, one per test case.
    """
    params: list[tuple[float, int, int]] = []
    for ratio in CLAUSE_RATIOS:
        num_clauses = round(NUM_VARS * ratio)
        for seed in range(SEEDS_PER_RATIO):
            params.append((ratio, num_clauses, seed))
    return params


TEST_PARAMS = _build_test_params()

# Readable IDs for pytest output: "ratio=4.267_clauses=34_seed=3"
TEST_IDS = [
    f"ratio={ratio}_clauses={nc}_seed={seed}"
    for ratio, nc, seed in TEST_PARAMS
]


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------


class TestCrossValidation:
    """BruteForceSolver and DPLLSolver agree on random 3-SAT instances."""

    @pytest.mark.parametrize(
        "ratio, num_clauses, seed",
        TEST_PARAMS,
        ids=TEST_IDS,
    )
    def test_solvers_agree(
        self, ratio: float, num_clauses: int, seed: int
    ) -> None:
        """Both solvers agree on satisfiability and produce valid solutions.

        For each random 3-SAT instance:
        1. BruteForceSolver and DPLLSolver must agree on SAT vs UNSAT.
        2. If SAT, both assignments must satisfy the formula (CNFFormula.evaluate).
        3. If SAT, both assignments must pass SATDecisionProblem.verify.
        """
        formula = generate_random_ksat(
            k=K, num_vars=NUM_VARS, num_clauses=num_clauses, seed=seed
        )

        brute = BruteForceSolver(timeout_seconds=TIMEOUT)
        dpll = DPLLSolver(timeout_seconds=TIMEOUT)

        brute_result = brute.solve(formula)
        dpll_result = dpll.solve(formula)

        _assert_agreement(formula, brute_result, dpll_result, ratio, seed)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_agreement(
    formula: CNFFormula,
    brute_result: dict[int, bool] | None,
    dpll_result: dict[int, bool] | None,
    ratio: float,
    seed: int,
) -> None:
    """Assert that both solvers agree and produce valid solutions.

    Args:
        formula: The CNF formula that was solved.
        brute_result: BruteForceSolver result (assignment or None).
        dpll_result: DPLLSolver result (assignment or None).
        ratio: Clause-to-variable ratio (for error messages).
        seed: Random seed used (for error messages).
    """
    context = f"ratio={ratio}, seed={seed}"

    brute_sat = brute_result is not None
    dpll_sat = dpll_result is not None

    # Both must agree on satisfiability
    assert brute_sat == dpll_sat, (
        f"Solvers disagree on satisfiability ({context}): "
        f"BruteForce={'SAT' if brute_sat else 'UNSAT'}, "
        f"DPLL={'SAT' if dpll_sat else 'UNSAT'}"
    )

    if brute_sat:
        _verify_solution(formula, brute_result, "BruteForce", context)
        _verify_solution(formula, dpll_result, "DPLL", context)


def _verify_solution(
    formula: CNFFormula,
    assignment: dict[int, bool],
    solver_name: str,
    context: str,
) -> None:
    """Verify that a solution actually satisfies the formula.

    Uses two independent verification paths:
    1. CNFFormula.evaluate() -- direct evaluation
    2. SATDecisionProblem.verify() -- decision problem certificate check

    Args:
        formula: The CNF formula.
        assignment: The proposed satisfying assignment.
        solver_name: Name of the solver (for error messages).
        context: Additional context string (for error messages).
    """
    # Verification path 1: direct evaluation
    assert formula.evaluate(assignment), (
        f"{solver_name} returned an invalid assignment ({context}): "
        f"formula.evaluate() is False"
    )

    # Verification path 2: SATDecisionProblem.verify
    problem = SATDecisionProblem()
    assert problem.verify(formula, assignment), (
        f"{solver_name} assignment rejected by SATDecisionProblem.verify ({context})"
    )


# ---------------------------------------------------------------------------
# Summary statistics (collected after all tests via session-scoped fixture)
# ---------------------------------------------------------------------------


class TestCrossValidationCoverage:
    """Meta-test verifying the cross-validation suite has adequate coverage."""

    def test_total_instances_at_least_50(self) -> None:
        """The parametrized test suite contains at least 50 test cases."""
        assert len(TEST_PARAMS) >= 50, (
            f"Expected >= 50 test cases, got {len(TEST_PARAMS)}"
        )

    def test_all_ratios_covered(self) -> None:
        """All six clause ratios are present in the test parameters."""
        covered_ratios = sorted(set(r for r, _, _ in TEST_PARAMS))
        assert covered_ratios == sorted(CLAUSE_RATIOS), (
            f"Missing ratios: expected {sorted(CLAUSE_RATIOS)}, "
            f"got {covered_ratios}"
        )

    def test_instances_span_sat_and_unsat(self) -> None:
        """At least one ratio likely produces SAT and one likely produces UNSAT.

        Ratios below 4.267 tend toward SAT; ratios above tend toward UNSAT.
        This test verifies the range includes both regimes.
        """
        has_below_threshold = any(r < 4.267 for r, _, _ in TEST_PARAMS)
        has_above_threshold = any(r > 4.267 for r, _, _ in TEST_PARAMS)
        assert has_below_threshold, "No test cases below phase transition"
        assert has_above_threshold, "No test cases above phase transition"
