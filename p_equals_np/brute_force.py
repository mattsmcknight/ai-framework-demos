"""Brute-force exhaustive search SAT solver.

Enumerates all 2^n truth assignments for n variables and evaluates each
against the CNF formula. This is the ground-truth baseline solver:
correctness is paramount, and no heuristic pruning is applied.

The only optimization permitted is short-circuit evaluation within
clauses (stop evaluating a clause once a satisfying literal is found)
and within the formula (stop evaluating clauses once an unsatisfied
clause is found). These are correctness-preserving and follow directly
from the semantics of AND/OR.

Complexity: O(2^n * m) where n = number of variables, m = number of clauses.
This is exponential in n, which is expected for a brute-force NP-complete solver.

Example:
    >>> from p_equals_np.sat_types import Variable, Literal, Clause, CNFFormula
    >>> x1, x2 = Variable(1), Variable(2)
    >>> clause = Clause((Literal(x1), Literal(x2, positive=False)))
    >>> formula = CNFFormula((clause,))
    >>> solver = BruteForceSolver()
    >>> result = solver.solve(formula)
    >>> result is not None
    True
"""

from __future__ import annotations

import itertools
import time
from typing import Optional

from p_equals_np.sat_types import CNFFormula

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIMEOUT_CHECK_INTERVAL = 1000


# ---------------------------------------------------------------------------
# BruteForceSolver
# ---------------------------------------------------------------------------


class BruteForceSolver:
    """Exhaustive brute-force SAT solver enumerating all 2^n assignments.

    Implements the Solver protocol defined in ``p_equals_np.definitions``.
    This solver is the correctness baseline: every other solver's results
    should agree with this one on small instances.

    The solver checks for timeout every ``_TIMEOUT_CHECK_INTERVAL``
    assignments to avoid excessive ``time.perf_counter()`` overhead.

    Attributes:
        timeout_seconds: Maximum wall-clock seconds before raising
            TimeoutError. Set to 0 or negative to disable timeout.
    """

    __slots__ = ("timeout_seconds", "_assignments_tried")

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        """Initialize the brute-force solver.

        Args:
            timeout_seconds: Maximum wall-clock time in seconds for a
                single solve or count_solutions call. Defaults to 30.
                Set to 0 or negative to disable the timeout.
        """
        self.timeout_seconds = timeout_seconds
        self._assignments_tried: int = 0

    @property
    def assignments_tried(self) -> int:
        """Number of assignments evaluated in the most recent call.

        Returns:
            The count of assignments checked during the last invocation
            of ``solve`` or ``count_solutions``.
        """
        return self._assignments_tried

    def solve(self, formula: CNFFormula) -> Optional[dict[int, bool]]:
        """Find a satisfying assignment by exhaustive enumeration.

        Iterates over all 2^n possible truth assignments in lexicographic
        order (by variable index, False before True). Returns the first
        satisfying assignment found, or None if the formula is unsatisfiable.

        Uses short-circuit evaluation: a clause is satisfied as soon as
        one literal evaluates to True, and the formula is falsified as
        soon as one clause evaluates to False.

        Args:
            formula: A CNF formula to solve.

        Returns:
            A dict mapping variable indices to truth values if the
            formula is satisfiable, or None if unsatisfiable.

        Raises:
            TimeoutError: If the solver exceeds ``timeout_seconds``.
        """
        self._assignments_tried = 0
        var_indices = _sorted_variable_indices(formula)

        if not var_indices:
            if _evaluate_formula(formula, {}):
                return {}
            return None

        deadline = _compute_deadline(self.timeout_seconds)

        for values in itertools.product((False, True), repeat=len(var_indices)):
            self._assignments_tried += 1

            assignment = dict(zip(var_indices, values))
            if _evaluate_formula(formula, assignment):
                return assignment

            if deadline is not None and self._assignments_tried % _TIMEOUT_CHECK_INTERVAL == 0:
                _check_timeout(deadline, self._assignments_tried)

        return None

    def count_solutions(
        self, formula: CNFFormula, max_count: int = 0
    ) -> int:
        """Count satisfying assignments by exhaustive enumeration.

        Iterates over all 2^n assignments and counts how many satisfy
        the formula. If ``max_count`` is positive, stops early once
        that many solutions have been found.

        Args:
            formula: A CNF formula to count solutions for.
            max_count: Stop counting after this many solutions. Use 0
                (default) to count all solutions.

        Returns:
            The number of satisfying assignments found.

        Raises:
            TimeoutError: If the solver exceeds ``timeout_seconds``.
        """
        self._assignments_tried = 0
        var_indices = _sorted_variable_indices(formula)

        if not var_indices:
            return 1 if _evaluate_formula(formula, {}) else 0

        count = 0
        deadline = _compute_deadline(self.timeout_seconds)

        for values in itertools.product((False, True), repeat=len(var_indices)):
            self._assignments_tried += 1

            assignment = dict(zip(var_indices, values))
            if _evaluate_formula(formula, assignment):
                count += 1
                if max_count > 0 and count >= max_count:
                    return count

            if deadline is not None and self._assignments_tried % _TIMEOUT_CHECK_INTERVAL == 0:
                _check_timeout(deadline, self._assignments_tried)

        return count

    def name(self) -> str:
        """Return the human-readable solver name.

        Returns:
            The string ``"BruteForce"``.
        """
        return "BruteForce"

    def complexity_claim(self) -> str:
        """State the time complexity of this solver.

        Returns:
            The string ``"O(2^n * m)"`` where n is variables
            and m is clauses.
        """
        return "O(2^n * m)"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sorted_variable_indices(formula: CNFFormula) -> list[int]:
    """Extract and sort variable indices from a formula.

    Args:
        formula: A CNF formula.

    Returns:
        A sorted list of variable indices appearing in the formula.
    """
    return sorted(v.index for v in formula.get_variables())


def _evaluate_formula(formula: CNFFormula, assignment: dict[int, bool]) -> bool:
    """Evaluate a CNF formula with short-circuit semantics.

    Stops as soon as any clause is unsatisfied (short-circuit AND).
    Within each clause, stops as soon as any literal is satisfied
    (short-circuit OR). This is the only optimization applied.

    Args:
        formula: The CNF formula to evaluate.
        assignment: Mapping from variable index to truth value.

    Returns:
        True if the formula is satisfied, False otherwise.
    """
    for clause in formula.clauses:
        if not _evaluate_clause(clause, assignment):
            return False
    return True


def _evaluate_clause(clause, assignment: dict[int, bool]) -> bool:
    """Evaluate a single clause with short-circuit OR semantics.

    Args:
        clause: A Clause object containing literals.
        assignment: Mapping from variable index to truth value.

    Returns:
        True if at least one literal is satisfied.
    """
    for literal in clause.literals:
        value = assignment[literal.variable.index]
        if literal.positive == value:
            return True
    return False


def _compute_deadline(timeout_seconds: float) -> Optional[float]:
    """Compute the absolute deadline from a timeout duration.

    Args:
        timeout_seconds: Timeout in seconds. Non-positive disables timeout.

    Returns:
        Absolute time (via ``time.perf_counter``) of the deadline,
        or None if timeout is disabled.
    """
    if timeout_seconds > 0:
        return time.perf_counter() + timeout_seconds
    return None


def _check_timeout(deadline: float, assignments_tried: int) -> None:
    """Raise TimeoutError if the deadline has passed.

    Args:
        deadline: Absolute time of the deadline.
        assignments_tried: Number of assignments tried so far
            (included in the error message for diagnostics).

    Raises:
        TimeoutError: If the current time exceeds the deadline.
    """
    if time.perf_counter() > deadline:
        raise TimeoutError(
            f"BruteForceSolver exceeded timeout after "
            f"{assignments_tried} assignments"
        )
