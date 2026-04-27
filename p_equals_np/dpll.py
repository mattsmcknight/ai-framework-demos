"""DPLL SAT solver with unit propagation and pure literal elimination.

Implements the Davis-Putnam-Logemann-Loveland (DPLL) algorithm, the
foundational backtracking SAT solver. While worst-case exponential
(O(2^n)), DPLL is dramatically faster than brute force on structured
instances thanks to three pruning mechanisms:

1. **Unit propagation**: If a clause has exactly one unset literal,
   that literal must be True. Propagating this forced assignment can
   cascade, resolving many clauses without branching.

2. **Pure literal elimination**: If a variable appears only positively
   (or only negatively) across all remaining clauses, it can be set to
   satisfy all clauses containing it. No branching needed.

3. **Branching with MOMS heuristic**: When forced moves are exhausted,
   pick the variable appearing most frequently in the shortest remaining
   clauses, try True then False, and backtrack on conflict.

Internal representation uses lists of frozensets of ints for speed:
positive int = positive literal, negative int = negated literal. This
avoids the overhead of Literal/Clause/Variable objects in the hot path.

Complexity: O(2^n) worst case, often much better in practice.

Example:
    >>> from p_equals_np.sat_types import Variable, Literal, Clause, CNFFormula
    >>> x1, x2 = Variable(1), Variable(2)
    >>> clause = Clause((Literal(x1), Literal(x2, positive=False)))
    >>> formula = CNFFormula((clause,))
    >>> solver = DPLLSolver()
    >>> result = solver.solve(formula)
    >>> result is not None
    True
"""

from __future__ import annotations

import time
from typing import Optional

from p_equals_np.sat_types import CNFFormula


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIMEOUT_CHECK_INTERVAL = 500


# ---------------------------------------------------------------------------
# DPLLSolver
# ---------------------------------------------------------------------------


class DPLLSolver:
    """DPLL SAT solver with unit propagation and pure literal elimination.

    Implements the Solver protocol defined in ``p_equals_np.definitions``.
    Uses an integer-based internal clause representation for performance:
    variable i with positive polarity is represented as +i, negated as -i.

    Attributes:
        timeout_seconds: Maximum wall-clock seconds before raising
            TimeoutError. Set to 0 or negative to disable timeout.
    """

    __slots__ = (
        "timeout_seconds",
        "_decisions",
        "_propagations",
        "_backtracks",
        "_deadline",
        "_step_counter",
    )

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        """Initialize the DPLL solver.

        Args:
            timeout_seconds: Maximum wall-clock time in seconds for a
                single solve call. Defaults to 30. Set to 0 or negative
                to disable the timeout.
        """
        self.timeout_seconds = timeout_seconds
        self._decisions: int = 0
        self._propagations: int = 0
        self._backtracks: int = 0
        self._deadline: Optional[float] = None
        self._step_counter: int = 0

    # --- Public properties ---

    @property
    def decisions(self) -> int:
        """Number of branching decisions made in the most recent solve call."""
        return self._decisions

    @property
    def propagations(self) -> int:
        """Number of unit propagations performed in the most recent solve call."""
        return self._propagations

    @property
    def backtracks(self) -> int:
        """Number of backtracks performed in the most recent solve call."""
        return self._backtracks

    # --- Solver protocol ---

    def solve(self, formula: CNFFormula) -> Optional[dict[int, bool]]:
        """Find a satisfying assignment using the DPLL algorithm.

        Converts the formula to an internal integer-based representation,
        then runs recursive DPLL with unit propagation, pure literal
        elimination, and MOMS-based variable selection.

        Args:
            formula: A CNF formula to solve.

        Returns:
            A dict mapping variable indices to truth values if the
            formula is satisfiable, or None if unsatisfiable.

        Raises:
            TimeoutError: If the solver exceeds ``timeout_seconds``.
        """
        self._decisions = 0
        self._propagations = 0
        self._backtracks = 0
        self._step_counter = 0
        self._deadline = _compute_deadline(self.timeout_seconds)

        clauses = _formula_to_int_clauses(formula)
        variables = _collect_variables(clauses)

        return self._dpll(clauses, {}, variables)

    def name(self) -> str:
        """Return the human-readable solver name.

        Returns:
            The string ``"DPLL"``.
        """
        return "DPLL"

    def complexity_claim(self) -> str:
        """State the time complexity of this solver.

        Returns:
            A string describing the worst-case and practical complexity.
        """
        return "O(2^n) worst case, often much better in practice"

    # --- Core DPLL algorithm ---

    def _dpll(
        self,
        clauses: list[frozenset[int]],
        assignment: dict[int, bool],
        variables: frozenset[int],
    ) -> Optional[dict[int, bool]]:
        """Core recursive DPLL procedure.

        Applies unit propagation and pure literal elimination as forced
        moves, then branches on an unassigned variable using the MOMS
        heuristic when no forced moves remain.

        Args:
            clauses: List of clauses, each a frozenset of int literals.
            assignment: Current partial truth assignment (var index -> bool).
            variables: All variable indices in the original formula.

        Returns:
            A complete satisfying assignment, or None if unsatisfiable.
        """
        # Unit propagation (may cascade)
        clauses, assignment, conflict = self._unit_propagate(clauses, assignment)
        if conflict:
            return None

        # Pure literal elimination
        clauses, assignment = self._pure_literal_eliminate(clauses, assignment)

        # Check termination conditions
        if not clauses:
            return _complete_assignment(assignment, variables)

        if _has_empty_clause(clauses):
            return None

        # Branch on unassigned variable
        var = self._choose_variable(clauses, assignment)
        if var is None:
            return _complete_assignment(assignment, variables)

        self._decisions += 1
        self._check_timeout()

        # Try True branch
        result = self._try_branch(clauses, assignment, variables, var, True)
        if result is not None:
            return result

        # Try False branch (backtrack)
        self._backtracks += 1
        return self._try_branch(clauses, assignment, variables, var, False)

    def _try_branch(
        self,
        clauses: list[frozenset[int]],
        assignment: dict[int, bool],
        variables: frozenset[int],
        var: int,
        value: bool,
    ) -> Optional[dict[int, bool]]:
        """Try assigning a value to a variable and recurse.

        Args:
            clauses: Current clause set.
            assignment: Current partial assignment.
            variables: All variable indices.
            var: Variable index to assign.
            value: Truth value to assign.

        Returns:
            A satisfying assignment, or None if this branch fails.
        """
        new_clauses = _simplify(clauses, var, value)
        new_assignment = dict(assignment)
        new_assignment[var] = value
        return self._dpll(new_clauses, new_assignment, variables)

    # --- Unit propagation ---

    def _unit_propagate(
        self,
        clauses: list[frozenset[int]],
        assignment: dict[int, bool],
    ) -> tuple[list[frozenset[int]], dict[int, bool], bool]:
        """Find and propagate unit clauses until no more remain.

        A unit clause contains exactly one literal, which must be True.
        Setting that literal may create new unit clauses, so propagation
        repeats until a fixed point.

        Args:
            clauses: Current clause set.
            assignment: Current partial assignment.

        Returns:
            A tuple of (simplified_clauses, updated_assignment, conflict).
            conflict is True if an empty clause was produced.
        """
        assignment = dict(assignment)
        changed = True

        while changed:
            changed = False
            unit_literal = _find_unit_literal(clauses)

            if unit_literal is None:
                break

            var = abs(unit_literal)
            value = unit_literal > 0
            assignment[var] = value
            clauses = _simplify(clauses, var, value)
            self._propagations += 1
            changed = True

            if _has_empty_clause(clauses):
                return clauses, assignment, True

        return clauses, assignment, False

    # --- Pure literal elimination ---

    def _pure_literal_eliminate(
        self,
        clauses: list[frozenset[int]],
        assignment: dict[int, bool],
    ) -> tuple[list[frozenset[int]], dict[int, bool]]:
        """Find and assign pure literals.

        A literal is pure if it appears in the clause set but its
        negation does not. Pure literals can be set to satisfy all
        clauses containing them without risk of conflict.

        Args:
            clauses: Current clause set.
            assignment: Current partial assignment.

        Returns:
            A tuple of (simplified_clauses, updated_assignment).
        """
        assignment = dict(assignment)
        pure_literals = _find_pure_literals(clauses)

        for literal in pure_literals:
            var = abs(literal)
            if var not in assignment:
                value = literal > 0
                assignment[var] = value
                clauses = _simplify(clauses, var, value)
                self._propagations += 1

        return clauses, assignment

    # --- Variable selection ---

    def _choose_variable(
        self,
        clauses: list[frozenset[int]],
        assignment: dict[int, bool],
    ) -> Optional[int]:
        """Choose an unassigned variable for branching using MOMS heuristic.

        MOMS (Maximum Occurrences in clauses of Minimum Size) selects
        the variable appearing most frequently in the shortest remaining
        clauses. This focuses branching on the most constrained variables.

        Args:
            clauses: Current clause set.
            assignment: Current partial assignment.

        Returns:
            A variable index, or None if all variables are assigned.
        """
        if not clauses:
            return None

        min_len = min(len(c) for c in clauses)
        shortest_clauses = [c for c in clauses if len(c) == min_len]

        counts: dict[int, int] = {}
        for clause in shortest_clauses:
            for literal in clause:
                var = abs(literal)
                if var not in assignment:
                    counts[var] = counts.get(var, 0) + 1

        if not counts:
            # All variables in shortest clauses are assigned; fall back
            # to any unassigned variable in remaining clauses
            for clause in clauses:
                for literal in clause:
                    var = abs(literal)
                    if var not in assignment:
                        return var
            return None

        return max(counts, key=counts.__getitem__)

    # --- Timeout ---

    def _check_timeout(self) -> None:
        """Periodically check whether the timeout has been exceeded.

        Only performs the actual time check every _TIMEOUT_CHECK_INTERVAL
        steps to amortize the cost of time.perf_counter().

        Raises:
            TimeoutError: If the deadline has passed.
        """
        if self._deadline is None:
            return
        self._step_counter += 1
        if self._step_counter % _TIMEOUT_CHECK_INTERVAL == 0:
            if time.perf_counter() > self._deadline:
                raise TimeoutError(
                    f"DPLLSolver exceeded timeout after "
                    f"{self._decisions} decisions, "
                    f"{self._propagations} propagations"
                )


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _formula_to_int_clauses(formula: CNFFormula) -> list[frozenset[int]]:
    """Convert a CNFFormula to the internal integer-based representation.

    Positive literal for variable i -> +i, negated literal -> -i.

    Args:
        formula: A CNFFormula with Literal/Clause objects.

    Returns:
        A list of frozensets of int literals.
    """
    result: list[frozenset[int]] = []
    for clause in formula.clauses:
        int_lits: set[int] = set()
        for literal in clause.literals:
            idx = literal.variable.index
            int_lits.add(idx if literal.positive else -idx)
        result.append(frozenset(int_lits))
    return result


def _collect_variables(clauses: list[frozenset[int]]) -> frozenset[int]:
    """Collect all variable indices from a clause set.

    Args:
        clauses: List of int-literal frozensets.

    Returns:
        A frozenset of variable indices (always positive).
    """
    variables: set[int] = set()
    for clause in clauses:
        for literal in clause:
            variables.add(abs(literal))
    return frozenset(variables)


# ---------------------------------------------------------------------------
# Clause manipulation helpers
# ---------------------------------------------------------------------------


def _simplify(
    clauses: list[frozenset[int]], var: int, value: bool
) -> list[frozenset[int]]:
    """Simplify clauses after assigning a value to a variable.

    - Clauses containing the satisfied literal are removed entirely.
    - The falsified literal is removed from remaining clauses.

    Args:
        clauses: Current clause set.
        var: Variable index being assigned.
        value: Truth value being assigned.

    Returns:
        A new simplified clause list.
    """
    satisfied_literal = var if value else -var
    falsified_literal = -var if value else var

    result: list[frozenset[int]] = []
    for clause in clauses:
        if satisfied_literal in clause:
            continue
        if falsified_literal in clause:
            result.append(clause - {falsified_literal})
        else:
            result.append(clause)
    return result


def _find_unit_literal(clauses: list[frozenset[int]]) -> Optional[int]:
    """Find a unit literal (clause with exactly one literal).

    Args:
        clauses: Current clause set.

    Returns:
        The single literal from the first unit clause found, or None.
    """
    for clause in clauses:
        if len(clause) == 1:
            return next(iter(clause))
    return None


def _has_empty_clause(clauses: list[frozenset[int]]) -> bool:
    """Check whether any clause is empty (conflict).

    Args:
        clauses: Current clause set.

    Returns:
        True if any clause has zero literals.
    """
    for clause in clauses:
        if not clause:
            return True
    return False


def _find_pure_literals(clauses: list[frozenset[int]]) -> list[int]:
    """Find all pure literals in the clause set.

    A literal is pure if it appears but its negation does not.

    Args:
        clauses: Current clause set.

    Returns:
        A list of pure literals (signed ints).
    """
    all_literals: set[int] = set()
    for clause in clauses:
        all_literals.update(clause)

    pure: list[int] = []
    for literal in all_literals:
        if -literal not in all_literals:
            pure.append(literal)
    return pure


def _complete_assignment(
    assignment: dict[int, bool],
    variables: frozenset[int],
) -> dict[int, bool]:
    """Fill in any unassigned variables with False.

    Variables not appearing in the remaining clauses (eliminated by
    pure literal or not constrained) default to False.

    Args:
        assignment: Current partial assignment.
        variables: All variable indices in the original formula.

    Returns:
        A complete assignment covering all variables.
    """
    result = dict(assignment)
    for var in variables:
        if var not in result:
            result[var] = False
    return result


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
