"""LP relaxation approach to SAT solving.

Theoretical basis: Any CNF-SAT instance can be formulated as an Integer
Linear Program (ILP) with binary variables x_i in {0, 1}. Each clause
becomes a linear inequality. For example, the clause (x1 OR ~x2 OR x3)
becomes x1 + (1 - x2) + x3 >= 1, i.e., x1 - x2 + x3 >= 0.

Relaxing the integrality constraint to 0 <= x_i <= 1 yields a Linear
Program (LP) solvable in polynomial time (e.g., interior point methods
run in O(n^3.5)). If the LP relaxation yields an integral solution
(all x_i in {0, 1}), we have solved the original SAT instance. Otherwise,
we must round the fractional solution, and this rounding step is where
the polynomial-time guarantee breaks down.

Key insight: For any clause of width k >= 2, setting every x_i = 0.5
satisfies all LP constraints (each clause contributes at least k/2 >= 1).
This means the LP relaxation is almost always feasible, but the trivial
feasible solution x = 0.5 is maximally uninformative.

Why this cannot solve SAT in polynomial time:
    1. The integrality gap for SAT ILPs can be arbitrarily large.
       The LP may report "feasible" (with fractional values) even for
       UNSAT instances.
    2. Near the phase transition (clause ratio ~4.267 for 3-SAT), the LP
       relaxation produces values clustered around 0.5, providing no
       useful guidance for rounding.
    3. The Sherali-Adams hierarchy of LP/SDP relaxations requires an
       exponential number of rounds to exactly capture SAT, establishing
       that LP-based approaches fundamentally cannot bridge the gap.
    4. Rounding (threshold, randomized, or iterative) is a heuristic
       with no worst-case polynomial-time guarantee of finding a
       satisfying assignment.

This module implements the LP relaxation and three rounding strategies
to empirically demonstrate these limitations.
"""

from __future__ import annotations

import random
from typing import Optional

from p_equals_np.sat_types import CNFFormula


# ---------------------------------------------------------------------------
# LP Relaxation Solver
# ---------------------------------------------------------------------------


class LPRelaxationSolver:
    """SAT solver via LP relaxation and rounding.

    Formulates SAT as an ILP, relaxes to LP (0 <= x_i <= 1), solves
    the relaxation, then attempts multiple rounding strategies to
    recover an integral satisfying assignment.

    Implements the Solver protocol from ``p_equals_np.definitions``.

    Attributes:
        lp_feasible: Whether the most recent LP relaxation was feasible.
        rounding_attempts: Total rounding attempts across all solve calls.
        rounding_successes: Total rounding successes across all solve calls.
    """

    def __init__(self) -> None:
        """Initialize the LP relaxation solver."""
        self.lp_feasible: bool = False
        self.rounding_attempts: int = 0
        self.rounding_successes: int = 0

    def formula_to_lp(
        self, formula: CNFFormula
    ) -> tuple[list[list[float]], list[float], list[tuple[float, float]]]:
        """Convert a CNF formula to LP constraint form.

        Each clause (l1 OR l2 OR ... OR lk) becomes a linear inequality.
        For a positive literal xi, the coefficient is +1. For a negated
        literal ~xi, we substitute (1 - xi), contributing -1 to the
        coefficient and +1 to the constant offset. The resulting
        constraint is: sum(coefficients * x) >= 1 - (number of negations).

        Rewritten as: A * x >= b, with 0 <= x_i <= 1 for all i.

        Args:
            formula: A CNF formula to convert.

        Returns:
            A tuple (A, b, bounds) where:
            - A: Coefficient matrix (one row per clause).
            - b: Right-hand side vector (one entry per clause).
            - bounds: Variable bounds as (lower, upper) pairs.
        """
        var_indices = _sorted_variable_indices(formula)
        var_to_col = {idx: col for col, idx in enumerate(var_indices)}
        n_vars = len(var_indices)

        constraint_matrix: list[list[float]] = []
        rhs_vector: list[float] = []

        for clause in formula.clauses:
            row = [0.0] * n_vars
            negation_count = 0

            for literal in clause.literals:
                col = var_to_col[literal.variable.index]
                if literal.positive:
                    row[col] = 1.0
                else:
                    row[col] = -1.0
                    negation_count += 1

            rhs_vector.append(1.0 - negation_count)
            constraint_matrix.append(row)

        bounds = [(0.0, 1.0)] * n_vars
        return constraint_matrix, rhs_vector, bounds

    def solve_lp_relaxation(
        self, formula: CNFFormula
    ) -> Optional[list[float]]:
        """Solve the LP relaxation of the SAT ILP.

        Since x_i = 0.5 for all i is always feasible for clauses of
        width >= 2 (each clause contributes at least k/2 >= 1), we
        start from this trivial feasible point and iteratively adjust
        values toward integrality while maintaining feasibility.

        The iterative approach uses gradient steps that push each x_i
        toward the nearest integer (0 or 1) while projecting back to
        the feasible region after each step.

        Args:
            formula: A CNF formula to solve the LP relaxation for.

        Returns:
            A list of fractional variable values in variable-index order,
            or None if the LP is infeasible (only possible for clauses
            of width 0 or 1 with contradictory constraints).
        """
        constraint_matrix, rhs_vector, bounds = self.formula_to_lp(formula)
        n_vars = len(bounds)

        if n_vars == 0:
            self.lp_feasible = not formula.clauses
            return [] if self.lp_feasible else None

        # Start at the trivial feasible point
        x = [0.5] * n_vars

        if not _is_lp_feasible(constraint_matrix, rhs_vector, bounds, x):
            # Try to find a feasible starting point for unit clauses
            x = _find_feasible_start(constraint_matrix, rhs_vector, bounds)
            if x is None:
                self.lp_feasible = False
                return None

        # Iteratively push toward integrality
        x = _push_toward_integrality(
            x, constraint_matrix, rhs_vector, bounds
        )

        self.lp_feasible = True
        return x

    def round_threshold(
        self,
        fractional: list[float],
        formula: CNFFormula,
    ) -> dict[int, bool]:
        """Round fractional LP solution using a fixed threshold.

        Simple deterministic rounding: x_i >= 0.5 maps to True,
        x_i < 0.5 maps to False. This is the simplest rounding
        strategy and works well when the LP relaxation produces
        values far from 0.5.

        Args:
            fractional: Fractional variable values from LP relaxation.
            formula: The original CNF formula (for variable indices).

        Returns:
            A truth assignment mapping variable indices to booleans.
        """
        var_indices = _sorted_variable_indices(formula)
        return {
            var_indices[i]: fractional[i] >= 0.5
            for i in range(len(var_indices))
        }

    def round_randomized(
        self,
        fractional: list[float],
        formula: CNFFormula,
        seed: Optional[int] = None,
    ) -> dict[int, bool]:
        """Round fractional LP solution using randomized rounding.

        Each x_i is set to True with probability equal to its fractional
        value. This is the theoretically grounded approach: for MAX-SAT,
        randomized rounding achieves a (1 - 1/e) approximation ratio.
        However, for decision SAT we need ALL clauses satisfied, which
        randomized rounding cannot guarantee.

        Args:
            fractional: Fractional variable values from LP relaxation.
            formula: The original CNF formula (for variable indices).
            seed: Optional random seed for reproducibility.

        Returns:
            A truth assignment mapping variable indices to booleans.
        """
        rng = random.Random(seed)
        var_indices = _sorted_variable_indices(formula)
        return {
            var_indices[i]: rng.random() < fractional[i]
            for i in range(len(var_indices))
        }

    def round_iterative(
        self,
        fractional: list[float],
        formula: CNFFormula,
    ) -> dict[int, bool]:
        """Round fractional LP solution iteratively.

        Repeatedly fixes the most integral variable (closest to 0 or 1),
        propagates implications, and re-solves the reduced LP. This is
        the most sophisticated rounding strategy but still has no
        polynomial-time satisfiability guarantee.

        Args:
            fractional: Fractional variable values from LP relaxation.
            formula: The original CNF formula (for variable indices).

        Returns:
            A truth assignment mapping variable indices to booleans.
        """
        var_indices = _sorted_variable_indices(formula)
        n = len(var_indices)
        current = list(fractional)
        assignment: dict[int, bool] = {}
        fixed: set[int] = set()

        for _ in range(n):
            best_col = _find_most_integral(current, fixed)
            if best_col is None:
                break

            value = current[best_col] >= 0.5
            assignment[var_indices[best_col]] = value
            fixed.add(best_col)
            current[best_col] = 1.0 if value else 0.0

            # Propagate: nudge unfixed variables toward satisfying
            # remaining unsatisfied clauses
            _propagate_fixed(
                current, fixed, assignment, var_indices, formula
            )

        # Assign any remaining unfixed variables
        for col in range(n):
            if col not in fixed:
                assignment[var_indices[col]] = current[col] >= 0.5

        return assignment

    def solve(
        self, formula: CNFFormula
    ) -> Optional[dict[int, bool]]:
        """Solve SAT via LP relaxation and rounding.

        Solves the LP relaxation, then tries all three rounding
        strategies (threshold, randomized with multiple seeds,
        iterative). Returns the first satisfying assignment found,
        or None if no rounding succeeds.

        Args:
            formula: A CNF formula to solve.

        Returns:
            A satisfying assignment if found, or None.
        """
        fractional = self.solve_lp_relaxation(formula)
        if fractional is None:
            return None

        # Strategy 1: Threshold rounding
        assignment = self.round_threshold(fractional, formula)
        self.rounding_attempts += 1
        if formula.evaluate(assignment):
            self.rounding_successes += 1
            return assignment

        # Strategy 2: Randomized rounding (multiple attempts)
        for seed in range(10):
            assignment = self.round_randomized(
                fractional, formula, seed=seed
            )
            self.rounding_attempts += 1
            if formula.evaluate(assignment):
                self.rounding_successes += 1
                return assignment

        # Strategy 3: Iterative rounding
        assignment = self.round_iterative(fractional, formula)
        self.rounding_attempts += 1
        if formula.evaluate(assignment):
            self.rounding_successes += 1
            return assignment

        return None

    def compute_integrality_gap(self, fractional: list[float]) -> float:
        """Compute the integrality gap of a fractional LP solution.

        The integrality gap measures how far the fractional solution is
        from any integral solution. Computed as the sum of min(x_i, 1-x_i)
        over all variables. A value of 0 means the solution is integral;
        the maximum value of n/2 (for n variables all at 0.5) indicates
        maximum fractional ambiguity.

        Args:
            fractional: Fractional variable values from LP relaxation.

        Returns:
            The integrality gap (sum of distances to nearest integer).
        """
        return sum(min(x, 1.0 - x) for x in fractional)

    def name(self) -> str:
        """Return the solver name.

        Returns:
            The string identifying this solver.
        """
        return "LP Relaxation + Rounding"

    def complexity_claim(self) -> str:
        """State the claimed complexity.

        Returns:
            A string describing the complexity and its limitations.
        """
        return (
            "O(n^3.5) for LP + O(n) for rounding, "
            "but rounding not guaranteed"
        )


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------


def _sorted_variable_indices(formula: CNFFormula) -> list[int]:
    """Extract sorted variable indices from a formula.

    Args:
        formula: A CNF formula.

    Returns:
        Sorted list of variable indices.
    """
    return sorted(v.index for v in formula.get_variables())


def _is_lp_feasible(
    constraint_matrix: list[list[float]],
    rhs_vector: list[float],
    bounds: list[tuple[float, float]],
    x: list[float],
) -> bool:
    """Check whether a point satisfies all LP constraints.

    Args:
        constraint_matrix: Coefficient matrix (A).
        rhs_vector: Right-hand side (b) for A*x >= b.
        bounds: Variable bounds (lower, upper).
        x: Point to check.

    Returns:
        True if x is feasible.
    """
    tolerance = 1e-9

    for i, (lo, hi) in enumerate(bounds):
        if x[i] < lo - tolerance or x[i] > hi + tolerance:
            return False

    for row_idx, row in enumerate(constraint_matrix):
        lhs = sum(row[j] * x[j] for j in range(len(x)))
        if lhs < rhs_vector[row_idx] - tolerance:
            return False

    return True


def _find_feasible_start(
    constraint_matrix: list[list[float]],
    rhs_vector: list[float],
    bounds: list[tuple[float, float]],
) -> Optional[list[float]]:
    """Find a feasible starting point for the LP.

    Handles unit clauses and simple constraints by iteratively
    adjusting variables from the midpoint. Falls back to None
    if no feasible point can be found.

    Args:
        constraint_matrix: Coefficient matrix.
        rhs_vector: Right-hand side vector.
        bounds: Variable bounds.

    Returns:
        A feasible point, or None if infeasible.
    """
    n = len(bounds)
    x = [0.5] * n

    # Process constraints to determine forced values
    for row_idx, row in enumerate(constraint_matrix):
        nonzero = [(j, row[j]) for j in range(n) if abs(row[j]) > 1e-12]
        if len(nonzero) == 1:
            j, coeff = nonzero[0]
            # Single-variable constraint: coeff * x_j >= rhs
            if coeff > 0:
                x[j] = max(x[j], rhs_vector[row_idx] / coeff)
            elif coeff < 0:
                x[j] = min(x[j], rhs_vector[row_idx] / coeff)

    # Clamp to bounds
    for i in range(n):
        x[i] = max(bounds[i][0], min(bounds[i][1], x[i]))

    if _is_lp_feasible(constraint_matrix, rhs_vector, bounds, x):
        return x
    return None


def _push_toward_integrality(
    x: list[float],
    constraint_matrix: list[list[float]],
    rhs_vector: list[float],
    bounds: list[tuple[float, float]],
    max_iterations: int = 100,
    step_size: float = 0.05,
) -> list[float]:
    """Iteratively push LP solution toward integrality.

    Applies gradient steps that move each variable toward its nearest
    integer value, then projects back to the feasible region. This
    is a heuristic: it may get stuck at a fractional point if the
    feasible region does not extend to an integral vertex.

    Args:
        x: Starting feasible point.
        constraint_matrix: LP constraint matrix.
        rhs_vector: LP right-hand side.
        bounds: Variable bounds.
        max_iterations: Maximum iterations to attempt.
        step_size: Step size for each gradient update.

    Returns:
        The adjusted (possibly more integral) feasible point.
    """
    current = list(x)
    n = len(current)

    for _ in range(max_iterations):
        # Compute gradient toward nearest integer
        gradient = [0.0] * n
        for i in range(n):
            if current[i] < 0.5:
                gradient[i] = -step_size  # Push toward 0
            else:
                gradient[i] = step_size  # Push toward 1

        # Tentative step
        candidate = [
            max(bounds[i][0], min(bounds[i][1], current[i] + gradient[i]))
            for i in range(n)
        ]

        # Project back to feasible region
        candidate = _project_to_feasible(
            candidate, constraint_matrix, rhs_vector, bounds
        )

        # Accept if still feasible
        if _is_lp_feasible(
            constraint_matrix, rhs_vector, bounds, candidate
        ):
            current = candidate
        else:
            # Reduce step size and try again with smaller step
            step_size *= 0.5
            if step_size < 1e-10:
                break

    return current


def _project_to_feasible(
    x: list[float],
    constraint_matrix: list[list[float]],
    rhs_vector: list[float],
    bounds: list[tuple[float, float]],
    max_corrections: int = 50,
) -> list[float]:
    """Project a point back to the LP feasible region.

    Iterates through violated constraints and adjusts variables to
    restore feasibility. This is a simple iterative projection, not
    a full LP solve.

    Args:
        x: Point to project.
        constraint_matrix: LP constraint matrix.
        rhs_vector: LP right-hand side.
        bounds: Variable bounds.
        max_corrections: Maximum correction iterations.

    Returns:
        The projected (approximately feasible) point.
    """
    current = list(x)
    n = len(current)
    tolerance = 1e-9

    for _ in range(max_corrections):
        all_feasible = True

        for row_idx, row in enumerate(constraint_matrix):
            lhs = sum(row[j] * current[j] for j in range(n))
            violation = rhs_vector[row_idx] - lhs

            if violation <= tolerance:
                continue

            all_feasible = False
            # Distribute correction across nonzero coefficients
            nonzero = [
                (j, row[j]) for j in range(n) if abs(row[j]) > 1e-12
            ]
            if not nonzero:
                continue

            correction = violation / len(nonzero)
            for j, coeff in nonzero:
                if coeff > 0:
                    current[j] += correction / coeff
                else:
                    current[j] -= correction / abs(coeff)
                current[j] = max(
                    bounds[j][0], min(bounds[j][1], current[j])
                )

        if all_feasible:
            break

    return current


def _find_most_integral(
    values: list[float],
    fixed: set[int],
) -> Optional[int]:
    """Find the unfixed variable closest to 0 or 1.

    Args:
        values: Current fractional variable values.
        fixed: Set of already-fixed variable column indices.

    Returns:
        Column index of the most integral unfixed variable,
        or None if all are fixed.
    """
    best_col: Optional[int] = None
    best_distance = float("inf")

    for col in range(len(values)):
        if col in fixed:
            continue
        distance = min(values[col], 1.0 - values[col])
        if distance < best_distance:
            best_distance = distance
            best_col = col

    return best_col


def _propagate_fixed(
    current: list[float],
    fixed: set[int],
    assignment: dict[int, bool],
    var_indices: list[int],
    formula: CNFFormula,
) -> None:
    """Propagate fixed variable effects on unfixed variables.

    After fixing a variable, checks each clause for satisfaction.
    For unsatisfied clauses where only one unfixed variable remains
    with a positive contribution, nudges that variable toward
    satisfying the clause.

    Args:
        current: Current fractional values (modified in place).
        fixed: Set of fixed column indices.
        assignment: Current partial truth assignment.
        var_indices: Sorted variable indices.
        formula: The original CNF formula.
    """
    var_to_col = {idx: col for col, idx in enumerate(var_indices)}

    for clause in formula.clauses:
        # Check if clause is already satisfied by fixed variables
        clause_satisfied = False
        unfixed_helpers: list[tuple[int, bool]] = []

        for literal in clause.literals:
            col = var_to_col[literal.variable.index]
            if col in fixed:
                val = assignment[var_indices[col]]
                if literal.positive == val:
                    clause_satisfied = True
                    break
            else:
                unfixed_helpers.append((col, literal.positive))

        if clause_satisfied:
            continue

        # Nudge unfixed variables toward satisfying this clause
        if not unfixed_helpers:
            continue

        for col, is_positive in unfixed_helpers:
            if is_positive:
                current[col] = min(1.0, current[col] + 0.1)
            else:
                current[col] = max(0.0, current[col] - 0.1)
