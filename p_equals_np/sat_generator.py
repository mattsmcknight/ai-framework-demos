"""SAT instance generators for random, structured, and planted instances.

Provides functions to generate CNF formulas with controlled properties
for benchmarking SAT solvers. Includes random k-SAT at arbitrary and
threshold ratios, guaranteed-satisfiable planted instances, small
unsatisfiable instances, and structured instances from combinatorial
problems (pigeonhole, XOR chains, graph coloring).

All generators use explicit seeds for reproducibility. Random generators
avoid duplicate clauses.

Example:
    >>> formula = generate_random_3sat_at_threshold(20, seed=42)
    >>> formula.num_variables
    20
    >>> abs(formula.clause_variable_ratio - 4.267) < 0.1
    True
"""

from __future__ import annotations

import math
import random
from typing import Optional

from p_equals_np.sat_types import Clause, CNFFormula, Literal, Variable


# ---------------------------------------------------------------------------
# Phase transition threshold for 3-SAT
# ---------------------------------------------------------------------------

_3SAT_THRESHOLD_RATIO = 4.267


# ---------------------------------------------------------------------------
# Random k-SAT Generation
# ---------------------------------------------------------------------------


def generate_random_ksat(
    k: int,
    num_vars: int,
    num_clauses: int,
    seed: Optional[int] = None,
) -> CNFFormula:
    """Generate a random k-SAT formula.

    Each clause contains exactly k distinct variables with random polarity.
    No duplicate clauses are produced. If the requested number of clauses
    exceeds the number of possible distinct k-clauses, a ValueError is raised.

    Args:
        k: Number of literals per clause.
        num_vars: Number of Boolean variables (1-indexed).
        num_clauses: Number of clauses to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        A random CNFFormula with the specified parameters.

    Raises:
        ValueError: If parameters are invalid or too many clauses requested.
    """
    _validate_ksat_params(k, num_vars, num_clauses)
    rng = random.Random(seed)

    seen: set[tuple[tuple[int, bool], ...]] = set()
    clauses: list[Clause] = []

    while len(clauses) < num_clauses:
        var_indices = rng.sample(range(1, num_vars + 1), k)
        polarities = tuple(rng.choice((True, False)) for _ in range(k))
        clause_key = tuple(sorted(zip(var_indices, polarities)))

        if clause_key in seen:
            continue
        seen.add(clause_key)

        literals = tuple(
            Literal(Variable(idx), pos)
            for idx, pos in zip(var_indices, polarities)
        )
        clauses.append(Clause(literals))

    return CNFFormula(tuple(clauses))


def _validate_ksat_params(k: int, num_vars: int, num_clauses: int) -> None:
    """Validate parameters for k-SAT generation.

    Args:
        k: Literals per clause.
        num_vars: Number of variables.
        num_clauses: Number of clauses.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if num_vars < 1:
        raise ValueError(f"num_vars must be >= 1, got {num_vars}")
    if num_clauses < 0:
        raise ValueError(f"num_clauses must be >= 0, got {num_clauses}")
    if k > num_vars:
        raise ValueError(
            f"k ({k}) cannot exceed num_vars ({num_vars})"
        )
    # Maximum distinct k-clauses: C(n,k) * 2^k
    max_clauses = math.comb(num_vars, k) * (2 ** k)
    if num_clauses > max_clauses:
        raise ValueError(
            f"num_clauses ({num_clauses}) exceeds maximum possible "
            f"distinct {k}-clauses ({max_clauses}) for {num_vars} variables"
        )


# ---------------------------------------------------------------------------
# Threshold 3-SAT
# ---------------------------------------------------------------------------


def generate_random_3sat_at_threshold(
    num_vars: int,
    seed: Optional[int] = None,
) -> CNFFormula:
    """Generate a random 3-SAT formula at the phase transition threshold.

    The satisfiability phase transition for random 3-SAT occurs near a
    clause-to-variable ratio of approximately 4.267. Instances generated
    at this ratio are empirically the hardest for SAT solvers.

    Args:
        num_vars: Number of Boolean variables.
        seed: Optional random seed for reproducibility.

    Returns:
        A random 3-SAT formula near the phase transition.

    Raises:
        ValueError: If num_vars < 3.
    """
    if num_vars < 3:
        raise ValueError(
            f"num_vars must be >= 3 for 3-SAT, got {num_vars}"
        )
    num_clauses = round(num_vars * _3SAT_THRESHOLD_RATIO)
    return generate_random_ksat(3, num_vars, num_clauses, seed=seed)


# ---------------------------------------------------------------------------
# Planted Satisfiable Instance
# ---------------------------------------------------------------------------


def generate_satisfiable_instance(
    num_vars: int,
    num_clauses: int,
    k: int = 3,
    seed: Optional[int] = None,
) -> tuple[CNFFormula, dict[int, bool]]:
    """Generate a guaranteed-satisfiable k-SAT instance by planting an assignment.

    First generates a random truth assignment, then generates clauses that
    are all satisfied by that assignment. Each clause has exactly k literals
    with at least one literal consistent with the planted assignment.

    Args:
        num_vars: Number of Boolean variables.
        num_clauses: Number of clauses to generate.
        k: Number of literals per clause (default 3).
        seed: Optional random seed for reproducibility.

    Returns:
        A tuple of (formula, planted_assignment) where the planted
        assignment is guaranteed to satisfy the formula.

    Raises:
        ValueError: If parameters are invalid.
    """
    _validate_ksat_params(k, num_vars, num_clauses)
    rng = random.Random(seed)

    # Plant a random truth assignment
    planted = {i: rng.choice((True, False)) for i in range(1, num_vars + 1)}

    seen: set[tuple[tuple[int, bool], ...]] = set()
    clauses: list[Clause] = []

    while len(clauses) < num_clauses:
        var_indices = rng.sample(range(1, num_vars + 1), k)
        polarities = _generate_satisfying_polarities(var_indices, planted, rng)
        clause_key = tuple(sorted(zip(var_indices, polarities)))

        if clause_key in seen:
            continue
        seen.add(clause_key)

        literals = tuple(
            Literal(Variable(idx), pos)
            for idx, pos in zip(var_indices, polarities)
        )
        clauses.append(Clause(literals))

    return CNFFormula(tuple(clauses)), planted


def _generate_satisfying_polarities(
    var_indices: list[int],
    assignment: dict[int, bool],
    rng: random.Random,
) -> tuple[bool, ...]:
    """Generate polarities ensuring at least one literal satisfies the assignment.

    Generates random polarities, then checks if at least one literal is
    satisfied. If none are, flips one randomly chosen literal to make it
    satisfying.

    Args:
        var_indices: Variable indices for this clause.
        assignment: The planted truth assignment.
        rng: Random number generator.

    Returns:
        A tuple of polarities (one per variable).
    """
    polarities = [rng.choice((True, False)) for _ in var_indices]

    # Check if any literal is satisfied under the planted assignment
    satisfied = any(
        (pol and assignment[idx]) or (not pol and not assignment[idx])
        for idx, pol in zip(var_indices, polarities)
    )

    if not satisfied:
        # Flip one literal to match the planted assignment
        fix_pos = rng.randrange(len(var_indices))
        polarities[fix_pos] = assignment[var_indices[fix_pos]]

    return tuple(polarities)


# ---------------------------------------------------------------------------
# Unsatisfiable Instance
# ---------------------------------------------------------------------------


def generate_unsatisfiable_instance(
    num_vars: int,
    seed: Optional[int] = None,
) -> CNFFormula:
    """Generate a small unsatisfiable CNF formula.

    For small num_vars (up to about 10), produces a formula that is
    guaranteed unsatisfiable by including contradictory unit clauses
    for every variable: both (xi) and (~xi) for each variable i.

    For larger num_vars, uses a pigeonhole-like construction that
    constrains the formula to be unsatisfiable.

    Args:
        num_vars: Number of Boolean variables to use.
        seed: Optional random seed for reproducibility (used only for
            larger instances with randomized clause ordering).

    Returns:
        A CNFFormula that is unsatisfiable.

    Raises:
        ValueError: If num_vars < 1.
    """
    if num_vars < 1:
        raise ValueError(f"num_vars must be >= 1, got {num_vars}")

    rng = random.Random(seed)

    # For small instances: contradictory unit clauses
    # (x1) AND (~x1) AND (x2) AND (~x2) AND ...
    # This is trivially unsatisfiable.
    clauses: list[Clause] = []
    for i in range(1, num_vars + 1):
        var = Variable(i)
        clauses.append(Clause((Literal(var, positive=True),)))
        clauses.append(Clause((Literal(var, positive=False),)))

    if seed is not None:
        rng.shuffle(clauses)

    return CNFFormula(tuple(clauses))


# ---------------------------------------------------------------------------
# Structured Instances
# ---------------------------------------------------------------------------


def generate_structured_instance(
    pattern: str,
    size: int,
) -> CNFFormula:
    """Generate a structured SAT instance from a known combinatorial pattern.

    Structured instances have known properties that make them useful for
    testing solver behavior on non-random formulas.

    Supported patterns:
        - "pigeonhole": Pigeonhole principle (n+1 pigeons, n holes).
          Always unsatisfiable. Exponential for resolution proofs.
        - "xor_chain": Chain of XOR constraints encoded in CNF.
          Exactly one satisfying assignment per parity class.
        - "graph_coloring": 3-coloring of a cycle graph on `size` vertices.
          Satisfiable if and only if size is not 1 or 2 with 3 colors.

    Args:
        pattern: One of "pigeonhole", "xor_chain", "graph_coloring".
        size: Controls the instance size (meaning depends on pattern).

    Returns:
        A CNFFormula encoding the structured problem.

    Raises:
        ValueError: If pattern is unknown or size is invalid.
    """
    generators = {
        "pigeonhole": _generate_pigeonhole,
        "xor_chain": _generate_xor_chain,
        "graph_coloring": _generate_graph_coloring,
    }
    if pattern not in generators:
        raise ValueError(
            f"Unknown pattern {pattern!r}. "
            f"Supported: {sorted(generators.keys())}"
        )
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    return generators[pattern](size)


def _generate_pigeonhole(n: int) -> CNFFormula:
    """Encode the pigeonhole principle: n+1 pigeons into n holes.

    Variables p_{i,j} represent "pigeon i is in hole j" where
    i in 1..n+1 and j in 1..n. The formula asserts:
    1. Each pigeon is in at least one hole.
    2. No two pigeons share a hole.

    This formula is always unsatisfiable (by the pigeonhole principle)
    and requires exponential-size resolution proofs.

    Args:
        n: Number of holes (pigeons = n + 1).

    Returns:
        An unsatisfiable CNFFormula.
    """
    pigeons = n + 1

    def var_index(pigeon: int, hole: int) -> int:
        """Map (pigeon, hole) to 1-indexed variable."""
        return (pigeon - 1) * n + hole

    clauses: list[Clause] = []

    # Each pigeon must be in at least one hole
    for i in range(1, pigeons + 1):
        literals = tuple(
            Literal(Variable(var_index(i, j)), positive=True)
            for j in range(1, n + 1)
        )
        clauses.append(Clause(literals))

    # No two pigeons in the same hole
    for j in range(1, n + 1):
        for i1 in range(1, pigeons + 1):
            for i2 in range(i1 + 1, pigeons + 1):
                clauses.append(Clause((
                    Literal(Variable(var_index(i1, j)), positive=False),
                    Literal(Variable(var_index(i2, j)), positive=False),
                )))

    return CNFFormula(tuple(clauses))


def _generate_xor_chain(n: int) -> CNFFormula:
    """Encode a chain of XOR constraints in CNF.

    Encodes x1 XOR x2, x2 XOR x3, ..., x_{n-1} XOR x_n.
    Each XOR(a, b) is encoded as two clauses: (a OR b) AND (~a OR ~b).

    For n=1, produces a trivially satisfiable formula (just x1 in a clause).
    The chain has a satisfying assignment: alternating True/False values.

    Args:
        n: Number of variables in the chain.

    Returns:
        A CNFFormula encoding the XOR chain.
    """
    if n == 1:
        return CNFFormula((Clause((Literal(Variable(1), positive=True),)),))

    clauses: list[Clause] = []
    for i in range(1, n):
        var_a = Variable(i)
        var_b = Variable(i + 1)
        # XOR(a, b) = (a OR b) AND (~a OR ~b)
        clauses.append(Clause((
            Literal(var_a, positive=True),
            Literal(var_b, positive=True),
        )))
        clauses.append(Clause((
            Literal(var_a, positive=False),
            Literal(var_b, positive=False),
        )))

    return CNFFormula(tuple(clauses))


def _generate_graph_coloring(n: int) -> CNFFormula:
    """Encode 3-coloring of a cycle graph on n vertices.

    Variables: x_{v,c} for vertex v in 1..n and color c in 1..3.
    Constraints:
    1. Each vertex has at least one color.
    2. Each vertex has at most one color (at-most-one via pairwise exclusion).
    3. Adjacent vertices in the cycle have different colors.

    A cycle on n >= 3 vertices is 3-colorable. A cycle on n=1 is
    trivially 3-colorable. A cycle on n=2 is 3-colorable (two
    adjacent vertices, different colors).

    Args:
        n: Number of vertices in the cycle.

    Returns:
        A CNFFormula encoding the graph coloring problem.
    """
    num_colors = 3

    def var_index(vertex: int, color: int) -> int:
        """Map (vertex, color) to 1-indexed variable."""
        return (vertex - 1) * num_colors + color

    clauses: list[Clause] = []

    # Each vertex has at least one color
    for v in range(1, n + 1):
        literals = tuple(
            Literal(Variable(var_index(v, c)), positive=True)
            for c in range(1, num_colors + 1)
        )
        clauses.append(Clause(literals))

    # Each vertex has at most one color (pairwise exclusion)
    for v in range(1, n + 1):
        for c1 in range(1, num_colors + 1):
            for c2 in range(c1 + 1, num_colors + 1):
                clauses.append(Clause((
                    Literal(Variable(var_index(v, c1)), positive=False),
                    Literal(Variable(var_index(v, c2)), positive=False),
                )))

    # Adjacent vertices have different colors (cycle edges)
    if n >= 2:
        edges = [(v, v + 1) for v in range(1, n)] + [(n, 1)]
        for u, v in edges:
            for c in range(1, num_colors + 1):
                clauses.append(Clause((
                    Literal(Variable(var_index(u, c)), positive=False),
                    Literal(Variable(var_index(v, c)), positive=False),
                )))

    return CNFFormula(tuple(clauses))
