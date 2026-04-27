"""Shared pytest fixtures for the p_equals_np test suite.

Provides reproducible SAT/UNSAT instances at varying difficulty levels.
All random fixtures use fixed seeds so tests are deterministic.
"""

from __future__ import annotations

import pytest

from p_equals_np.sat_types import Clause, CNFFormula, Literal, Variable
from p_equals_np.sat_generator import (
    generate_random_ksat,
    generate_random_3sat_at_threshold,
    generate_satisfiable_instance,
    generate_unsatisfiable_instance,
)


# ---------------------------------------------------------------------------
# Trivial / edge-case fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trivial_formula() -> CNFFormula:
    """A single clause with a single positive literal: (x1).

    Satisfiable with {1: True}.
    """
    x1 = Variable(1)
    clause = Clause((Literal(x1, positive=True),))
    return CNFFormula((clause,))


@pytest.fixture
def empty_formula() -> CNFFormula:
    """An empty CNF formula (no clauses).

    Vacuously satisfiable: every assignment satisfies it.
    """
    return CNFFormula(())


# ---------------------------------------------------------------------------
# Simple fixtures (hand-crafted, known answers)
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_sat_formula() -> CNFFormula:
    """A small satisfiable formula: 3 variables, 2 clauses.

    (x1 OR ~x2) AND (x2 OR x3)

    Known satisfying assignment: {1: True, 2: True, 3: True}
    Also satisfiable by: {1: True, 2: False, 3: True}, etc.
    """
    x1 = Variable(1)
    x2 = Variable(2)
    x3 = Variable(3)

    clause1 = Clause((Literal(x1, positive=True), Literal(x2, positive=False)))
    clause2 = Clause((Literal(x2, positive=True), Literal(x3, positive=True)))

    return CNFFormula((clause1, clause2))


@pytest.fixture
def known_satisfying_assignment() -> dict[int, bool]:
    """A known satisfying assignment for simple_sat_formula.

    {1: True, 2: True, 3: True} satisfies:
      - (x1 OR ~x2): x1=True makes this True
      - (x2 OR x3): x2=True makes this True
    """
    return {1: True, 2: True, 3: True}


@pytest.fixture
def simple_unsat_formula() -> CNFFormula:
    """A small provably unsatisfiable formula: 1 variable, 2 clauses.

    (x1) AND (~x1)

    No assignment can satisfy both clauses.
    """
    x1 = Variable(1)
    clause1 = Clause((Literal(x1, positive=True),))
    clause2 = Clause((Literal(x1, positive=False),))
    return CNFFormula((clause1, clause2))


# ---------------------------------------------------------------------------
# Medium-difficulty fixtures (generator-based, fixed seeds)
# ---------------------------------------------------------------------------


@pytest.fixture
def medium_sat_formula() -> CNFFormula:
    """A medium-sized satisfiable formula: 10 variables, ~30 clauses.

    Generated via planted instance to guarantee satisfiability.
    Seed 42 for reproducibility.
    """
    formula, _assignment = generate_satisfiable_instance(
        num_vars=10, num_clauses=30, k=3, seed=42
    )
    return formula


@pytest.fixture
def medium_sat_assignment() -> dict[int, bool]:
    """The planted assignment for medium_sat_formula."""
    _formula, assignment = generate_satisfiable_instance(
        num_vars=10, num_clauses=30, k=3, seed=42
    )
    return assignment


# ---------------------------------------------------------------------------
# Hard fixtures (phase-transition region)
# ---------------------------------------------------------------------------


@pytest.fixture
def hard_sat_formula() -> CNFFormula:
    """A hard random 3-SAT formula at the phase transition.

    15 variables at clause-to-variable ratio ~4.267.
    These instances are empirically the hardest for SAT solvers.
    Seed 7 chosen to produce a satisfiable instance.
    """
    return generate_random_3sat_at_threshold(num_vars=15, seed=7)
