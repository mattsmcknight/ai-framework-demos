"""Tests for p_equals_np.sat_types module.

Covers: Variable, Literal, Clause, CNFFormula (including DIMACS
serialization round-trip), and SATDecisionProblem.
Also tests generator functions from sat_generator.
"""

from __future__ import annotations

import pytest

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
    generate_random_3sat_at_threshold,
)


# ---------------------------------------------------------------------------
# Variable
# ---------------------------------------------------------------------------


class TestVariable:
    """Tests for the Variable class."""

    def test_creation(self) -> None:
        """Variable stores its index."""
        v = Variable(1)
        assert v.index == 1

    def test_equality(self) -> None:
        """Variables with the same index are equal."""
        assert Variable(3) == Variable(3)

    def test_inequality(self) -> None:
        """Variables with different indices are not equal."""
        assert Variable(1) != Variable(2)

    def test_hashing(self) -> None:
        """Equal variables hash to the same value; usable in sets."""
        v_set = {Variable(1), Variable(2), Variable(1)}
        assert len(v_set) == 2

    def test_repr(self) -> None:
        """Repr includes the index."""
        assert "5" in repr(Variable(5))

    def test_invalid_index_zero(self) -> None:
        """Index 0 raises ValueError (must be positive)."""
        with pytest.raises(ValueError, match="positive"):
            Variable(0)

    def test_invalid_index_negative(self) -> None:
        """Negative index raises ValueError."""
        with pytest.raises(ValueError):
            Variable(-1)

    def test_invalid_index_string(self) -> None:
        """Non-integer index raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            Variable("a")  # type: ignore[arg-type]

    def test_not_equal_to_int(self) -> None:
        """Variable is not equal to a plain integer."""
        assert Variable(1) != 1


# ---------------------------------------------------------------------------
# Literal
# ---------------------------------------------------------------------------


class TestLiteral:
    """Tests for the Literal class."""

    def test_positive_literal(self) -> None:
        """Positive literal stores variable and polarity."""
        v = Variable(1)
        lit = Literal(v, positive=True)
        assert lit.variable == v
        assert lit.positive is True

    def test_negative_literal(self) -> None:
        """Negative literal has positive=False."""
        lit = Literal(Variable(2), positive=False)
        assert lit.positive is False

    def test_negation(self) -> None:
        """Negation flips polarity, keeps same variable."""
        lit = Literal(Variable(1), positive=True)
        neg = lit.negation()
        assert neg.positive is False
        assert neg.variable == lit.variable

    def test_double_negation(self) -> None:
        """Double negation returns to original polarity."""
        lit = Literal(Variable(3), positive=True)
        assert lit.negation().negation() == lit

    def test_evaluate_positive_true(self) -> None:
        """Positive literal evaluates True when variable is True."""
        lit = Literal(Variable(1), positive=True)
        assert lit.evaluate({1: True}) is True

    def test_evaluate_positive_false(self) -> None:
        """Positive literal evaluates False when variable is False."""
        lit = Literal(Variable(1), positive=True)
        assert lit.evaluate({1: False}) is False

    def test_evaluate_negative_true(self) -> None:
        """Negative literal evaluates True when variable is False."""
        lit = Literal(Variable(1), positive=False)
        assert lit.evaluate({1: False}) is True

    def test_evaluate_negative_false(self) -> None:
        """Negative literal evaluates False when variable is True."""
        lit = Literal(Variable(1), positive=False)
        assert lit.evaluate({1: True}) is False

    def test_evaluate_missing_variable_raises(self) -> None:
        """Evaluating with missing variable raises KeyError."""
        lit = Literal(Variable(5), positive=True)
        with pytest.raises(KeyError):
            lit.evaluate({1: True})

    def test_equality(self) -> None:
        """Same variable and polarity means equal literals."""
        a = Literal(Variable(1), True)
        b = Literal(Variable(1), True)
        assert a == b

    def test_inequality_polarity(self) -> None:
        """Different polarity means unequal literals."""
        a = Literal(Variable(1), True)
        b = Literal(Variable(1), False)
        assert a != b

    def test_hashing(self) -> None:
        """Equal literals hash to the same value."""
        a = Literal(Variable(1), True)
        b = Literal(Variable(1), True)
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Clause
# ---------------------------------------------------------------------------


class TestClause:
    """Tests for the Clause class."""

    def test_evaluate_all_true(self) -> None:
        """Clause with all-True literals evaluates True."""
        # (x1 OR x2) with x1=True, x2=True
        c = Clause((Literal(Variable(1), True), Literal(Variable(2), True)))
        assert c.evaluate({1: True, 2: True}) is True

    def test_evaluate_all_false(self) -> None:
        """Clause with all-False literals evaluates False."""
        # (x1 OR x2) with x1=False, x2=False
        c = Clause((Literal(Variable(1), True), Literal(Variable(2), True)))
        assert c.evaluate({1: False, 2: False}) is False

    def test_evaluate_mixed(self) -> None:
        """Clause with one True literal evaluates True."""
        # (x1 OR ~x2) with x1=False, x2=False -> ~x2=True
        c = Clause((
            Literal(Variable(1), True),
            Literal(Variable(2), False),
        ))
        assert c.evaluate({1: False, 2: False}) is True

    def test_is_unit_true(self) -> None:
        """Single-literal clause is a unit clause."""
        c = Clause((Literal(Variable(1), True),))
        assert c.is_unit() is True

    def test_is_unit_false(self) -> None:
        """Multi-literal clause is not a unit clause."""
        c = Clause((Literal(Variable(1), True), Literal(Variable(2), False)))
        assert c.is_unit() is False

    def test_is_empty_true(self) -> None:
        """Clause with no literals is empty."""
        c = Clause(())
        assert c.is_empty() is True

    def test_is_empty_false(self) -> None:
        """Clause with literals is not empty."""
        c = Clause((Literal(Variable(1), True),))
        assert c.is_empty() is False

    def test_empty_clause_evaluates_false(self) -> None:
        """Empty clause is unsatisfiable by convention."""
        c = Clause(())
        assert c.evaluate({}) is False

    def test_len(self) -> None:
        """len() returns number of literals."""
        c = Clause((Literal(Variable(1), True), Literal(Variable(2), True)))
        assert len(c) == 2

    def test_iter(self) -> None:
        """Clause is iterable over its literals."""
        lits = (Literal(Variable(1), True), Literal(Variable(2), False))
        c = Clause(lits)
        assert list(c) == list(lits)

    def test_get_variables(self) -> None:
        """get_variables returns the set of variables in the clause."""
        c = Clause((
            Literal(Variable(1), True),
            Literal(Variable(3), False),
        ))
        variables = c.get_variables()
        assert Variable(1) in variables
        assert Variable(3) in variables
        assert len(variables) == 2


# ---------------------------------------------------------------------------
# CNFFormula
# ---------------------------------------------------------------------------


class TestCNFFormula:
    """Tests for the CNFFormula class."""

    def test_evaluate_satisfiable(self, simple_sat_formula, known_satisfying_assignment) -> None:
        """Formula evaluates True under a satisfying assignment."""
        assert simple_sat_formula.evaluate(known_satisfying_assignment) is True

    def test_evaluate_unsatisfiable(self, simple_unsat_formula) -> None:
        """UNSAT formula evaluates False under all assignments for single var."""
        assert simple_unsat_formula.evaluate({1: True}) is False
        assert simple_unsat_formula.evaluate({1: False}) is False

    def test_empty_formula_is_true(self, empty_formula) -> None:
        """Empty formula is vacuously True."""
        assert empty_formula.evaluate({}) is True

    def test_num_variables(self, simple_sat_formula) -> None:
        """num_variables returns correct count."""
        assert simple_sat_formula.num_variables == 3

    def test_num_clauses(self, simple_sat_formula) -> None:
        """num_clauses returns correct count."""
        assert simple_sat_formula.num_clauses == 2

    def test_clause_variable_ratio(self) -> None:
        """clause_variable_ratio is computed correctly."""
        # 4 clauses, 2 variables -> ratio = 2.0
        x1 = Variable(1)
        x2 = Variable(2)
        clauses = (
            Clause((Literal(x1, True),)),
            Clause((Literal(x1, False),)),
            Clause((Literal(x2, True),)),
            Clause((Literal(x2, False),)),
        )
        f = CNFFormula(clauses)
        assert f.clause_variable_ratio == pytest.approx(2.0)

    def test_clause_variable_ratio_no_variables(self, empty_formula) -> None:
        """Ratio is 0.0 when there are no variables."""
        assert empty_formula.clause_variable_ratio == 0.0

    def test_get_variables(self, simple_sat_formula) -> None:
        """get_variables returns all variables in the formula."""
        variables = simple_sat_formula.get_variables()
        indices = {v.index for v in variables}
        assert indices == {1, 2, 3}

    def test_to_dimacs_from_dimacs_roundtrip(self, simple_sat_formula) -> None:
        """to_dimacs -> from_dimacs produces an equivalent formula."""
        dimacs_text = simple_sat_formula.to_dimacs()
        reconstructed = CNFFormula.from_dimacs(dimacs_text)

        # Same number of clauses and variables
        assert reconstructed.num_clauses == simple_sat_formula.num_clauses
        assert reconstructed.num_variables == simple_sat_formula.num_variables

        # Same evaluation on known assignment
        assignment = {1: True, 2: True, 3: True}
        assert reconstructed.evaluate(assignment) == simple_sat_formula.evaluate(assignment)

    def test_to_dimacs_format(self, simple_sat_formula) -> None:
        """DIMACS output has correct header format."""
        dimacs = simple_sat_formula.to_dimacs()
        lines = dimacs.strip().splitlines()
        header = lines[0]
        assert header.startswith("p cnf")
        parts = header.split()
        assert int(parts[2]) == 3  # max variable index
        assert int(parts[3]) == 2  # number of clauses

    def test_from_dimacs_invalid_header(self) -> None:
        """from_dimacs raises on invalid header."""
        with pytest.raises(ValueError, match="header"):
            CNFFormula.from_dimacs("p sat 3 2\n1 2 0")

    def test_from_dimacs_missing_header(self) -> None:
        """from_dimacs raises when header is missing."""
        with pytest.raises(ValueError, match="header"):
            CNFFormula.from_dimacs("1 2 0\n-1 3 0")

    def test_from_dimacs_wrong_clause_count(self) -> None:
        """from_dimacs raises when clause count mismatches."""
        with pytest.raises(ValueError, match="Expected"):
            CNFFormula.from_dimacs("p cnf 3 5\n1 2 0")

    def test_roundtrip_medium(self, medium_sat_formula) -> None:
        """DIMACS round-trip works for medium formulas."""
        dimacs_text = medium_sat_formula.to_dimacs()
        reconstructed = CNFFormula.from_dimacs(dimacs_text)
        assert reconstructed.num_clauses == medium_sat_formula.num_clauses

    def test_equality(self) -> None:
        """Two CNFFormulas with same clauses are equal."""
        c = Clause((Literal(Variable(1), True),))
        f1 = CNFFormula((c,))
        f2 = CNFFormula((c,))
        assert f1 == f2

    def test_hash_consistency(self) -> None:
        """Equal formulas hash to the same value."""
        c = Clause((Literal(Variable(1), True),))
        f1 = CNFFormula((c,))
        f2 = CNFFormula((c,))
        assert hash(f1) == hash(f2)


# ---------------------------------------------------------------------------
# SATDecisionProblem
# ---------------------------------------------------------------------------


class TestSATDecisionProblem:
    """Tests for the SATDecisionProblem concrete DecisionProblem."""

    def test_name(self) -> None:
        """Name includes 'SAT'."""
        problem = SATDecisionProblem()
        assert "SAT" in problem.name

    def test_encode_returns_dimacs(self, simple_sat_formula) -> None:
        """encode() returns a DIMACS string."""
        problem = SATDecisionProblem()
        encoded = problem.encode(simple_sat_formula)
        assert "p cnf" in encoded

    def test_encode_type_check(self) -> None:
        """encode() raises TypeError on non-CNFFormula input."""
        problem = SATDecisionProblem()
        with pytest.raises(TypeError):
            problem.encode("not a formula")

    def test_decide_sat(self, simple_sat_formula) -> None:
        """decide() returns True for satisfiable formula."""
        problem = SATDecisionProblem()
        assert problem.decide(simple_sat_formula) is True

    def test_decide_unsat(self, simple_unsat_formula) -> None:
        """decide() returns False for unsatisfiable formula."""
        problem = SATDecisionProblem()
        assert problem.decide(simple_unsat_formula) is False

    def test_decide_empty(self, empty_formula) -> None:
        """decide() returns True for empty formula."""
        problem = SATDecisionProblem()
        assert problem.decide(empty_formula) is True

    def test_verify_correct_certificate(
        self, simple_sat_formula, known_satisfying_assignment
    ) -> None:
        """verify() returns True for a valid certificate."""
        problem = SATDecisionProblem()
        assert problem.verify(simple_sat_formula, known_satisfying_assignment) is True

    def test_verify_incorrect_certificate(self, simple_sat_formula) -> None:
        """verify() returns False for an invalid certificate."""
        problem = SATDecisionProblem()
        bad_assignment = {1: False, 2: True, 3: False}
        # (x1 OR ~x2) -> x1=F, ~x2=F -> False
        assert problem.verify(simple_sat_formula, bad_assignment) is False

    def test_verify_type_check_instance(self) -> None:
        """verify() raises TypeError for non-CNFFormula instance."""
        problem = SATDecisionProblem()
        with pytest.raises(TypeError):
            problem.verify("not a formula", {})

    def test_verify_type_check_certificate(self, simple_sat_formula) -> None:
        """verify() raises TypeError for non-dict certificate."""
        problem = SATDecisionProblem()
        with pytest.raises(TypeError):
            problem.verify(simple_sat_formula, "not a dict")


# ---------------------------------------------------------------------------
# Generator tests (sat_generator via sat_types)
# ---------------------------------------------------------------------------


class TestGenerators:
    """Tests for SAT instance generators."""

    def test_random_ksat_size(self) -> None:
        """generate_random_ksat produces correct number of clauses and vars."""
        formula = generate_random_ksat(k=3, num_vars=10, num_clauses=20, seed=0)
        assert formula.num_clauses == 20
        assert formula.num_variables <= 10

    def test_random_ksat_clause_width(self) -> None:
        """Each clause in random k-SAT has exactly k literals."""
        formula = generate_random_ksat(k=3, num_vars=8, num_clauses=15, seed=1)
        for clause in formula.clauses:
            assert len(clause) == 3

    def test_satisfiable_instance_verifies(self) -> None:
        """Planted satisfiable instance is actually satisfiable."""
        formula, assignment = generate_satisfiable_instance(
            num_vars=10, num_clauses=30, k=3, seed=42
        )
        assert formula.evaluate(assignment) is True

    def test_satisfiable_instance_seed_reproducible(self) -> None:
        """Same seed produces the same formula and assignment."""
        f1, a1 = generate_satisfiable_instance(num_vars=5, num_clauses=10, seed=99)
        f2, a2 = generate_satisfiable_instance(num_vars=5, num_clauses=10, seed=99)
        assert f1 == f2
        assert a1 == a2

    def test_threshold_ratio(self) -> None:
        """generate_random_3sat_at_threshold produces ratio near 4.267."""
        formula = generate_random_3sat_at_threshold(num_vars=20, seed=0)
        ratio = formula.clause_variable_ratio
        assert abs(ratio - 4.267) < 0.5  # generous tolerance for rounding

    def test_threshold_too_few_vars(self) -> None:
        """generate_random_3sat_at_threshold raises for num_vars < 3."""
        with pytest.raises(ValueError):
            generate_random_3sat_at_threshold(num_vars=2, seed=0)
