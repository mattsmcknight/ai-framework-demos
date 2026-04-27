"""SAT data types: Variable, Literal, Clause, CNFFormula, and SATDecisionProblem.

Provides immutable, memory-efficient representations of Boolean satisfiability
problem components following the DIMACS CNF conventions. Variables are 1-indexed
(positive integers), and literals are signed integers where negative values
represent negation.

These types form the core data structures for the P=NP research exploration,
consumed by all solvers, generators, and analysis tools.

Example:
    >>> x1 = Variable(1)
    >>> lit = Literal(x1, positive=True)
    >>> clause = Clause((lit, Literal(Variable(2), positive=False)))
    >>> formula = CNFFormula((clause,))
    >>> formula.evaluate({1: True, 2: False})
    True
"""

from __future__ import annotations

import itertools
from typing import Any, Optional

from p_equals_np.definitions import DecisionProblem


# ---------------------------------------------------------------------------
# Variable
# ---------------------------------------------------------------------------


class Variable:
    """A Boolean variable identified by a positive integer index (DIMACS).

    Variables are 1-indexed following the DIMACS CNF convention. Two variables
    are equal if and only if they have the same index.

    Attributes:
        index: Positive integer identifying this variable.
    """

    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        """Create a variable with the given index.

        Args:
            index: Positive integer variable identifier.

        Raises:
            ValueError: If index is not a positive integer.
        """
        if not isinstance(index, int) or index < 1:
            raise ValueError(
                f"Variable index must be a positive integer, got {index!r}"
            )
        self.index = index

    def __repr__(self) -> str:
        return f"Variable({self.index})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Variable):
            return NotImplemented
        return self.index == other.index

    def __hash__(self) -> int:
        return hash(self.index)


# ---------------------------------------------------------------------------
# Literal
# ---------------------------------------------------------------------------


class Literal:
    """A Boolean literal: a variable or its negation.

    A positive literal represents the variable itself; a negative literal
    represents the negation. Displayed as "x3" (positive) or "~x3" (negative).

    Attributes:
        variable: The underlying Boolean variable.
        positive: True if this literal is the variable itself, False if negated.
    """

    __slots__ = ("variable", "positive")

    def __init__(self, variable: Variable, positive: bool = True) -> None:
        """Create a literal from a variable and polarity.

        Args:
            variable: The Boolean variable.
            positive: True for positive literal, False for negation.
        """
        self.variable = variable
        self.positive = positive

    def negation(self) -> Literal:
        """Return the negation of this literal.

        Returns:
            A new Literal with the opposite polarity.
        """
        return Literal(self.variable, not self.positive)

    def evaluate(self, assignment: dict[int, bool]) -> bool:
        """Evaluate this literal under a truth assignment.

        Args:
            assignment: Mapping from variable index to truth value.

        Returns:
            The truth value of this literal under the assignment.

        Raises:
            KeyError: If the variable's index is not in the assignment.
        """
        value = assignment[self.variable.index]
        return value if self.positive else not value

    def __repr__(self) -> str:
        prefix = "" if self.positive else "~"
        return f"{prefix}x{self.variable.index}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Literal):
            return NotImplemented
        return self.variable == other.variable and self.positive == other.positive

    def __hash__(self) -> int:
        return hash((self.variable.index, self.positive))


# ---------------------------------------------------------------------------
# Clause
# ---------------------------------------------------------------------------


class Clause:
    """A disjunction (OR) of literals.

    A clause is satisfied if at least one of its literals evaluates to True.
    An empty clause is unsatisfiable by convention.

    Attributes:
        literals: Tuple of literals in this clause.
    """

    __slots__ = ("literals",)

    def __init__(self, literals: tuple[Literal, ...]) -> None:
        """Create a clause from a tuple of literals.

        Args:
            literals: The literals forming the disjunction.
        """
        self.literals = literals

    def evaluate(self, assignment: dict[int, bool]) -> bool:
        """Evaluate this clause under a truth assignment.

        A clause is true if at least one literal is true. An empty
        clause is false (unsatisfiable by convention).

        Args:
            assignment: Mapping from variable index to truth value.

        Returns:
            True if the clause is satisfied under the assignment.
        """
        return any(lit.evaluate(assignment) for lit in self.literals)

    def is_unit(self) -> bool:
        """Check whether this clause contains exactly one literal.

        Returns:
            True if this is a unit clause.
        """
        return len(self.literals) == 1

    def is_empty(self) -> bool:
        """Check whether this clause contains no literals.

        Returns:
            True if this clause is empty (always unsatisfiable).
        """
        return len(self.literals) == 0

    def get_variables(self) -> frozenset[Variable]:
        """Return the set of variables appearing in this clause.

        Returns:
            A frozenset of Variable objects.
        """
        return frozenset(lit.variable for lit in self.literals)

    def __len__(self) -> int:
        return len(self.literals)

    def __iter__(self):
        return iter(self.literals)

    def __repr__(self) -> str:
        if not self.literals:
            return "Clause()"
        inner = " | ".join(repr(lit) for lit in self.literals)
        return f"({inner})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Clause):
            return NotImplemented
        return self.literals == other.literals

    def __hash__(self) -> int:
        return hash(self.literals)


# ---------------------------------------------------------------------------
# CNFFormula
# ---------------------------------------------------------------------------


class CNFFormula:
    """A Boolean formula in Conjunctive Normal Form (CNF).

    A CNF formula is a conjunction (AND) of clauses, where each clause is
    a disjunction (OR) of literals. The formula is satisfied when every
    clause is satisfied.

    Attributes:
        clauses: Tuple of clauses forming the conjunction.
    """

    __slots__ = ("clauses",)

    def __init__(self, clauses: tuple[Clause, ...]) -> None:
        """Create a CNF formula from a tuple of clauses.

        Args:
            clauses: The clauses forming the conjunction.
        """
        self.clauses = clauses

    @property
    def num_variables(self) -> int:
        """Return the number of distinct variables in the formula."""
        return len(self.get_variables())

    @property
    def num_clauses(self) -> int:
        """Return the number of clauses in the formula."""
        return len(self.clauses)

    @property
    def clause_variable_ratio(self) -> float:
        """Return the clause-to-variable ratio.

        This ratio is significant for random k-SAT: for 3-SAT, the
        satisfiability threshold occurs near ratio 4.267.

        Returns:
            The ratio of clauses to variables, or 0.0 if no variables.
        """
        n_vars = self.num_variables
        if n_vars == 0:
            return 0.0
        return self.num_clauses / n_vars

    def evaluate(self, assignment: dict[int, bool]) -> bool:
        """Evaluate the formula under a truth assignment.

        A CNF formula is true if and only if every clause is true.
        An empty formula (no clauses) is vacuously true.

        Args:
            assignment: Mapping from variable index to truth value.

        Returns:
            True if the formula is satisfied under the assignment.
        """
        return all(clause.evaluate(assignment) for clause in self.clauses)

    def get_variables(self) -> frozenset[Variable]:
        """Return the set of all variables appearing in the formula.

        Returns:
            A frozenset of Variable objects.
        """
        variables: set[Variable] = set()
        for clause in self.clauses:
            variables.update(clause.get_variables())
        return frozenset(variables)

    def to_dimacs(self) -> str:
        """Serialize this formula to DIMACS CNF format.

        The DIMACS format is:
            p cnf <num_vars> <num_clauses>
            <literal> <literal> ... 0
            ...

        Positive literals are represented by their variable index,
        negative literals by the negation of their variable index.

        Returns:
            A string in DIMACS CNF format.
        """
        var_indices = sorted(v.index for v in self.get_variables())
        max_var = var_indices[-1] if var_indices else 0
        lines = [f"p cnf {max_var} {self.num_clauses}"]
        for clause in self.clauses:
            tokens = []
            for lit in clause.literals:
                dimacs_val = lit.variable.index if lit.positive else -lit.variable.index
                tokens.append(str(dimacs_val))
            tokens.append("0")
            lines.append(" ".join(tokens))
        return "\n".join(lines)

    @staticmethod
    def from_dimacs(text: str) -> CNFFormula:
        """Parse a CNF formula from DIMACS format text.

        Args:
            text: A string in DIMACS CNF format.

        Returns:
            A CNFFormula parsed from the text.

        Raises:
            ValueError: If the text is not valid DIMACS CNF format.
        """
        clauses: list[Clause] = []
        expected_vars = 0
        expected_clauses = 0
        found_header = False

        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) < 4 or parts[1] != "cnf":
                    raise ValueError(f"Invalid DIMACS header: {line!r}")
                expected_vars = int(parts[2])
                expected_clauses = int(parts[3])
                found_header = True
                continue
            if not found_header:
                raise ValueError("DIMACS clause line before header")
            tokens = line.split()
            literals: list[Literal] = []
            for token in tokens:
                val = int(token)
                if val == 0:
                    break
                var = Variable(abs(val))
                literals.append(Literal(var, positive=(val > 0)))
            clauses.append(Clause(tuple(literals)))

        if not found_header:
            raise ValueError("No DIMACS header found (missing 'p cnf ...' line)")

        if len(clauses) != expected_clauses:
            raise ValueError(
                f"Expected {expected_clauses} clauses from header, "
                f"got {len(clauses)}"
            )

        return CNFFormula(tuple(clauses))

    def __repr__(self) -> str:
        if not self.clauses:
            return "CNFFormula(empty)"
        inner = " & ".join(repr(c) for c in self.clauses)
        return f"CNFFormula({inner})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CNFFormula):
            return NotImplemented
        return self.clauses == other.clauses

    def __hash__(self) -> int:
        return hash(self.clauses)


# ---------------------------------------------------------------------------
# SATDecisionProblem
# ---------------------------------------------------------------------------


class SATDecisionProblem(DecisionProblem):
    """The Boolean Satisfiability decision problem (SAT).

    Given a CNF formula, determine whether there exists a truth assignment
    to its variables that satisfies the formula. This is the canonical
    NP-complete problem (Cook-Levin theorem, 1971).

    - ``encode``: Serializes a CNFFormula to DIMACS format.
    - ``decide``: Brute-force enumeration of all 2^n assignments (exponential).
    - ``verify``: Evaluates the formula under a proposed assignment (polynomial).
    """

    @property
    def name(self) -> str:
        """Human-readable name of this decision problem."""
        return "Boolean Satisfiability (SAT)"

    def encode(self, instance: Any) -> str:
        """Encode a CNFFormula instance as a DIMACS string.

        Args:
            instance: A CNFFormula object.

        Returns:
            The DIMACS CNF string representation.

        Raises:
            TypeError: If instance is not a CNFFormula.
        """
        if not isinstance(instance, CNFFormula):
            raise TypeError(
                f"Expected CNFFormula, got {type(instance).__name__}"
            )
        return instance.to_dimacs()

    def decide(self, instance: Any) -> bool:
        """Decide SAT by brute-force enumeration of all truth assignments.

        Enumerates all 2^n possible assignments and checks each one.
        This is an exponential-time algorithm, serving as the baseline
        for correctness.

        Args:
            instance: A CNFFormula object.

        Returns:
            True if the formula is satisfiable, False otherwise.

        Raises:
            TypeError: If instance is not a CNFFormula.
        """
        if not isinstance(instance, CNFFormula):
            raise TypeError(
                f"Expected CNFFormula, got {type(instance).__name__}"
            )
        var_indices = sorted(v.index for v in instance.get_variables())
        if not var_indices:
            return instance.evaluate({})

        for values in itertools.product((False, True), repeat=len(var_indices)):
            assignment = dict(zip(var_indices, values))
            if instance.evaluate(assignment):
                return True
        return False

    def verify(self, instance: Any, certificate: Any) -> bool:
        """Verify a proposed truth assignment satisfies the formula.

        This runs in polynomial time: it simply evaluates the formula
        under the given assignment. This polynomial-time verification
        is what places SAT in NP.

        Args:
            instance: A CNFFormula object.
            certificate: A dict mapping variable indices (int) to
                truth values (bool).

        Returns:
            True if the assignment satisfies the formula.

        Raises:
            TypeError: If instance is not a CNFFormula or certificate
                is not a dict.
        """
        if not isinstance(instance, CNFFormula):
            raise TypeError(
                f"Expected CNFFormula, got {type(instance).__name__}"
            )
        if not isinstance(certificate, dict):
            raise TypeError(
                f"Expected dict for certificate, got {type(certificate).__name__}"
            )
        return instance.evaluate(certificate)
