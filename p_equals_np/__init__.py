"""P=NP Research Exploration: rigorous computational investigation of P vs NP.

This package implements a structured exploration of the P vs NP problem
through the lens of Boolean Satisfiability (SAT), the canonical NP-complete
problem established by the Cook-Levin theorem (1971).

The package provides:

- **Formal definitions**: Executable representations of decision problems,
  complexity classes (P, NP, NP-complete, NP-hard), polynomial-time
  reductions, and solver protocols.

- **SAT data types**: Immutable, memory-efficient representations of
  Boolean variables, literals, clauses, and CNF formulas with DIMACS
  serialization support.

- **Instance generators**: Random k-SAT (including at the phase transition
  threshold), planted satisfiable instances, guaranteed unsatisfiable
  instances, and structured instances from combinatorial problems.

- **Solvers**: Brute-force baseline (exponential), DPLL with unit
  propagation (exponential worst-case, practical pruning), and
  experimental polynomial-time attempts.

- **Complexity analysis**: Empirical runtime measurement and polynomial
  vs exponential scaling detection.

Intellectual honesty note:
    The honest expectation is that no polynomial-time SAT algorithm will
    be found. The value of this project lies in the rigor of the exploration
    and understanding WHY each approach fails to achieve polynomial time.
"""

__version__ = "1.1.0"

# --- Formal complexity definitions ---
from p_equals_np.definitions import (
    ComplexityClass,
    DecisionProblem,
    PolynomialReduction,
    Solver,
    SolverBenchmark,
    is_polynomial,
    measure_scaling,
)

# --- SAT data types ---
from p_equals_np.sat_types import (
    Clause,
    CNFFormula,
    Literal,
    SATDecisionProblem,
    Variable,
)

# --- Instance generators ---
from p_equals_np.sat_generator import (
    generate_random_3sat_at_threshold,
    generate_random_ksat,
    generate_satisfiable_instance,
    generate_structured_instance,
    generate_unsatisfiable_instance,
)

# --- LLM Oracle solver ---
from p_equals_np.experimental.llm_oracle_approach import (
    LLMOracleSolver,
    MockLLMBackend,
)
from p_equals_np.experimental.oracle_scaling_experiment import (
    OracleScalingExperiment,
)

__all__ = [
    # Version
    "__version__",
    # Formal definitions
    "DecisionProblem",
    "ComplexityClass",
    "PolynomialReduction",
    "Solver",
    "SolverBenchmark",
    "is_polynomial",
    "measure_scaling",
    # SAT data types
    "Variable",
    "Literal",
    "Clause",
    "CNFFormula",
    "SATDecisionProblem",
    # Instance generators
    "generate_random_ksat",
    "generate_random_3sat_at_threshold",
    "generate_satisfiable_instance",
    "generate_unsatisfiable_instance",
    "generate_structured_instance",
    # LLM Oracle solver
    "LLMOracleSolver",
    "MockLLMBackend",
    "OracleScalingExperiment",
]
