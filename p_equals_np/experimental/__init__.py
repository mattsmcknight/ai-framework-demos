"""Experimental polynomial-time SAT solving approaches.

This subpackage contains creative attempts at polynomial-time algorithms
for Boolean Satisfiability. Each module implements a distinct approach
grounded in a different mathematical framework:

- **algebraic_approach**: Polynomial system / Groebner basis methods
- **spectral_approach**: Graph spectral / eigenvalue-based methods
- **geometric_approach**: Linear programming relaxation methods
- **structural_approach**: Exploiting formula structure (treewidth, backdoors)
- **llm_oracle_approach**: LLM as probabilistic oracle with deterministic verification

Each approach documents its theoretical basis, claimed complexity,
the conditions under which it might succeed, and an honest assessment
of where and why it breaks down on hard instances.

Important:
    The honest expectation is that none of these approaches will achieve
    genuine polynomial-time SAT solving. Their value lies in illuminating
    the structural barriers that separate P from NP.

    The LLM Oracle approach maps the P/NP boundary into the probabilistic
    domain: the oracle hits an *information-theoretic wall* rather than a
    *computational wall*.
"""

from p_equals_np.experimental.llm_oracle_approach import (
    LLMOracleSolver,
    MockLLMBackend,
    OracleAggregateMetrics,
    OracleSolveMetrics,
)
from p_equals_np.experimental.oracle_scaling_experiment import (
    OracleBenchmarkReport,
    OracleInstanceResult,
    OracleScalingExperiment,
)

__all__ = [
    "LLMOracleSolver",
    "MockLLMBackend",
    "OracleSolveMetrics",
    "OracleAggregateMetrics",
    "OracleScalingExperiment",
    "OracleInstanceResult",
    "OracleBenchmarkReport",
]
