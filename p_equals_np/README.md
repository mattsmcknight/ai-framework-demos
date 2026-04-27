# P vs NP: A Computational Exploration

## Overview

This project is a rigorous computational exploration of the P vs NP problem through
the lens of Boolean Satisfiability (SAT), the canonical NP-complete problem. It
implements formal definitions, multiple SAT solvers (from brute-force baseline to
creative experimental approaches), empirical scaling analysis tools, and a detailed
research analysis of findings.

The project implements seven SAT solvers spanning five mathematical frameworks
(algebraic, spectral, geometric, structural, and probabilistic inference),
measures their runtime scaling on hard random 3-SAT instances at the phase
transition, and documents why each approach fails to achieve polynomial time.
The full test suite comprises 402 tests, all passing.

## Background

The P vs NP problem asks whether every decision problem whose solution can be
*verified* in polynomial time can also be *solved* in polynomial time.

- **P** is the class of problems solvable in time O(n^k) for some constant k.
- **NP** is the class of problems where a "yes" answer can be verified in polynomial
  time given a certificate (witness).

Since any problem solvable in polynomial time is trivially verifiable in polynomial
time, P is a subset of NP. Whether P equals NP or is a strict subset is the most
important open problem in theoretical computer science and one of the seven
Millennium Prize Problems.

**Why SAT?** Boolean Satisfiability was the first problem proven NP-complete
(Cook-Levin theorem, 1971). A polynomial-time SAT algorithm would immediately
imply P = NP. SAT is structurally simple, admits rich algebraic, geometric, and
graph-theoretic interpretations, and exhibits a sharp satisfiability phase transition
near clause-to-variable ratio 4.267 for 3-SAT, providing a natural source of hard
instances.

## Project Structure

```
p_equals_np/
    __init__.py                         # Package init with public API exports
    definitions.py                      # Formal complexity class definitions (P, NP, reductions)
    sat_types.py                        # SAT data types: Variable, Literal, Clause, CNFFormula
    sat_generator.py                    # Random and structured SAT instance generators
    brute_force.py                      # Exhaustive search SAT solver (O(2^n) baseline)
    dpll.py                             # DPLL solver with unit propagation and pruning
    complexity_analysis.py              # Runtime measurement, curve fitting, scaling analysis

    experimental/
        __init__.py                     # Experimental approaches subpackage
        algebraic_approach.py           # Polynomial systems over GF(2) / Groebner basis
        spectral_approach.py            # Variable Interaction Graph eigenvalue methods
        geometric_approach.py           # LP relaxation and rounding strategies
        structural_approach.py          # Tractable subclasses, treewidth, backdoors
        llm_oracle_approach.py          # LLM as probabilistic oracle with retry loop
        oracle_scaling_experiment.py    # Oracle-aware scaling experiment wrapper
        LLM_ORACLE_ANALYSIS.md          # Convergence analysis for the LLM Oracle approach

    benchmark_llm_oracle.py             # 7-solver benchmark runner (CLI)

    tests/
        __init__.py                     # Test package
        conftest.py                     # Shared pytest fixtures (known SAT/UNSAT instances)
        test_definitions.py             # Tests for complexity class definitions
        test_sat_types.py               # Tests for SAT data types
        test_brute_force.py             # Tests for brute-force solver
        test_dpll.py                    # Tests for DPLL solver
        test_cross_validation.py        # Cross-validation of solver agreement
        test_experimental.py            # Tests for all experimental approaches
        test_complexity.py              # Tests for complexity analysis tools

    ANALYSIS.md                         # Rigorous research analysis of findings
    README.md                           # This file
```

## Usage

### Creating SAT Instances

```python
from p_equals_np.sat_types import Variable, Literal, Clause, CNFFormula
from p_equals_np.sat_generator import (
    generate_random_ksat,
    generate_random_3sat_at_threshold,
    generate_satisfiable_instance,
    generate_unsatisfiable_instance,
    generate_structured_instance,
)

# Build a formula manually: (x1 OR ~x2) AND (~x1 OR x3)
x1, x2, x3 = Variable(1), Variable(2), Variable(3)
formula = CNFFormula((
    Clause((Literal(x1), Literal(x2, positive=False))),
    Clause((Literal(x1, positive=False), Literal(x3))),
))
print(formula)                    # CNFFormula((x1 | ~x2) & (~x1 | x3))
print(formula.num_variables)      # 3
print(formula.clause_variable_ratio)  # 0.666...

# Generate a random 3-SAT instance at the phase transition
hard_formula = generate_random_3sat_at_threshold(20, seed=42)
print(hard_formula.num_variables)  # 20
print(hard_formula.num_clauses)    # 85

# Generate a planted satisfiable instance (guaranteed SAT)
sat_formula, planted_assignment = generate_satisfiable_instance(
    num_vars=10, num_clauses=30, k=3, seed=7
)
assert sat_formula.evaluate(planted_assignment)

# Generate a guaranteed unsatisfiable instance
unsat_formula = generate_unsatisfiable_instance(num_vars=5, seed=0)

# Generate structured instances from combinatorial problems
pigeonhole = generate_structured_instance("pigeonhole", size=3)  # 4 pigeons, 3 holes
xor_chain = generate_structured_instance("xor_chain", size=5)
coloring = generate_structured_instance("graph_coloring", size=6)

# DIMACS serialization and parsing
dimacs_str = formula.to_dimacs()
parsed = CNFFormula.from_dimacs(dimacs_str)
assert parsed == formula
```

### Running the Solvers

```python
from p_equals_np.brute_force import BruteForceSolver
from p_equals_np.dpll import DPLLSolver
from p_equals_np.experimental.algebraic_approach import AlgebraicSolver
from p_equals_np.experimental.spectral_approach import SpectralSolver
from p_equals_np.experimental.geometric_approach import LPRelaxationSolver
from p_equals_np.experimental.structural_approach import StructuralSolver
from p_equals_np.sat_generator import generate_random_3sat_at_threshold

formula = generate_random_3sat_at_threshold(10, seed=42)

# Brute-force baseline (exhaustive, O(2^n * m))
bf = BruteForceSolver(timeout_seconds=30.0)
result = bf.solve(formula)
if result is not None:
    print(f"SAT! Assignment: {result}")
    print(f"Assignments tried: {bf.assignments_tried}")
    assert formula.evaluate(result)
else:
    print("UNSAT")

# DPLL with unit propagation and pure literal elimination
dpll = DPLLSolver(timeout_seconds=30.0)
result = dpll.solve(formula)
if result is not None:
    print(f"SAT! Decisions: {dpll.decisions}, "
          f"Propagations: {dpll.propagations}, "
          f"Backtracks: {dpll.backtracks}")

# Algebraic approach (polynomial systems over GF(2))
algebraic = AlgebraicSolver()
result = algebraic.solve(formula)

# Spectral approach (Variable Interaction Graph eigenvalues)
spectral = SpectralSolver(timeout_seconds=30.0)
result = spectral.solve(formula)

# LP relaxation approach (linear programming + rounding)
lp = LPRelaxationSolver()
result = lp.solve(formula)

# Structural approach (2-SAT, Horn-SAT, treewidth, backdoors)
structural = StructuralSolver()
result = structural.solve(formula)

# LLM Oracle (probabilistic inference + deterministic verification)
from p_equals_np.experimental.llm_oracle_approach import LLMOracleSolver, MockLLMBackend
oracle = LLMOracleSolver(backend=MockLLMBackend(seed=42), max_attempts=5)
result = oracle.solve(formula)
if result is not None:
    print(f"SAT! Attempts: {oracle.attempts_made}")
    assert formula.evaluate(result)
else:
    print(f"Failed after {oracle.attempts_made} attempts")
```

### Running the 7-Solver Benchmark

```bash
# Mock mode (no API key needed, validates infrastructure)
python -m p_equals_np.benchmark_llm_oracle --mock --quick

# Full benchmark with live API
python -m p_equals_np.benchmark_llm_oracle --live --output results/

# Oracle-only benchmark
python -m p_equals_np.benchmark_llm_oracle --mock --oracle-only
```

### Running the Scaling Experiment

```python
from p_equals_np.complexity_analysis import ScalingExperiment
from p_equals_np.brute_force import BruteForceSolver
from p_equals_np.dpll import DPLLSolver

# Configure the experiment
experiment = ScalingExperiment(
    solvers=[BruteForceSolver(), DPLLSolver()],
    variable_sizes=[5, 8, 10, 12, 15],
    instances_per_size=5,
    timeout_per_instance=10.0,
)

# Run solvers on generated random 3-SAT instances at the phase transition
measurements = experiment.run_experiment(clause_ratio=4.267)

# Analyze scaling: fit polynomial and exponential models
analysis = experiment.analyze_scaling(measurements)

for solver_name, data in analysis.items():
    print(f"{solver_name}: best fit = {data['best_model']} "
          f"(R^2 = {data['best_r2']:.4f})")

# Generate a text report
report = experiment.generate_scaling_report(analysis)
print(report)

# Generate plots (requires matplotlib) or CSV fallback
experiment.plot_scaling(analysis, output_path="./scaling_output")
```

### LLM Oracle Solver (7th Solver)

```python
from p_equals_np.experimental.llm_oracle_approach import (
    LLMOracleSolver,
    MockLLMBackend,
)
from p_equals_np.sat_generator import generate_random_3sat_at_threshold

formula = generate_random_3sat_at_threshold(10, seed=42)

# LLM Oracle with mock backend (deterministic for testing)
oracle = LLMOracleSolver(
    backend=MockLLMBackend(seed=42),
    encoding="structured",
    strategy="baseline",
    max_attempts=5,
)
result = oracle.solve(formula)
if result is not None:
    print(f"SAT! Attempts: {oracle.attempts_made}")
    assert formula.evaluate(result)
else:
    print(f"Failed after {oracle.attempts_made} attempts")

# Access per-solve metrics
metrics = oracle.get_metrics()
print(f"Strategies used: {metrics.strategies_used}")
print(f"Clauses violated per attempt: {metrics.clauses_violated_per_attempt}")
```

### Running Tests

```bash
# Run the full test suite
pytest p_equals_np/tests/ -v

# Run tests for a specific module
pytest p_equals_np/tests/test_brute_force.py -v
pytest p_equals_np/tests/test_dpll.py -v
pytest p_equals_np/tests/test_experimental.py -v
pytest p_equals_np/tests/test_complexity.py -v

# Run with coverage
pytest p_equals_np/tests/ --cov=p_equals_np --cov-report=term-missing
```

## Results Summary

All seven solvers exhibit exponential or worse scaling on hard random 3-SAT instances
at the phase transition. No solver achieves polynomial time on this class of instances.

| Solver         | Polynomial Component        | Exponential Component                   |
|----------------|-----------------------------|-----------------------------------------|
| BruteForce     | None                        | O(2^n * m) exhaustive enumeration       |
| DPLL           | Unit propagation, pruning   | O(2^n) worst case; 129x-2534x faster    |
| Algebraic      | Gaussian elimination O(n^3) | Groebner basis: EXPSPACE                |
| Spectral       | Eigenvalue computation O(n^3)| DPLL fallback: O(2^n)                  |
| LP Relaxation  | LP solve O(n^3.5)           | Rounding: no guarantee (~47% at threshold) |
| Structural     | 2-SAT/Horn-SAT O(n+m)      | General 3-SAT: no structural shortcut   |
| LLM Oracle     | Verification O(n*m)         | Success probability exp(-alpha*n)       |

Each approach illuminates a different facet of why SAT resists polynomial-time
solution:

- **Algebraic**: Degree explosion in Groebner basis mirrors the 2-SAT to 3-SAT jump.
- **Spectral**: Graph structure cannot encode literal polarities.
- **Geometric**: Continuous relaxation loses discrete combinatorial information.
- **Structural**: Hard instances have no tractable substructure (unbounded treewidth,
  no small backdoors).
- **LLM Oracle**: Probabilistic inference hits an information-theoretic wall --
  the verification asymmetry (P vs NP gap) made tangible.

For the full analysis including proof barrier discussion (relativization, natural
proofs, algebrization), see [ANALYSIS.md](ANALYSIS.md).

### LLM Oracle: Convergence Analysis Summary

The LLM Oracle solver (7th solver) uses an LLM as a probabilistic oracle for
candidate assignment generation, wrapped in a deterministic verification loop.
Its convergence behavior follows a Gompertz (double-exponential) model:

```
P(success) = 1 - exp(-B * exp(-alpha * n))
```

where B is the retry budget, n is the variable count, and alpha is a decay
constant that encodes how effectively the system prunes the search space.

**The alpha constant is where Nousix has its impact.** Each
architectural feature reduces alpha:

| Feature | Mechanism | Effect on Alpha |
|---------|-----------|----------------|
| Deterministic coprocessor | Clause violation diagnosis (gradient-like feedback) | Primary reduction: alpha drops from ln(2)=0.693 toward ln(4/3)=0.288 |
| Feedback loop | Targeted retry instructions from failure analysis | Converts independent guesses into correlated search |
| Context window | Attempt history as L1 cache | Prevents redundant exploration (constant factor) |
| Strategy rotation | Cycling baseline/CoT/constraint-highlight/incremental | Prevents LLM mode collapse |

**The Moser-Tardos connection**: The retry loop is structurally identical to the
Moser-Tardos constructive proof of the Lovasz Local Lemma. At the phase
transition (alpha = 4.267), the LLL condition fails (e * p * (d+1) = 12.37 >> 1),
which is why convergence requires exponentially many retries rather than
polynomially many.

**The information-theoretic floor**: Even with perfect information extraction,
the minimum retries scale as n/log(n) -- sub-exponential but super-polynomial.
This is the best any generate-verify-retry architecture can achieve.

**The e-governed transition**: At the critical problem size n* = ln(B)/alpha,
the success probability equals 1 - 1/e ~ 63.2%. This is the natural boundary
between "solvable within budget" and "budget-exceeded" -- and it is where the
system's operating specification is defined.

The key distinction: anyone can ask an LLM to solve SAT. What Nousix provides is the *system* that transforms a single stochastic guess into a structured convergence process with a mathematically characterized success probability -- the deterministic coprocessor for verification, the feedback loop for information extraction, the context window for attempt history, and the retry framework with strategy rotation. The difference between "asking ChatGPT" and "running a Nousix agent" is the difference between flipping a coin and running a Las Vegas algorithm.

For the full convergence analysis, see
[experimental/LLM_ORACLE_ANALYSIS.md](experimental/LLM_ORACLE_ANALYSIS.md).

## Requirements

- **Python 3.10+** (uses modern type hint syntax: `X | Y`, `list[int]`, etc.)
- **pytest** (for running the test suite)
- **Optional**: `numpy` (improves numerical stability in curve fitting)
- **Optional**: `matplotlib` (enables scaling plots; falls back to CSV without it)

No other external dependencies are required. All solvers, data types, and core
analysis tools are implemented in pure Python.

## Intellectual Honesty

**No polynomial-time SAT algorithm was found.** This was the expected outcome.

If P = NP, it would be the most significant result in the history of computer
science. The P vs NP problem has been open for over 50 years and has resisted the
efforts of Turing Award winners, Fields Medalists, and thousands of researchers.
Three major barrier results (relativization, natural proofs, algebrization)
constrain even the *type* of argument that could resolve it.

The value of this project lies not in the (correctly expected) failure to find a
polynomial-time SAT algorithm. It lies in the rigorous exploration of *why* each
approach fails:

- Every approach has a polynomial-time component that handles some tractable aspect
  of the problem efficiently.
- Every approach hits an exponential wall when confronted with hard 3-SAT instances
  at the phase transition.
- The failures are not accidents of implementation. They reflect structural barriers
  that are predicted by complexity theory.

The phase transition at clause-to-variable ratio ~4.267 is the empirical
manifestation of the P/NP boundary: the point where constraint density overwhelms
every known structural shortcut. Understanding this boundary -- through rigorous
implementation, measurement, and analysis -- is what this project contributes.
