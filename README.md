# AI Framework Demo

Production-ready algorithm solutions built entirely by AI from a single prompt. Each project in this repository was generated end-to-end -- architecture, implementation, tests, and documentation -- without human intervention beyond the initial instruction. 

For the demo of this system's ability to review and remediate code changes see [https://github.com/mattsmcknight/xai-sdk-python-AI-Review-Demo](https://github.com/mattsmcknight/xai-sdk-python-AI-Review-Demo). Demo includes reports and a PR to demonstrate automated code changes.
For other domains see the included [ai-system-domain-capabilities.md](ai-system-domain-capabilities.md)

## What This Demonstrates

A single natural-language prompt produces a complete, production-quality codebase. The AI Agent handles everything: planning the work, writing the code, writing comprehensive tests, finding and fixing its own bugs, and documenting the result.

## Example Prompt

> I want the Agent to  create a sudoku solver using the dancing links algorithm in python in ./sudoku_solver. It should create production ready code.

That single prompt produced everything below -- no further human input required.

## Projects

### `sudoku_solver/` -- Dancing Links (DLX) / Algorithm X

A complete Sudoku solver implementing Knuth's Dancing Links algorithm in pure Python.

| Metric | Value |
|--------|-------|
| Files | 21 |
| Lines of code | 5,224 (1,889 source + 3,335 test) |
| Tests | 174 passing |
| External dependencies | 0 |
| Bugs found & fixed by AI | 1 critical (generator/matrix state corruption) |

**Highlights**:
- Generic exact cover solver separated from Sudoku-specific logic (reusable for any exact cover problem)
- Full CLI with stdin/file/argument input, multiple output formats
- Performance benchmarks including "AI Escargot" (world's hardest class)
- Input validation, logging, metrics collection
- Type-annotated, linted, documented

### `p_equals_np/` -- Computational Exploration of P vs NP

A rigorous computational exploration of the P vs NP problem through Boolean Satisfiability (SAT). Implements formal complexity definitions, multiple SAT solvers, empirical scaling analysis, and a detailed research write-up of why each approach fails to achieve polynomial time.

| Metric | Value |
|--------|-------|
| Files | 21 |
| Lines of code | 10,227 (6,737 source + 3,490 test) |
| Tests | 402 passing |
| External dependencies | 0 (numpy/matplotlib optional) |
| Solvers implemented | 6 (BruteForce, DPLL, Algebraic, Spectral, LP Relaxation, Structural) |

**Highlights**:
- Formal executable definitions of P, NP, NP-completeness, and polynomial reductions
- Six SAT solvers spanning four mathematical frameworks (algebraic, spectral, geometric, structural)
- Random 3-SAT instance generation at the phase transition (clause-to-variable ratio ~4.267)
- Empirical scaling analysis with polynomial vs exponential curve fitting
- Structured instances from combinatorial problems (pigeonhole, graph coloring, XOR chains)
- DIMACS format serialization/parsing for interoperability
- Comprehensive research analysis documenting why each approach hits an exponential wall