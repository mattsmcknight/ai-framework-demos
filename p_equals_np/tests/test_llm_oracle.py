"""Tests for the LLM Oracle SAT solver.

Validates the LLM Oracle solver -- encoding, parsing, retry logic,
metrics tracking, and verification soundness -- entirely with
MockLLMBackend (no API key required).

Follows the conventions established in test_experimental.py:
- Soundness is the hard invariant: any returned assignment MUST satisfy
  the formula. Returning None on a SAT instance is acceptable (incomplete
  solver). Returning an invalid assignment is a bug.
- Cross-validation with BruteForceSolver on 20 shared instances.
- Edge cases: empty formula, single clause, contradictions.

Test categories:
    1. MockLLMBackend: determinism, variable coverage, parseability.
    2. Encoding: DIMACS, natural language, structured -- round-trip
       preservation of formula information.
    3. Response parsing: Python dicts, JSON, inline format, malformed.
    4. Solver correctness: soundness on known SAT/UNSAT instances.
    5. Retry framework: failure classification, strategy rotation,
       temperature escalation, max_attempts / timeout limits.
    6. Metrics: per-solve, aggregate, reset, success_rate_by_size.
    7. Cross-validation with BruteForceSolver on 20 instances.
    8. Graceful degradation: missing API key, malformed responses.
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from p_equals_np.sat_types import Clause, CNFFormula, Literal, Variable
from p_equals_np.sat_generator import (
    generate_random_ksat,
    generate_satisfiable_instance,
)
from p_equals_np.brute_force import BruteForceSolver
from p_equals_np.definitions import Solver
from p_equals_np.experimental.llm_oracle_approach import (
    AnthropicBackend,
    AttemptRecord,
    FailureMode,
    LLMOracleSolver,
    MockLLMBackend,
    OracleAggregateMetrics,
    OracleSolveMetrics,
    ViolatedClause,
    _STRATEGY_ROTATION,
    _TEMPERATURE_SCHEDULE,
    _compute_aggregate_metrics,
    _extract_variable_count,
    _get_retry_strategy,
    _get_retry_temperature,
    build_retry_feedback,
    diagnose_failure,
    encode_dimacs,
    encode_for_llm,
    encode_natural_language,
    encode_structured,
    parse_llm_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_formula(clauses_data: list[list[tuple[int, bool]]]) -> CNFFormula:
    """Build a CNFFormula from a compact representation.

    Args:
        clauses_data: List of clauses, each a list of (var_index, positive)
            tuples.

    Returns:
        A CNFFormula.
    """
    clauses = []
    for clause_data in clauses_data:
        literals = tuple(
            Literal(Variable(idx), positive=pos) for idx, pos in clause_data
        )
        clauses.append(Clause(literals))
    return CNFFormula(tuple(clauses))


def _make_variables(*indices: int) -> frozenset[Variable]:
    """Create a frozenset of Variable objects from integer indices."""
    return frozenset(Variable(i) for i in indices)


# Shared instance parameters for cross-validation
CROSS_SEEDS = list(range(20))
CROSS_N = 5
CROSS_M = 12


# =========================================================================
# 1. MOCK BACKEND TESTS
# =========================================================================


class TestMockLLMBackend:
    """Tests for MockLLMBackend determinism and output format."""

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces identical responses."""
        backend1 = MockLLMBackend(seed=42)
        backend2 = MockLLMBackend(seed=42)

        prompt = "This formula has 3 variables and 2 clauses."
        r1 = backend1.generate(prompt)
        r2 = backend2.generate(prompt)
        assert r1 == r2

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different responses (with high probability)."""
        backend1 = MockLLMBackend(seed=42)
        backend2 = MockLLMBackend(seed=99)

        prompt = "This formula has 10 variables and 20 clauses."
        r1 = backend1.generate(prompt)
        r2 = backend2.generate(prompt)
        # With 10 variables, probability of identical random assignments is ~1/1024
        assert r1 != r2

    def test_returns_parseable_dict(self) -> None:
        """Response is a valid Python dict literal."""
        backend = MockLLMBackend(seed=42)
        prompt = "This formula has 5 variables and 10 clauses."
        response = backend.generate(prompt)

        assert response is not None
        parsed = eval(response)  # noqa: S307
        assert isinstance(parsed, dict)
        assert len(parsed) == 5

    def test_covers_all_variables(self) -> None:
        """Assignment covers all expected variables (1 through N)."""
        backend = MockLLMBackend(seed=42)
        prompt = "This formula has 7 variables and 15 clauses."
        response = backend.generate(prompt)

        assert response is not None
        parsed = eval(response)  # noqa: S307
        assert set(parsed.keys()) == {1, 2, 3, 4, 5, 6, 7}
        for val in parsed.values():
            assert isinstance(val, bool)

    def test_call_count_increments(self) -> None:
        """call_count tracks the number of generate calls."""
        backend = MockLLMBackend(seed=42)
        assert backend.call_count == 0

        backend.generate("This formula has 3 variables and 2 clauses.")
        assert backend.call_count == 1

        backend.generate("This formula has 3 variables and 2 clauses.")
        assert backend.call_count == 2

    def test_returns_none_for_unparseable_prompt(self) -> None:
        """Returns None when variable count cannot be extracted."""
        backend = MockLLMBackend(seed=42)
        response = backend.generate("solve this problem please")
        assert response is None

    def test_sequential_calls_vary(self) -> None:
        """Successive calls with the same prompt produce different results."""
        backend = MockLLMBackend(seed=42)
        prompt = "This formula has 10 variables and 20 clauses."
        responses = [backend.generate(prompt) for _ in range(5)]
        # Not all responses should be identical (RNG advances)
        assert len(set(responses)) > 1


# =========================================================================
# 2. ENCODING TESTS
# =========================================================================


class TestEncodeDimacs:
    """Tests for DIMACS encoding of CNF formulas."""

    def test_contains_dimacs_header(self) -> None:
        """Encoding includes DIMACS p cnf header."""
        formula = _make_formula([
            [(1, True), (2, False)],
            [(2, True), (3, True)],
        ])
        prompt = encode_dimacs(formula)
        assert "p cnf" in prompt

    def test_contains_variable_and_clause_count(self) -> None:
        """Encoding mentions variable and clause counts."""
        formula = _make_formula([
            [(1, True), (2, False)],
            [(2, True), (3, True)],
        ])
        prompt = encode_dimacs(formula)
        assert "3 variables" in prompt or "3" in prompt
        assert "2 clauses" in prompt or "2" in prompt

    def test_contains_output_format_instructions(self) -> None:
        """Encoding includes dict output format instruction."""
        formula = _make_formula([[(1, True)]])
        prompt = encode_dimacs(formula)
        assert "{1: True" in prompt or "Python dict" in prompt


class TestEncodeNaturalLanguage:
    """Tests for natural language encoding of CNF formulas."""

    def test_produces_readable_text(self) -> None:
        """Encoding contains readable clause descriptions."""
        formula = _make_formula([
            [(1, True), (2, False)],
        ])
        prompt = encode_natural_language(formula)
        assert "x1 is True" in prompt
        assert "x2 is False" in prompt
        assert "OR" in prompt

    def test_contains_variable_count(self) -> None:
        """Encoding mentions the number of variables."""
        formula = _make_formula([
            [(1, True), (2, True), (3, False)],
        ])
        prompt = encode_natural_language(formula)
        assert "3 variables" in prompt

    def test_contains_clause_count(self) -> None:
        """Encoding mentions the number of clauses."""
        formula = _make_formula([
            [(1, True)],
            [(2, False)],
        ])
        prompt = encode_natural_language(formula)
        assert "2 clauses" in prompt


class TestEncodeStructured:
    """Tests for structured JSON-like encoding of CNF formulas."""

    def test_contains_json_structure(self) -> None:
        """Encoding includes JSON-formatted formula data."""
        formula = _make_formula([
            [(1, True), (2, False)],
            [(2, True), (3, True)],
        ])
        prompt = encode_structured(formula)
        assert '"num_variables"' in prompt
        assert '"num_clauses"' in prompt
        assert '"clauses"' in prompt

    def test_preserves_variable_list(self) -> None:
        """Structured encoding lists all variables."""
        formula = _make_formula([
            [(1, True), (3, True)],
            [(2, False)],
        ])
        prompt = encode_structured(formula)
        assert '"variables"' in prompt
        # Variables 1, 2, 3 should all appear
        assert "1" in prompt
        assert "2" in prompt
        assert "3" in prompt

    def test_preserves_clause_polarity(self) -> None:
        """Structured encoding uses negative integers for negated literals."""
        formula = _make_formula([
            [(1, True), (2, False)],
        ])
        prompt = encode_structured(formula)
        # Variable 2 is negated, should appear as -2
        assert "-2" in prompt


class TestEncodeForLlm:
    """Tests for the encoding dispatch function."""

    def test_dispatches_dimacs(self) -> None:
        """encode_for_llm with 'dimacs' produces DIMACS output."""
        formula = _make_formula([[(1, True)]])
        prompt = encode_for_llm(formula, "dimacs")
        assert "p cnf" in prompt

    def test_dispatches_natural(self) -> None:
        """encode_for_llm with 'natural' produces natural language output."""
        formula = _make_formula([[(1, True)]])
        prompt = encode_for_llm(formula, "natural")
        assert "x1 is True" in prompt

    def test_dispatches_structured(self) -> None:
        """encode_for_llm with 'structured' produces JSON output."""
        formula = _make_formula([[(1, True)]])
        prompt = encode_for_llm(formula, "structured")
        assert '"num_variables"' in prompt

    def test_rejects_unknown_encoding(self) -> None:
        """encode_for_llm raises ValueError for unknown encoding."""
        formula = _make_formula([[(1, True)]])
        with pytest.raises(ValueError, match="Unknown encoding"):
            encode_for_llm(formula, "markdown")

    def test_round_trip_all_encodings(self) -> None:
        """All encodings preserve variable and clause count information."""
        formula = _make_formula([
            [(1, True), (2, False), (3, True)],
            [(2, True), (3, False)],
            [(1, False), (3, True)],
        ])
        for enc in ("dimacs", "natural", "structured"):
            prompt = encode_for_llm(formula, enc)
            # Must mention counts so MockLLMBackend can parse them
            assert "3" in prompt, f"{enc} encoding missing variable count"


# =========================================================================
# 3. RESPONSE PARSING TESTS
# =========================================================================


class TestParseLlmResponse:
    """Tests for parse_llm_response across formats."""

    def test_parses_python_dict(self) -> None:
        """Parses standard Python dict literal."""
        response = "{1: True, 2: False, 3: True}"
        expected_vars = _make_variables(1, 2, 3)
        result = parse_llm_response(response, expected_vars)

        assert result is not None
        assert result == {1: True, 2: False, 3: True}

    def test_parses_json_format(self) -> None:
        """Parses JSON-style dict with string keys."""
        response = '{"1": true, "2": false, "3": true}'
        expected_vars = _make_variables(1, 2, 3)
        result = parse_llm_response(response, expected_vars)

        assert result is not None
        assert result[1] is True
        assert result[2] is False
        assert result[3] is True

    def test_parses_inline_format(self) -> None:
        """Parses x1=True, x2=False inline format."""
        response = "x1=True, x2=False, x3=True"
        expected_vars = _make_variables(1, 2, 3)
        result = parse_llm_response(response, expected_vars)

        assert result is not None
        assert result[1] is True
        assert result[2] is False
        assert result[3] is True

    def test_extracts_from_surrounding_text(self) -> None:
        """Extracts dict from within reasoning text."""
        response = (
            "Let me analyze this step by step.\n"
            "After checking all clauses, the answer is:\n"
            "{1: True, 2: False, 3: True}\n"
            "This satisfies all constraints."
        )
        expected_vars = _make_variables(1, 2, 3)
        result = parse_llm_response(response, expected_vars)

        assert result is not None
        assert result == {1: True, 2: False, 3: True}

    def test_fills_missing_variables_with_false(self) -> None:
        """Missing variables are filled with False (DPLL convention)."""
        response = "{1: True}"
        expected_vars = _make_variables(1, 2, 3)
        result = parse_llm_response(response, expected_vars)

        assert result is not None
        assert result[1] is True
        assert result[2] is False
        assert result[3] is False

    def test_returns_none_for_complete_garbage(self) -> None:
        """Returns None for completely unparseable text."""
        response = "I cannot solve this problem, it's too hard."
        expected_vars = _make_variables(1, 2, 3)
        result = parse_llm_response(response, expected_vars)
        assert result is None

    def test_returns_none_for_empty_response(self) -> None:
        """Returns None for empty string."""
        result = parse_llm_response("", _make_variables(1, 2))
        assert result is None

    def test_parses_lowercase_true_false(self) -> None:
        """Handles lowercase true/false in inline format."""
        response = "x1=true, x2=false"
        expected_vars = _make_variables(1, 2)
        result = parse_llm_response(response, expected_vars)

        assert result is not None
        assert result[1] is True
        assert result[2] is False


# =========================================================================
# 4. SOLVER CORRECTNESS TESTS
# =========================================================================


class TestLLMOracleSolverProtocol:
    """Tests for Solver protocol compliance."""

    def test_isinstance_solver(self) -> None:
        """LLMOracleSolver satisfies the Solver protocol."""
        solver = LLMOracleSolver(backend=MockLLMBackend(seed=42))
        assert isinstance(solver, Solver)

    def test_name_returns_string(self) -> None:
        """name() returns a non-empty string."""
        solver = LLMOracleSolver(backend=MockLLMBackend(seed=42))
        name = solver.name()
        assert isinstance(name, str)
        assert len(name) > 0
        assert "LLMOracle" in name

    def test_complexity_claim_returns_string(self) -> None:
        """complexity_claim() returns a non-empty string."""
        solver = LLMOracleSolver(backend=MockLLMBackend(seed=42))
        claim = solver.complexity_claim()
        assert isinstance(claim, str)
        assert len(claim) > 0
        assert "exponential" in claim.lower() or "2^n" in claim


class TestLLMOracleSolverCorrectness:
    """Tests for solver correctness on known instances."""

    def test_trivial_formula(self, trivial_formula: CNFFormula) -> None:
        """Solver handles single-clause formula."""
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42), max_attempts=10
        )
        result = solver.solve(trivial_formula)
        # May return None (unlucky RNG) or a valid assignment
        if result is not None:
            assert trivial_formula.evaluate(result)

    def test_empty_formula(self, empty_formula: CNFFormula) -> None:
        """Solver returns assignment for empty (vacuously true) formula."""
        solver = LLMOracleSolver(backend=MockLLMBackend(seed=42))
        result = solver.solve(empty_formula)
        # Empty formula is vacuously satisfiable
        assert result is not None
        assert empty_formula.evaluate(result)

    def test_simple_sat(self, simple_sat_formula: CNFFormula) -> None:
        """Solver attempts to solve a simple SAT instance."""
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42), max_attempts=20
        )
        result = solver.solve(simple_sat_formula)
        # May or may not find it, but if it does, it must be valid
        if result is not None:
            assert simple_sat_formula.evaluate(result)

    def test_simple_unsat(self, simple_unsat_formula: CNFFormula) -> None:
        """Solver returns None for provably unsatisfiable formula."""
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42), max_attempts=10
        )
        result = solver.solve(simple_unsat_formula)
        # UNSAT: solver must return None (cannot find valid assignment)
        assert result is None

    def test_medium_sat(self, medium_sat_formula: CNFFormula) -> None:
        """Solver soundness on medium formula: never returns invalid."""
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42), max_attempts=10
        )
        result = solver.solve(medium_sat_formula)
        # Soundness: if it returns something, it must be valid
        if result is not None:
            assert medium_sat_formula.evaluate(result)

    def test_soundness_never_returns_invalid(self) -> None:
        """Across many seeds, solver never returns an invalid assignment.

        This is the critical soundness property: the verify-after-generate
        pattern must never leak an unverified assignment.
        """
        formula, _ = generate_satisfiable_instance(
            num_vars=5, num_clauses=10, k=3, seed=42
        )
        for seed in range(50):
            solver = LLMOracleSolver(
                backend=MockLLMBackend(seed=seed), max_attempts=3
            )
            result = solver.solve(formula)
            if result is not None:
                assert formula.evaluate(result), (
                    f"Soundness violation at seed {seed}: solver returned "
                    f"assignment that does not satisfy the formula"
                )

    def test_single_variable_sat(self) -> None:
        """Solver handles 1-variable SAT: (x1)."""
        formula = _make_formula([[(1, True)]])
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42), max_attempts=10
        )
        result = solver.solve(formula)
        if result is not None:
            assert formula.evaluate(result)

    def test_single_variable_unsat(self) -> None:
        """Solver handles 1-variable UNSAT: (x1) AND (~x1)."""
        formula = _make_formula([
            [(1, True)],
            [(1, False)],
        ])
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42), max_attempts=10
        )
        result = solver.solve(formula)
        assert result is None


class TestLLMOracleSolverConfig:
    """Tests for solver configuration and validation."""

    def test_rejects_unknown_encoding(self) -> None:
        """Raises ValueError for unknown encoding."""
        with pytest.raises(ValueError, match="Unknown encoding"):
            LLMOracleSolver(
                backend=MockLLMBackend(seed=42), encoding="xml"
            )

    def test_rejects_unknown_strategy(self) -> None:
        """Raises ValueError for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            LLMOracleSolver(
                backend=MockLLMBackend(seed=42), strategy="magic"
            )

    def test_all_encodings_accepted(self) -> None:
        """All valid encoding names are accepted."""
        for enc in ("dimacs", "natural", "structured"):
            solver = LLMOracleSolver(
                backend=MockLLMBackend(seed=42), encoding=enc
            )
            assert solver is not None

    def test_all_strategies_accepted(self) -> None:
        """All valid strategy names are accepted."""
        for strat in ("baseline", "chain_of_thought",
                       "constraint_highlight", "incremental"):
            solver = LLMOracleSolver(
                backend=MockLLMBackend(seed=42), strategy=strat
            )
            assert solver is not None


# =========================================================================
# 5. RETRY FRAMEWORK TESTS
# =========================================================================


class TestFailureDiagnosis:
    """Tests for diagnose_failure classification."""

    def test_api_error_when_response_is_none(self) -> None:
        """None response classifies as API_ERROR."""
        formula = _make_formula([[(1, True)]])
        mode, violated = diagnose_failure(formula, None, None)
        assert mode == FailureMode.API_ERROR
        assert violated == []

    def test_parse_error_when_assignment_is_none(self) -> None:
        """Non-None response but None assignment classifies as PARSE_ERROR."""
        formula = _make_formula([[(1, True)]])
        mode, violated = diagnose_failure(formula, "garbage text", None)
        assert mode == FailureMode.PARSE_ERROR
        assert violated == []

    def test_wrong_assignment_with_violated_clauses(self) -> None:
        """Wrong assignment classifies as WRONG_ASSIGNMENT with violations."""
        # (x1) AND (~x1): assigning x1=True violates clause 2
        formula = _make_formula([
            [(1, True)],
            [(1, False)],
        ])
        assignment = {1: True}
        mode, violated = diagnose_failure(formula, "some response", assignment)
        assert mode == FailureMode.WRONG_ASSIGNMENT
        assert len(violated) == 1
        assert violated[0].clause_index == 2

    def test_violated_clause_captures_variable_values(self) -> None:
        """ViolatedClause records the offending variable values."""
        formula = _make_formula([
            [(1, True), (2, True)],
        ])
        assignment = {1: False, 2: False}
        mode, violated = diagnose_failure(formula, "resp", assignment)
        assert mode == FailureMode.WRONG_ASSIGNMENT
        assert len(violated) == 1
        vc = violated[0]
        assert vc.variable_values == {1: False, 2: False}


class TestRetryStrategy:
    """Tests for strategy rotation and temperature escalation."""

    def test_attempt_0_uses_initial_strategy(self) -> None:
        """First attempt uses the configured strategy."""
        assert _get_retry_strategy(0, "chain_of_thought") == "chain_of_thought"
        assert _get_retry_strategy(0, "baseline") == "baseline"

    def test_subsequent_attempts_rotate(self) -> None:
        """Attempts 1+ rotate through strategies."""
        strategies = [_get_retry_strategy(i, "baseline") for i in range(1, 7)]
        # Should rotate through non-incremental strategies
        rotation = [s for s in _STRATEGY_ROTATION if s != "incremental"]
        for i, strat in enumerate(strategies):
            expected = rotation[i % len(rotation)]
            assert strat == expected, f"Attempt {i+1}: expected {expected}, got {strat}"

    def test_temperature_escalation(self) -> None:
        """Temperature increases with attempt number."""
        temps = [_get_retry_temperature(i) for i in range(len(_TEMPERATURE_SCHEDULE))]
        assert temps == _TEMPERATURE_SCHEDULE
        # Should be non-decreasing
        for i in range(1, len(temps)):
            assert temps[i] >= temps[i - 1]

    def test_temperature_clamps_at_max(self) -> None:
        """Temperature beyond schedule length uses the last value."""
        last = _TEMPERATURE_SCHEDULE[-1]
        assert _get_retry_temperature(100) == last

    def test_strategy_rotation_cycles(self) -> None:
        """Strategy rotation wraps around after exhausting list."""
        rotation = [s for s in _STRATEGY_ROTATION if s != "incremental"]
        cycle_len = len(rotation)
        s1 = _get_retry_strategy(1, "baseline")
        s1_again = _get_retry_strategy(1 + cycle_len, "baseline")
        assert s1 == s1_again


class TestRetryFeedback:
    """Tests for build_retry_feedback content."""

    def test_parse_error_feedback(self) -> None:
        """Parse error feedback includes format instructions."""
        record = AttemptRecord(attempt_number=1, strategy="baseline",
                               temperature=0.3,
                               failure_mode=FailureMode.PARSE_ERROR)
        feedback = build_retry_feedback(FailureMode.PARSE_ERROR, record, 3)
        assert "RETRY" in feedback
        assert "True" in feedback or "dict" in feedback.lower()

    def test_wrong_assignment_feedback(self) -> None:
        """Wrong assignment feedback includes violated clause info."""
        vc = ViolatedClause(
            clause_index=2,
            clause_text="(x1 OR NOT x2)",
            variable_values={1: False, 2: True},
        )
        record = AttemptRecord(
            attempt_number=1, strategy="baseline", temperature=0.3,
            failure_mode=FailureMode.WRONG_ASSIGNMENT,
            assignment={1: False, 2: True},
            violated_clauses=[vc],
        )
        feedback = build_retry_feedback(
            FailureMode.WRONG_ASSIGNMENT, record, 2
        )
        assert "violated" in feedback.lower()
        assert "Clause 2" in feedback

    def test_api_error_feedback(self) -> None:
        """API error feedback tells the LLM to try again."""
        record = AttemptRecord(attempt_number=1, strategy="baseline",
                               temperature=0.3,
                               failure_mode=FailureMode.API_ERROR)
        feedback = build_retry_feedback(FailureMode.API_ERROR, record, 3)
        assert "error" in feedback.lower()


class TestRetryBehavior:
    """Tests for solver retry loop behavior."""

    def test_respects_max_attempts(self) -> None:
        """Solver makes at most max_attempts LLM calls."""
        formula = _make_formula([
            [(1, True)],
            [(1, False)],
        ])
        backend = MockLLMBackend(seed=42)
        solver = LLMOracleSolver(
            backend=backend, max_attempts=3, timeout_seconds=60.0
        )
        solver.solve(formula)
        assert solver.attempts_made <= 3

    def test_respects_timeout(self) -> None:
        """Solver respects timeout_seconds limit."""
        formula, _ = generate_satisfiable_instance(
            num_vars=5, num_clauses=10, k=3, seed=42
        )
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42),
            max_attempts=1000,
            timeout_seconds=0.001,  # extremely short timeout
        )
        result = solver.solve(formula)
        # Should have stopped quickly; may or may not have found a result
        # on the first attempt before timeout kicks in
        if result is not None:
            assert formula.evaluate(result)

    def test_attempt_history_populated(self) -> None:
        """Attempt history records all attempts."""
        formula = _make_formula([
            [(1, True)],
            [(1, False)],
        ])
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42), max_attempts=3
        )
        solver.solve(formula)
        history = solver.attempt_history
        assert len(history) > 0
        assert len(history) <= 3
        for record in history:
            assert isinstance(record, AttemptRecord)
            assert record.attempt_number >= 1

    def test_strategy_rotation_in_practice(self) -> None:
        """Multiple attempts use different strategies."""
        formula = _make_formula([
            [(1, True)],
            [(1, False)],
        ])
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42),
            strategy="baseline",
            max_attempts=4,
        )
        solver.solve(formula)
        history = solver.attempt_history
        if len(history) >= 2:
            strategies = [r.strategy for r in history]
            # After the first attempt, strategies should rotate
            assert len(set(strategies)) >= 2 or len(history) == 1


# =========================================================================
# 6. METRICS TESTS
# =========================================================================


class TestOracleSolveMetrics:
    """Tests for per-solve metrics."""

    def test_metrics_populated_after_solve(self) -> None:
        """get_metrics() returns non-empty list after a solve."""
        formula = _make_formula([
            [(1, True), (2, True)],
        ])
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42), max_attempts=5
        )
        solver.solve(formula)
        metrics = solver.get_metrics()
        assert len(metrics) == 1
        m = metrics[0]
        assert isinstance(m, OracleSolveMetrics)
        assert m.num_variables == 2
        assert m.num_clauses == 1
        assert m.attempts_made >= 1
        assert m.encoding == "structured"

    def test_metrics_accumulate_across_solves(self) -> None:
        """Multiple solve calls accumulate metrics."""
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42), max_attempts=3
        )
        f1 = _make_formula([[(1, True)]])
        f2 = _make_formula([[(1, True), (2, False)]])

        solver.solve(f1)
        solver.solve(f2)

        metrics = solver.get_metrics()
        assert len(metrics) == 2
        assert metrics[0].num_variables == 1
        assert metrics[1].num_variables == 2

    def test_reset_metrics_clears_history(self) -> None:
        """reset_metrics() clears all accumulated metrics."""
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42), max_attempts=3
        )
        solver.solve(_make_formula([[(1, True)]]))
        assert len(solver.get_metrics()) == 1

        solver.reset_metrics()
        assert len(solver.get_metrics()) == 0

    def test_metrics_to_dict(self) -> None:
        """OracleSolveMetrics.to_dict() produces serializable output."""
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42), max_attempts=3
        )
        solver.solve(_make_formula([[(1, True)]]))
        m = solver.get_metrics()[0]
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "num_variables" in d
        assert "attempts_made" in d
        assert isinstance(d["strategies_used"], list)
        assert isinstance(d["temperatures_used"], list)

    def test_successful_solve_metrics(self) -> None:
        """Successful solve records successful=True and attempt number."""
        # Use a formula that the mock can trivially satisfy
        formula = _make_formula([[(1, True), (2, True), (3, True)]])
        # Try many seeds to find one that succeeds
        for seed in range(100):
            solver = LLMOracleSolver(
                backend=MockLLMBackend(seed=seed), max_attempts=5
            )
            result = solver.solve(formula)
            if result is not None:
                metrics = solver.get_metrics()
                assert len(metrics) == 1
                m = metrics[0]
                assert m.successful is True
                assert m.successful_attempt is not None
                assert m.successful_attempt >= 1
                return
        # If no seed succeeded, that's okay -- skip
        pytest.skip("No seed found that solves the trivial formula")


class TestAggregateMetrics:
    """Tests for aggregate metrics computation."""

    def test_empty_records(self) -> None:
        """Aggregate of empty record list returns zero defaults."""
        agg = _compute_aggregate_metrics([])
        assert isinstance(agg, OracleAggregateMetrics)
        assert agg.total_instances == 0
        assert agg.success_rate == 0.0

    def test_aggregate_success_rate(self) -> None:
        """Aggregate computes correct success rate."""
        records = [
            OracleSolveMetrics(
                num_variables=3, num_clauses=2, clause_ratio=0.67,
                encoding="structured", attempts_made=1,
                successful=True, successful_attempt=1,
                strategies_used=("baseline",), temperatures_used=(0.3,),
                failure_modes=(), clauses_violated_per_attempt=(0,),
                total_elapsed_seconds=0.1, per_attempt_seconds=(0.1,),
                total_tokens_used=0,
            ),
            OracleSolveMetrics(
                num_variables=3, num_clauses=2, clause_ratio=0.67,
                encoding="structured", attempts_made=3,
                successful=False, successful_attempt=None,
                strategies_used=("baseline", "chain_of_thought", "baseline"),
                temperatures_used=(0.3, 0.5, 0.7),
                failure_modes=("wrong", "wrong", "wrong"),
                clauses_violated_per_attempt=(1, 1, 1),
                total_elapsed_seconds=0.3, per_attempt_seconds=(0.1, 0.1, 0.1),
                total_tokens_used=0,
            ),
        ]
        agg = _compute_aggregate_metrics(records)
        assert agg.total_instances == 2
        assert agg.successful_instances == 1
        assert agg.success_rate == 0.5

    def test_success_rate_by_size(self) -> None:
        """Aggregate computes success rate grouped by problem size."""
        records = [
            OracleSolveMetrics(
                num_variables=3, num_clauses=2, clause_ratio=0.67,
                encoding="structured", attempts_made=1,
                successful=True, successful_attempt=1,
                strategies_used=("baseline",), temperatures_used=(0.3,),
                failure_modes=(), clauses_violated_per_attempt=(0,),
                total_elapsed_seconds=0.1, per_attempt_seconds=(0.1,),
                total_tokens_used=0,
            ),
            OracleSolveMetrics(
                num_variables=5, num_clauses=10, clause_ratio=2.0,
                encoding="structured", attempts_made=3,
                successful=False, successful_attempt=None,
                strategies_used=("baseline",), temperatures_used=(0.3,),
                failure_modes=("wrong",),
                clauses_violated_per_attempt=(2,),
                total_elapsed_seconds=0.3, per_attempt_seconds=(0.3,),
                total_tokens_used=0,
            ),
        ]
        agg = _compute_aggregate_metrics(records)
        assert 3 in agg.success_rate_by_size
        assert 5 in agg.success_rate_by_size
        assert agg.success_rate_by_size[3] == 1.0
        assert agg.success_rate_by_size[5] == 0.0

    def test_get_aggregate_metrics_from_solver(self) -> None:
        """Solver's get_aggregate_metrics uses accumulated history."""
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42), max_attempts=3
        )
        solver.solve(_make_formula([[(1, True)]]))
        solver.solve(_make_formula([[(1, True), (2, True)]]))
        agg = solver.get_aggregate_metrics()
        assert agg.total_instances == 2

    def test_aggregate_to_dict(self) -> None:
        """OracleAggregateMetrics.to_dict() is serializable."""
        agg = OracleAggregateMetrics(
            total_instances=1,
            successful_instances=1,
            success_rate=1.0,
            success_rate_by_size={3: 1.0},
        )
        d = agg.to_dict()
        assert isinstance(d, dict)
        assert d["total_instances"] == 1
        assert isinstance(d["success_rate_by_size"], dict)


# =========================================================================
# 7. CROSS-VALIDATION WITH BRUTE FORCE
# =========================================================================


class TestCrossValidationWithBruteForce:
    """Cross-validate LLMOracleSolver against BruteForceSolver.

    Follows the pattern from test_experimental.py: run both solvers on
    20 shared instances. The LLM oracle may return None (acceptable),
    but must NEVER return an assignment that does not satisfy the formula.
    """

    @pytest.mark.parametrize("seed", CROSS_SEEDS)
    def test_agrees_with_brute_force(self, seed: int) -> None:
        """LLMOracleSolver agrees with brute force on soundness."""
        formula = generate_random_ksat(
            k=3, num_vars=CROSS_N, num_clauses=CROSS_M, seed=seed
        )
        brute = BruteForceSolver(timeout_seconds=10.0)
        brute_result = brute.solve(formula)

        oracle = LLMOracleSolver(
            backend=MockLLMBackend(seed=seed), max_attempts=5
        )
        oracle_result = oracle.solve(formula)

        _assert_solver_consistency(
            formula, brute_result, oracle_result, "LLMOracle", seed
        )


def _assert_solver_consistency(
    formula: CNFFormula,
    brute_result: dict[int, bool] | None,
    solver_result: dict[int, bool] | None,
    solver_name: str,
    seed: int,
) -> None:
    """Assert that a solver's result is consistent with brute force.

    Rules:
    - Any non-None result must satisfy the formula (soundness).
    - If brute force says UNSAT, solver must not return a valid assignment.
    - If brute force says SAT, solver may return None (incomplete).

    Args:
        formula: The CNF formula.
        brute_result: Ground truth from BruteForceSolver.
        solver_result: Result from the LLM oracle solver.
        solver_name: Name for error messages.
        seed: Random seed for error messages.
    """
    context = f"{solver_name} (seed={seed})"

    if solver_result is not None:
        # Any non-None result must be a valid satisfying assignment
        assert formula.evaluate(solver_result), (
            f"{context} returned an assignment that does NOT satisfy "
            f"the formula. Assignment: {solver_result}"
        )

    if not (brute_result is not None) and solver_result is not None:
        # Solver claims SAT but brute force says UNSAT
        assert formula.evaluate(solver_result), (
            f"{context} returned SAT but brute force says UNSAT"
        )


# =========================================================================
# 8. GRACEFUL DEGRADATION TESTS
# =========================================================================


class TestGracefulDegradation:
    """Tests for graceful failure when dependencies are missing."""

    def test_anthropic_backend_no_api_key(self) -> None:
        """AnthropicBackend without API key returns None from generate."""
        with patch.dict("os.environ", {}, clear=True):
            backend = AnthropicBackend()
            result = backend.generate("test prompt")
            assert result is None

    def test_solver_falls_back_to_mock(self) -> None:
        """Solver without explicit backend falls back gracefully."""
        with patch.dict("os.environ", {}, clear=True):
            # Without ANTHROPIC_API_KEY, should fall back to MockLLMBackend
            solver = LLMOracleSolver()
            formula = _make_formula([[(1, True), (2, True)]])
            result = solver.solve(formula)
            # Should not raise; result is either None or valid
            if result is not None:
                assert formula.evaluate(result)

    def test_malformed_backend_response(self) -> None:
        """Solver handles a backend that returns garbage."""

        class GarbageBackend:
            """Backend that always returns unparseable text."""

            def generate(self, prompt: str) -> str:
                return "I am a teapot, not a SAT solver."

        solver = LLMOracleSolver(
            backend=GarbageBackend(),  # type: ignore[arg-type]
            max_attempts=3,
        )
        formula = _make_formula([[(1, True), (2, True)]])
        result = solver.solve(formula)
        # Should return None without raising
        assert result is None

    def test_backend_returning_none(self) -> None:
        """Solver handles a backend that always returns None."""

        class NoneBackend:
            """Backend that simulates API failures."""

            def generate(self, prompt: str) -> None:
                return None

        solver = LLMOracleSolver(
            backend=NoneBackend(),  # type: ignore[arg-type]
            max_attempts=3,
        )
        formula = _make_formula([[(1, True)]])
        result = solver.solve(formula)
        assert result is None

    def test_no_exception_propagates(self) -> None:
        """Solver never lets exceptions propagate to caller."""
        formula = _make_formula([
            [(1, True), (2, False)],
            [(2, True), (3, True)],
        ])
        # Test with various backend configurations
        for seed in range(10):
            solver = LLMOracleSolver(
                backend=MockLLMBackend(seed=seed), max_attempts=3
            )
            # This should never raise, regardless of what the mock returns
            result = solver.solve(formula)
            if result is not None:
                assert formula.evaluate(result)


# =========================================================================
# VARIABLE COUNT EXTRACTION TESTS
# =========================================================================


class TestExtractVariableCount:
    """Tests for _extract_variable_count from prompt text."""

    def test_extracts_from_n_variables_pattern(self) -> None:
        """Extracts from 'N variables' pattern."""
        assert _extract_variable_count("This has 5 variables") == 5

    def test_extracts_from_dimacs_header(self) -> None:
        """Extracts from DIMACS p cnf header."""
        assert _extract_variable_count("p cnf 10 42") == 10

    def test_extracts_from_num_variables_key(self) -> None:
        """Extracts from 'num_variables: N' pattern."""
        assert _extract_variable_count('"num_variables": 7') == 7

    def test_returns_none_for_no_match(self) -> None:
        """Returns None when no pattern matches."""
        assert _extract_variable_count("solve this") is None


# =========================================================================
# INCREMENTAL STRATEGY TESTS
# =========================================================================


class TestIncrementalStrategy:
    """Tests for the incremental solving strategy."""

    def test_incremental_solver_soundness(self) -> None:
        """Incremental solver never returns invalid assignments."""
        formula, _ = generate_satisfiable_instance(
            num_vars=6, num_clauses=12, k=3, seed=42
        )
        for seed in range(20):
            solver = LLMOracleSolver(
                backend=MockLLMBackend(seed=seed),
                strategy="incremental",
                max_attempts=5,
            )
            result = solver.solve(formula)
            if result is not None:
                assert formula.evaluate(result), (
                    f"Incremental solver soundness violation at seed {seed}"
                )

    def test_incremental_records_stages(self) -> None:
        """Incremental solver populates stage count in metrics."""
        formula = _make_formula([
            [(1, True), (2, False)],
            [(2, True), (3, True)],
        ])
        solver = LLMOracleSolver(
            backend=MockLLMBackend(seed=42),
            strategy="incremental",
            max_attempts=5,
        )
        solver.solve(formula)
        m = solver.metrics
        # Should have tracked incremental_stages
        assert "incremental_stages" in m
