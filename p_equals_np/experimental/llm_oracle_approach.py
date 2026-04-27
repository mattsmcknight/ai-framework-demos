"""LLM Oracle approach: probabilistic inference via large language model.

This solver uses an LLM as a probabilistic oracle for candidate truth
assignment generation, instantiating the "correctness island" pattern
from WP5 (Deterministic Computation on a Probabilistic Substrate):
the LLM (stochastic) proposes candidate assignments, and
``CNFFormula.evaluate()`` (deterministic) verifies them.

The solver never trusts the LLM's output. Every candidate assignment
is verified in polynomial time via the deterministic evaluator. The LLM
serves only as a heuristic generator -- a "guess machine" whose outputs
are worthless until verified.

Why this fails to solve SAT in polynomial time:
    The LLM's probability of guessing a correct satisfying assignment
    decreases exponentially with the number of variables. For n Boolean
    variables, the search space is 2^n. Even if the LLM concentrates
    probability mass on "plausible" assignments (exploiting learned
    structural patterns from training data), the number of LLM calls
    required to find a satisfying assignment grows exponentially with n
    for hard random 3-SAT instances at the phase transition.

    This maps the P/NP boundary into the probabilistic domain: verification
    is cheap (polynomial), but generation of correct candidates is hard
    (exponential in expectation). The solver hits an *information-theoretic
    wall* rather than a *computational wall* -- a qualitatively different
    failure mode from the other six solvers.

Complexity claim:
    O(2^n / p(n)) expected LLM calls, where p(n) is the LLM's probability
    of producing a satisfying assignment for an n-variable formula. Each
    call has O(n * m) verification cost. For hard instances, p(n) decays
    exponentially, yielding exponential expected total cost.

Example:
    >>> from p_equals_np.sat_types import Variable, Literal, Clause, CNFFormula
    >>> x1, x2 = Variable(1), Variable(2)
    >>> f = CNFFormula((
    ...     Clause((Literal(x1), Literal(x2))),
    ...     Clause((Literal(x1, False), Literal(x2))),
    ... ))
    >>> solver = LLMOracleSolver(backend=MockLLMBackend(seed=42))
    >>> result = solver.solve(f)
    >>> result is None or f.evaluate(result)
    True
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol, runtime_checkable

from p_equals_np.sat_types import Clause, CNFFormula, Literal, Variable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OracleSolveMetrics:
    """Metrics from a single LLMOracleSolver.solve() invocation.

    Captures per-solve performance data for empirical analysis of how
    the LLM oracle's success probability scales with problem size.
    All tuples are ordered by attempt number (0-indexed).

    Attributes:
        num_variables: Number of Boolean variables in the formula.
        num_clauses: Number of clauses in the formula.
        clause_ratio: Clause-to-variable ratio (hardness indicator).
        encoding: The prompt encoding format used.
        attempts_made: Total number of LLM calls made.
        successful: Whether a satisfying assignment was found.
        successful_attempt: Which attempt succeeded (1-indexed), or None.
        strategies_used: Prompt strategy name per attempt.
        temperatures_used: Sampling temperature per attempt.
        failure_modes: FailureMode string per failed attempt.
        clauses_violated_per_attempt: Count of violated clauses per attempt.
        total_elapsed_seconds: Wall-clock time for the entire solve.
        per_attempt_seconds: Wall-clock time for each attempt.
        total_tokens_used: Approximate token count (from backend, if available).
    """

    num_variables: int
    num_clauses: int
    clause_ratio: float
    encoding: str
    attempts_made: int
    successful: bool
    successful_attempt: Optional[int]
    strategies_used: tuple[str, ...]
    temperatures_used: tuple[float, ...]
    failure_modes: tuple[str, ...]
    clauses_violated_per_attempt: tuple[int, ...]
    total_elapsed_seconds: float
    per_attempt_seconds: tuple[float, ...]
    total_tokens_used: int

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dict with all metrics fields, tuples converted to lists.
        """
        return {
            "num_variables": self.num_variables,
            "num_clauses": self.num_clauses,
            "clause_ratio": self.clause_ratio,
            "encoding": self.encoding,
            "attempts_made": self.attempts_made,
            "successful": self.successful,
            "successful_attempt": self.successful_attempt,
            "strategies_used": list(self.strategies_used),
            "temperatures_used": list(self.temperatures_used),
            "failure_modes": list(self.failure_modes),
            "clauses_violated_per_attempt": list(self.clauses_violated_per_attempt),
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "per_attempt_seconds": list(self.per_attempt_seconds),
            "total_tokens_used": self.total_tokens_used,
        }


@dataclass
class OracleAggregateMetrics:
    """Aggregate metrics across multiple solve calls.

    Computed on demand from accumulated OracleSolveMetrics records.
    Provides the key scaling analysis data: success rate by problem
    size, by strategy, and by encoding.

    Attributes:
        total_instances: Number of solve calls recorded.
        successful_instances: Number that found a satisfying assignment.
        success_rate: Fraction of successful solves.
        mean_attempts_to_success: Mean attempt count among successes.
        mean_clauses_violated_first_attempt: Mean violated clauses on
            the first attempt across all solves.
        success_rate_by_size: Success rate keyed by num_variables.
        attempts_distribution: Count of successes keyed by attempt number.
        strategy_success_rates: Success rate keyed by initial strategy.
        encoding_success_rates: Success rate keyed by encoding format.
        mean_elapsed_by_size: Mean wall-clock seconds keyed by num_variables.
    """

    total_instances: int = 0
    successful_instances: int = 0
    success_rate: float = 0.0
    mean_attempts_to_success: float = 0.0
    mean_clauses_violated_first_attempt: float = 0.0
    success_rate_by_size: dict[int, float] = field(default_factory=dict)
    attempts_distribution: dict[int, int] = field(default_factory=dict)
    strategy_success_rates: dict[str, float] = field(default_factory=dict)
    encoding_success_rates: dict[str, float] = field(default_factory=dict)
    mean_elapsed_by_size: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dict with all aggregate metrics fields.
        """
        return {
            "total_instances": self.total_instances,
            "successful_instances": self.successful_instances,
            "success_rate": self.success_rate,
            "mean_attempts_to_success": self.mean_attempts_to_success,
            "mean_clauses_violated_first_attempt": self.mean_clauses_violated_first_attempt,
            "success_rate_by_size": {
                str(k): v for k, v in self.success_rate_by_size.items()
            },
            "attempts_distribution": {
                str(k): v for k, v in self.attempts_distribution.items()
            },
            "strategy_success_rates": self.strategy_success_rates,
            "encoding_success_rates": self.encoding_success_rates,
            "mean_elapsed_by_size": {
                str(k): v for k, v in self.mean_elapsed_by_size.items()
            },
        }


def _compute_aggregate_metrics(
    records: list[OracleSolveMetrics],
) -> OracleAggregateMetrics:
    """Compute aggregate metrics from a list of per-solve records.

    Args:
        records: List of OracleSolveMetrics from individual solve calls.

    Returns:
        Computed OracleAggregateMetrics summarizing all records.
    """
    if not records:
        return OracleAggregateMetrics()

    total = len(records)
    successes = [r for r in records if r.successful]
    num_success = len(successes)

    # Success rate
    success_rate = num_success / total if total > 0 else 0.0

    # Mean attempts to success (among successful solves)
    mean_attempts = 0.0
    if successes:
        mean_attempts = sum(
            r.successful_attempt for r in successes
            if r.successful_attempt is not None
        ) / num_success

    # Mean clauses violated on first attempt
    first_attempt_violations: list[int] = []
    for r in records:
        if r.clauses_violated_per_attempt:
            first_attempt_violations.append(r.clauses_violated_per_attempt[0])
    mean_first_violated = (
        sum(first_attempt_violations) / len(first_attempt_violations)
        if first_attempt_violations
        else 0.0
    )

    # Success rate by problem size (num_variables)
    size_total: dict[int, int] = defaultdict(int)
    size_success: dict[int, int] = defaultdict(int)
    size_elapsed: dict[int, list[float]] = defaultdict(list)
    for r in records:
        size_total[r.num_variables] += 1
        size_elapsed[r.num_variables].append(r.total_elapsed_seconds)
        if r.successful:
            size_success[r.num_variables] += 1
    success_rate_by_size = {
        n: size_success.get(n, 0) / size_total[n]
        for n in sorted(size_total)
    }
    mean_elapsed_by_size = {
        n: sum(times) / len(times)
        for n, times in sorted(size_elapsed.items())
    }

    # Attempts distribution: which attempt number yielded success
    attempts_dist: dict[int, int] = defaultdict(int)
    for r in successes:
        if r.successful_attempt is not None:
            attempts_dist[r.successful_attempt] += 1

    # Strategy success rates (by first strategy used in each solve)
    strat_total: dict[str, int] = defaultdict(int)
    strat_success: dict[str, int] = defaultdict(int)
    for r in records:
        if r.strategies_used:
            s = r.strategies_used[0]
            strat_total[s] += 1
            if r.successful:
                strat_success[s] += 1
    strategy_rates = {
        s: strat_success.get(s, 0) / strat_total[s]
        for s in sorted(strat_total)
    }

    # Encoding success rates
    enc_total: dict[str, int] = defaultdict(int)
    enc_success: dict[str, int] = defaultdict(int)
    for r in records:
        enc_total[r.encoding] += 1
        if r.successful:
            enc_success[r.encoding] += 1
    encoding_rates = {
        e: enc_success.get(e, 0) / enc_total[e]
        for e in sorted(enc_total)
    }

    return OracleAggregateMetrics(
        total_instances=total,
        successful_instances=num_success,
        success_rate=success_rate,
        mean_attempts_to_success=mean_attempts,
        mean_clauses_violated_first_attempt=mean_first_violated,
        success_rate_by_size=success_rate_by_size,
        attempts_distribution=dict(attempts_dist),
        strategy_success_rates=strategy_rates,
        encoding_success_rates=encoding_rates,
        mean_elapsed_by_size=mean_elapsed_by_size,
    )


def export_metrics_json(
    records: list[OracleSolveMetrics],
    filepath: str,
    include_aggregate: bool = True,
) -> None:
    """Export metrics to a JSON file for external analysis.

    Writes per-solve records and optionally the computed aggregate
    metrics to a JSON file.

    Args:
        records: List of per-solve metrics to export.
        filepath: Output file path.
        include_aggregate: Whether to include aggregate summary.
    """
    data: dict[str, object] = {
        "per_solve": [r.to_dict() for r in records],
    }
    if include_aggregate:
        agg = _compute_aggregate_metrics(records)
        data["aggregate"] = agg.to_dict()
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Failure Diagnosis and Retry Infrastructure
# ---------------------------------------------------------------------------


class FailureMode(Enum):
    """Classification of why an LLM oracle attempt failed.

    Used by the retry framework to select appropriate prompt
    modifications for the next attempt.
    """

    PARSE_ERROR = "parse_error"
    INCOMPLETE_ASSIGNMENT = "incomplete"
    WRONG_ASSIGNMENT = "wrong"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"


@dataclass
class ViolatedClause:
    """A clause that was not satisfied by a candidate assignment.

    Captures the clause index, human-readable representation, and
    the variable values that caused the violation for diagnostic
    feedback to the LLM on retry.

    Attributes:
        clause_index: 1-based index of the clause in the formula.
        clause_text: Human-readable representation of the clause.
        variable_values: Mapping of variable index to the value
            assigned in the failed attempt, for variables in this clause.
    """

    clause_index: int
    clause_text: str
    variable_values: dict[int, bool] = field(default_factory=dict)


@dataclass
class AttemptRecord:
    """Record of a single solve attempt for retry analysis.

    Attributes:
        attempt_number: 1-based attempt number.
        strategy: The prompt strategy used.
        temperature: The sampling temperature used.
        failure_mode: Why the attempt failed, or None if it succeeded.
        assignment: The candidate assignment (if parsing succeeded).
        violated_clauses: Clauses not satisfied (if assignment was wrong).
        response_text: Raw LLM response (truncated for logging).
    """

    attempt_number: int
    strategy: str
    temperature: float
    failure_mode: Optional[FailureMode] = None
    assignment: Optional[dict[int, bool]] = None
    violated_clauses: list[ViolatedClause] = field(default_factory=list)
    response_text: Optional[str] = None
    elapsed_seconds: float = 0.0


def diagnose_failure(
    formula: CNFFormula,
    response: Optional[str],
    assignment: Optional[dict[int, bool]],
) -> tuple[FailureMode, list[ViolatedClause]]:
    """Classify the failure mode of an LLM oracle attempt.

    Determines why a candidate assignment failed and, for wrong
    assignments, identifies which clauses were violated and which
    variable values caused the violation.

    Args:
        formula: The CNF formula being solved.
        response: The raw LLM response text, or None if the API failed.
        assignment: The parsed assignment, or None if parsing failed.

    Returns:
        A tuple of (failure_mode, violated_clauses). violated_clauses
        is non-empty only for WRONG_ASSIGNMENT failures.
    """
    if response is None:
        return FailureMode.API_ERROR, []

    if assignment is None:
        return FailureMode.PARSE_ERROR, []

    # Check completeness
    expected_vars = formula.get_variables()
    expected_indices = frozenset(v.index for v in expected_vars)
    assigned_indices = set(assignment.keys()) & expected_indices
    if len(assigned_indices) < len(expected_indices):
        # Partial assignment -- still classify as wrong if it fails
        # evaluation after filling, but note the incompleteness
        pass

    # Check correctness
    violated: list[ViolatedClause] = []
    for i, clause in enumerate(formula.clauses, 1):
        if not clause.evaluate(assignment):
            clause_text = _format_clause_natural(clause)
            var_vals: dict[int, bool] = {}
            for lit in clause.literals:
                idx = lit.variable.index
                if idx in assignment:
                    var_vals[idx] = assignment[idx]
            violated.append(ViolatedClause(
                clause_index=i,
                clause_text=clause_text,
                variable_values=var_vals,
            ))

    if violated:
        return FailureMode.WRONG_ASSIGNMENT, violated

    # If we get here, the assignment actually satisfies -- shouldn't
    # happen in normal flow, but handle gracefully.
    return FailureMode.WRONG_ASSIGNMENT, []


def build_retry_feedback(
    failure_mode: FailureMode,
    attempt_record: AttemptRecord,
    num_vars: int,
) -> str:
    """Build prompt feedback text based on the previous attempt's failure.

    Generates targeted instructions for the LLM based on how the
    previous attempt failed, giving it specific information about
    what went wrong to guide the next attempt.

    Args:
        failure_mode: How the previous attempt failed.
        attempt_record: Full record of the previous attempt.
        num_vars: Number of variables in the formula.

    Returns:
        A feedback string to append to the next prompt.
    """
    sections: list[str] = [
        f"\n\n--- RETRY (attempt {attempt_record.attempt_number + 1}) ---\n"
    ]

    if failure_mode == FailureMode.PARSE_ERROR:
        sections.append(
            "Your previous response could not be parsed into a truth assignment.\n"
            "You MUST provide your answer as a Python dict literal on its own line:\n"
            f"  {{{', '.join(f'{i}: True' for i in range(1, min(4, num_vars + 1)))}, ...}}\n"
            "\n"
            "Example for a 4-variable problem:\n"
            "  {1: True, 2: False, 3: True, 4: False}\n"
            "\n"
            f"Include ALL {num_vars} variables. Use only True or False as values."
        )

    elif failure_mode == FailureMode.WRONG_ASSIGNMENT:
        if attempt_record.assignment:
            # Show the failed assignment (truncated for large instances)
            assign_str = repr(attempt_record.assignment)
            if len(assign_str) > 300:
                assign_str = assign_str[:297] + "..."
            sections.append(f"Your previous attempt: {assign_str}\n")

        if attempt_record.violated_clauses:
            sections.append(
                f"This assignment violated {len(attempt_record.violated_clauses)} "
                f"clause(s):\n"
            )
            # Show up to 10 violated clauses to avoid prompt bloat
            for vc in attempt_record.violated_clauses[:10]:
                val_strs = [
                    f"x{idx}={'True' if val else 'False'}"
                    for idx, val in sorted(vc.variable_values.items())
                ]
                sections.append(
                    f"  Clause {vc.clause_index}: {vc.clause_text} "
                    f"-- violated because {', '.join(val_strs)}"
                )
            if len(attempt_record.violated_clauses) > 10:
                remaining = len(attempt_record.violated_clauses) - 10
                sections.append(f"  ... and {remaining} more violated clauses")

            sections.append(
                "\nYou need to change at least one variable in each "
                "violated clause to fix the assignment."
            )

    elif failure_mode == FailureMode.INCOMPLETE_ASSIGNMENT:
        sections.append(
            "Your previous response was missing some variable assignments.\n"
            f"You MUST assign ALL {num_vars} variables (x1 through x{num_vars}).\n"
            "Do not skip any variable."
        )

    elif failure_mode == FailureMode.API_ERROR:
        sections.append(
            "The previous attempt encountered an error. "
            "Please try again with a complete assignment."
        )

    return "\n".join(sections)


# Strategy rotation order for retry attempts.
# Attempt 1 uses the solver's configured strategy; subsequent attempts
# rotate through these in order.
_STRATEGY_ROTATION: list[str] = [
    "baseline",
    "chain_of_thought",
    "constraint_highlight",
    "incremental",
]

# Temperature escalation schedule indexed by attempt number (0-based).
_TEMPERATURE_SCHEDULE: list[float] = [0.3, 0.5, 0.7, 0.9, 1.0]


def _get_retry_temperature(attempt_index: int) -> float:
    """Return the temperature for a given attempt index (0-based).

    Args:
        attempt_index: Zero-based attempt number.

    Returns:
        Sampling temperature for this attempt.
    """
    if attempt_index < len(_TEMPERATURE_SCHEDULE):
        return _TEMPERATURE_SCHEDULE[attempt_index]
    return _TEMPERATURE_SCHEDULE[-1]


def _get_retry_strategy(
    attempt_index: int, initial_strategy: str
) -> str:
    """Return the prompt strategy for a given attempt index (0-based).

    Attempt 0 uses the initial strategy. Subsequent attempts rotate
    through the strategy list, skipping 'incremental' (which has its
    own solve loop and is not compatible with the standard retry path).

    Args:
        attempt_index: Zero-based attempt number.
        initial_strategy: The solver's configured strategy.

    Returns:
        Strategy name for this attempt.
    """
    if attempt_index == 0:
        return initial_strategy
    # Rotate through non-incremental strategies
    rotation = [s for s in _STRATEGY_ROTATION if s != "incremental"]
    return rotation[(attempt_index - 1) % len(rotation)]


def _set_backend_temperature(backend: LLMBackend, temperature: float) -> None:
    """Set the temperature on a backend if it supports it.

    Silently does nothing for backends without a temperature attribute
    (e.g., MockLLMBackend).

    Args:
        backend: The LLM backend.
        temperature: The desired sampling temperature.
    """
    if hasattr(backend, "temperature"):
        backend.temperature = temperature  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# LLM Backend Protocol and Implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM API backends.

    Any object with a ``generate`` method matching this signature can
    serve as the oracle backend. This enables testing with mock backends
    and swapping LLM providers without changing solver code.
    """

    def generate(self, prompt: str) -> Optional[str]:
        """Generate a text response from a prompt.

        Args:
            prompt: The full prompt text to send to the LLM.

        Returns:
            The LLM's text response, or None if the call failed.
        """
        ...


class AnthropicBackend:
    """LLM backend using the Anthropic API (Claude models).

    Reads the API key from the ``ANTHROPIC_API_KEY`` environment variable.
    Requires the ``anthropic`` Python package (optional dependency).
    Fails gracefully if either is unavailable.

    Attributes:
        model: The Anthropic model identifier.
        temperature: Sampling temperature (higher = more random).
        max_tokens: Maximum tokens in the LLM response.
        total_input_tokens: Cumulative input tokens across all calls.
        total_output_tokens: Cumulative output tokens across all calls.
    """

    __slots__ = (
        "model",
        "temperature",
        "max_tokens",
        "total_input_tokens",
        "total_output_tokens",
        "_client",
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        """Initialize the Anthropic backend.

        Args:
            model: Anthropic model identifier.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum response tokens.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self._client: object = None

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic  # type: ignore[import-untyped]

                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                pass

    def generate(self, prompt: str) -> Optional[str]:
        """Generate a response using the Anthropic API.

        Args:
            prompt: The prompt text.

        Returns:
            The model's text response, or None if the API call failed.
        """
        if self._client is None:
            return None

        try:
            response = self._client.messages.create(  # type: ignore[union-attr]
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            return response.content[0].text  # type: ignore[union-attr]
        except Exception:
            return None


class MockLLMBackend:
    """Mock LLM backend for testing without API access.

    Generates pseudo-random truth assignments by parsing the variable
    count from the prompt and producing a random assignment. This
    simulates the LLM's stochastic behavior with a controllable seed,
    enabling deterministic testing.

    The mock intentionally has a low success rate on larger instances
    to simulate the exponential decay of correct guessing probability.

    Attributes:
        seed: Random seed for reproducibility.
        call_count: Number of ``generate`` calls made.
    """

    __slots__ = ("seed", "call_count", "_rng")

    def __init__(self, seed: int = 42) -> None:
        """Initialize the mock backend.

        Args:
            seed: Random seed for deterministic behavior.
        """
        self.seed = seed
        self.call_count: int = 0
        self._rng = random.Random(seed)

    def generate(self, prompt: str) -> Optional[str]:
        """Generate a random truth assignment based on the prompt.

        Parses the number of variables from the prompt and returns
        a random assignment formatted as a Python dict literal.

        Args:
            prompt: The prompt text (parsed for variable count).

        Returns:
            A string containing a Python dict of random assignments.
        """
        self.call_count += 1

        # Extract variable count from prompt
        num_vars = _extract_variable_count(prompt)
        if num_vars is None or num_vars < 1:
            return None

        # Generate random assignment
        assignment: dict[int, bool] = {}
        for var_idx in range(1, num_vars + 1):
            assignment[var_idx] = self._rng.choice([True, False])

        return repr(assignment)


def _extract_variable_count(prompt: str) -> Optional[int]:
    """Extract the number of variables from a SAT prompt.

    Looks for patterns like "N variables", "num_variables: N",
    or "p cnf N M" (DIMACS header).

    Args:
        prompt: The prompt text.

    Returns:
        The number of variables, or None if not found.
    """
    # Try "N variables" pattern
    match = re.search(r"(\d+)\s+variables", prompt)
    if match:
        return int(match.group(1))

    # Try DIMACS header "p cnf <vars> <clauses>"
    match = re.search(r"p\s+cnf\s+(\d+)\s+\d+", prompt)
    if match:
        return int(match.group(1))

    # Try "num_variables": N or "num_variables: N"
    match = re.search(r'"?num_variables"?\s*[:=]\s*(\d+)', prompt)
    if match:
        return int(match.group(1))

    return None


# ---------------------------------------------------------------------------
# SAT-to-Prompt Encoding
# ---------------------------------------------------------------------------


def encode_dimacs(formula: CNFFormula) -> str:
    """Encode a CNF formula as a DIMACS-format prompt for the LLM.

    Uses the formula's native DIMACS serialization with instructions
    for the LLM to produce a truth assignment.

    Args:
        formula: The CNF formula to encode.

    Returns:
        A prompt string containing the DIMACS representation.
    """
    dimacs = formula.to_dimacs()
    num_vars = formula.num_variables
    num_clauses = formula.num_clauses

    return (
        f"You are solving a Boolean Satisfiability (SAT) problem.\n"
        f"\n"
        f"The formula has {num_vars} variables and {num_clauses} clauses.\n"
        f"It is given in DIMACS CNF format below. Each line after the header\n"
        f"is a clause: positive integers are positive literals, negative\n"
        f"integers are negated literals, and 0 terminates each clause.\n"
        f"\n"
        f"{dimacs}\n"
        f"\n"
        f"Find a satisfying truth assignment. Think through your reasoning\n"
        f"step by step, then provide the assignment as a Python dict mapping\n"
        f"variable numbers (integers) to boolean values.\n"
        f"\n"
        f"Format your answer as: {{1: True, 2: False, ...}}\n"
        f"Include ALL {num_vars} variables in your assignment."
    )


def encode_natural_language(formula: CNFFormula) -> str:
    """Encode a CNF formula as natural language for the LLM.

    Translates each clause into readable English with logical
    connectives.

    Args:
        formula: The CNF formula to encode.

    Returns:
        A prompt string with the formula in natural language.
    """
    num_vars = formula.num_variables
    num_clauses = formula.num_clauses

    clause_strs: list[str] = []
    for i, clause in enumerate(formula.clauses, 1):
        lit_strs: list[str] = []
        for lit in clause.literals:
            if lit.positive:
                lit_strs.append(f"x{lit.variable.index} is True")
            else:
                lit_strs.append(f"x{lit.variable.index} is False")
        clause_strs.append(f"  Clause {i}: {' OR '.join(lit_strs)}")

    clauses_text = "\n".join(clause_strs)

    return (
        f"You are solving a Boolean Satisfiability (SAT) problem.\n"
        f"\n"
        f"There are {num_vars} variables (x1 through x{num_vars}) and\n"
        f"{num_clauses} clauses. ALL of the following clauses must be\n"
        f"satisfied simultaneously:\n"
        f"\n"
        f"{clauses_text}\n"
        f"\n"
        f"Find values for each variable (True or False) that satisfy\n"
        f"every clause. Think through your reasoning step by step,\n"
        f"then provide the assignment as a Python dict.\n"
        f"\n"
        f"Format your answer as: {{1: True, 2: False, ...}}\n"
        f"Include ALL {num_vars} variables in your assignment."
    )


def encode_structured(formula: CNFFormula) -> str:
    """Encode a CNF formula as a structured JSON-like representation.

    Provides a clear, machine-readable format with explicit variable
    list and clause list, along with instructions.

    Args:
        formula: The CNF formula to encode.

    Returns:
        A prompt string with structured formula representation.
    """
    variables = sorted(v.index for v in formula.get_variables())
    num_vars = len(variables)
    num_clauses = formula.num_clauses

    clause_list: list[list[int]] = []
    for clause in formula.clauses:
        lits: list[int] = []
        for lit in clause.literals:
            lits.append(lit.variable.index if lit.positive else -lit.variable.index)
        clause_list.append(lits)

    structured = {
        "num_variables": num_vars,
        "num_clauses": num_clauses,
        "variables": variables,
        "clauses": clause_list,
        "encoding": "positive integer = variable is True, negative = variable is False",
    }

    structured_text = json.dumps(structured, indent=2)

    return (
        f"You are solving a Boolean Satisfiability (SAT) problem.\n"
        f"\n"
        f"The formula has {num_vars} variables and {num_clauses} clauses.\n"
        f"Each clause is a disjunction (OR) of literals. All clauses must\n"
        f"be satisfied simultaneously (AND).\n"
        f"\n"
        f"A positive integer means that variable must be True; a negative\n"
        f"integer means that variable must be False. At least one literal\n"
        f"in each clause must be satisfied.\n"
        f"\n"
        f"{structured_text}\n"
        f"\n"
        f"Find a satisfying truth assignment. Think through your reasoning\n"
        f"step by step, then provide the assignment as a Python dict mapping\n"
        f"variable numbers (integers) to boolean values.\n"
        f"\n"
        f"Format your answer as: {{1: True, 2: False, ...}}\n"
        f"Include ALL {num_vars} variables in your assignment."
    )


# Encoding registry for dispatch
_ENCODERS: dict[str, type[object]] = {}  # unused, using function dispatch instead

_ENCODING_FUNCTIONS = {
    "dimacs": encode_dimacs,
    "natural": encode_natural_language,
    "structured": encode_structured,
}


def encode_for_llm(formula: CNFFormula, encoding: str = "structured") -> str:
    """Encode a CNF formula as a prompt for the LLM.

    Dispatches to the appropriate encoding function based on the
    encoding name.

    Args:
        formula: The CNF formula to encode.
        encoding: Encoding strategy name ("dimacs", "natural", "structured").

    Returns:
        A prompt string suitable for sending to an LLM.

    Raises:
        ValueError: If the encoding name is not recognized.
    """
    encoder = _ENCODING_FUNCTIONS.get(encoding)
    if encoder is None:
        valid = ", ".join(sorted(_ENCODING_FUNCTIONS.keys()))
        raise ValueError(
            f"Unknown encoding {encoding!r}. Valid encodings: {valid}"
        )
    return encoder(formula)


# ---------------------------------------------------------------------------
# Prompt Strategies
# ---------------------------------------------------------------------------
#
# A "prompt strategy" wraps an encoding with additional reasoning scaffolding.
# The baseline strategy uses the raw encoding. Other strategies add chain-of-
# thought instructions, structural hints, or multi-stage decomposition.
#
# Each strategy function takes a CNFFormula and an encoding name, returning
# the final prompt string. The incremental strategy is special: it returns
# a *callable* that manages multi-round interaction with the LLM backend.
# ---------------------------------------------------------------------------


def _format_clause_natural(clause: Clause) -> str:
    """Format a single clause as a human-readable OR expression.

    Args:
        clause: A clause to format.

    Returns:
        String like "(x1 OR NOT x2 OR x3)".
    """
    parts: list[str] = []
    for lit in clause.literals:
        if lit.positive:
            parts.append(f"x{lit.variable.index}")
        else:
            parts.append(f"NOT x{lit.variable.index}")
    return "(" + " OR ".join(parts) + ")"


def strategy_baseline(formula: CNFFormula, encoding: str = "structured") -> str:
    """Baseline prompt strategy: encode the formula and ask for an assignment.

    This is the simplest strategy -- it uses the raw encoding with no
    additional reasoning scaffolding.

    Args:
        formula: The CNF formula to solve.
        encoding: The encoding format to use.

    Returns:
        A prompt string.
    """
    return encode_for_llm(formula, encoding)


def strategy_chain_of_thought(
    formula: CNFFormula, encoding: str = "structured"
) -> str:
    """Chain-of-thought prompt strategy: guide the LLM through DPLL-style reasoning.

    Instructs the LLM to analyze the formula step by step before proposing
    an assignment. The steps mirror classical SAT-solving heuristics:
    unit propagation, pure literal elimination, and constraint-driven
    variable ordering.

    Args:
        formula: The CNF formula to solve.
        encoding: The encoding format to use.

    Returns:
        A prompt string with chain-of-thought instructions.
    """
    base_prompt = encode_for_llm(formula, encoding)
    num_vars = formula.num_variables

    cot_instructions = (
        "\n\n"
        "IMPORTANT: Before giving your answer, reason through the formula "
        "step by step using the following procedure:\n"
        "\n"
        "Step 1 -- Unit clauses: Identify any clauses with exactly one literal. "
        "These force a variable assignment. Apply those assignments and simplify.\n"
        "\n"
        "Step 2 -- Pure literals: Find any variable that appears only positively "
        "or only negatively across all clauses. Set those variables to satisfy "
        "their polarity and simplify.\n"
        "\n"
        "Step 3 -- Most constrained variable: Among remaining variables, pick "
        "the one appearing in the most clauses. Consider what value satisfies "
        "the most clauses containing it.\n"
        "\n"
        "Step 4 -- Propagate implications: After choosing a value for that "
        "variable, check which clauses are now satisfied and which are reduced. "
        "Look for new unit clauses or pure literals created by the assignment.\n"
        "\n"
        "Step 5 -- Repeat Steps 1-4 until all variables are assigned.\n"
        "\n"
        "Step 6 -- Verify: Check your proposed assignment against EVERY clause. "
        "If any clause is not satisfied, revisit your reasoning and fix it.\n"
        "\n"
        "Show your reasoning for each step, then provide the final assignment "
        f"as a Python dict: {{1: True, 2: False, ...}} with all {num_vars} variables."
    )

    return base_prompt + cot_instructions


def _compute_variable_stats(
    formula: CNFFormula,
) -> dict[str, object]:
    """Compute structural statistics about a CNF formula.

    Pre-analyzes the formula to extract features that help the LLM
    understand which variables and clauses are most constrained.

    Args:
        formula: The CNF formula to analyze.

    Returns:
        A dict with the following keys:
        - pos_counts: dict mapping var index to positive occurrence count
        - neg_counts: dict mapping var index to negative occurrence count
        - total_counts: dict mapping var index to total occurrence count
        - clause_lengths: list of clause lengths
        - unit_clauses: list of (clause_index, literal_str) for unit clauses
        - pure_literals: list of (var_index, polarity_str) for pure literals
        - most_constrained: list of (var_index, count) sorted by count desc
    """
    pos_counts: dict[int, int] = defaultdict(int)
    neg_counts: dict[int, int] = defaultdict(int)
    clause_lengths: list[int] = []
    unit_clauses: list[tuple[int, str]] = []

    for i, clause in enumerate(formula.clauses, 1):
        clause_lengths.append(len(clause.literals))
        if clause.is_unit():
            lit = clause.literals[0]
            lit_str = f"x{lit.variable.index}" if lit.positive else f"NOT x{lit.variable.index}"
            unit_clauses.append((i, lit_str))
        for lit in clause.literals:
            if lit.positive:
                pos_counts[lit.variable.index] += 1
            else:
                neg_counts[lit.variable.index] += 1

    all_vars = sorted(
        set(pos_counts.keys()) | set(neg_counts.keys())
    )
    total_counts: dict[int, int] = {
        v: pos_counts.get(v, 0) + neg_counts.get(v, 0) for v in all_vars
    }

    # Pure literals: appear only positively or only negatively
    pure_literals: list[tuple[int, str]] = []
    for v in all_vars:
        p = pos_counts.get(v, 0)
        n = neg_counts.get(v, 0)
        if p > 0 and n == 0:
            pure_literals.append((v, "positive"))
        elif n > 0 and p == 0:
            pure_literals.append((v, "negative"))

    # Most constrained: sorted by total occurrence count descending
    most_constrained = sorted(
        total_counts.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "pos_counts": dict(pos_counts),
        "neg_counts": dict(neg_counts),
        "total_counts": total_counts,
        "clause_lengths": clause_lengths,
        "unit_clauses": unit_clauses,
        "pure_literals": pure_literals,
        "most_constrained": most_constrained,
    }


def strategy_constraint_highlight(
    formula: CNFFormula, encoding: str = "structured"
) -> str:
    """Constraint-highlighting prompt strategy: include structural analysis.

    Pre-computes structural features of the formula -- variable occurrence
    counts, clause length distribution, unit clauses, pure literals, and
    the most constrained variables -- and includes them in the prompt so
    the LLM has explicit hints about where the hard constraints lie.

    Args:
        formula: The CNF formula to solve.
        encoding: The encoding format to use.

    Returns:
        A prompt string with structural analysis included.
    """
    base_prompt = encode_for_llm(formula, encoding)
    stats = _compute_variable_stats(formula)
    num_vars = formula.num_variables

    # Build the structural analysis section
    sections: list[str] = [
        "\n\n--- STRUCTURAL ANALYSIS (pre-computed) ---\n"
    ]

    # Clause length distribution
    clause_lengths = stats["clause_lengths"]
    length_dist: dict[int, int] = defaultdict(int)
    for cl in clause_lengths:  # type: ignore[union-attr]
        length_dist[cl] += 1
    dist_str = ", ".join(
        f"length {k}: {v} clauses" for k, v in sorted(length_dist.items())
    )
    sections.append(f"Clause lengths: {dist_str}")

    # Clause-to-variable ratio and hardness hint
    ratio = formula.clause_variable_ratio
    sections.append(f"Clause/variable ratio: {ratio:.2f}")
    if 4.0 <= ratio <= 4.5:
        sections.append(
            "WARNING: This ratio is near the 3-SAT phase transition (~4.267). "
            "The instance is likely to be very hard."
        )

    # Unit clauses
    unit_clauses = stats["unit_clauses"]
    if unit_clauses:
        uc_str = ", ".join(
            f"clause {idx}: {lit}" for idx, lit in unit_clauses  # type: ignore[union-attr]
        )
        sections.append(f"Unit clauses (forced assignments): {uc_str}")
    else:
        sections.append("Unit clauses: none")

    # Pure literals
    pure_literals = stats["pure_literals"]
    if pure_literals:
        pl_str = ", ".join(
            f"x{v} (always {pol})" for v, pol in pure_literals  # type: ignore[union-attr]
        )
        sections.append(f"Pure literals (safe to set): {pl_str}")
    else:
        sections.append("Pure literals: none")

    # Most constrained variables (top 5 or all if fewer)
    most_constrained = stats["most_constrained"]
    top_n = min(5, len(most_constrained))  # type: ignore[arg-type]
    if most_constrained:
        mc_str = ", ".join(
            f"x{v} ({c} occurrences)"
            for v, c in most_constrained[:top_n]  # type: ignore[index]
        )
        sections.append(f"Most constrained variables: {mc_str}")

    # Variable polarity bias
    pos_counts = stats["pos_counts"]
    neg_counts = stats["neg_counts"]
    bias_hints: list[str] = []
    for v, _count in most_constrained[:top_n]:  # type: ignore[union-attr]
        p = pos_counts.get(v, 0)  # type: ignore[union-attr]
        n = neg_counts.get(v, 0)  # type: ignore[union-attr]
        if p > n:
            bias_hints.append(f"x{v}: appears positive {p}x vs negative {n}x (try True)")
        elif n > p:
            bias_hints.append(f"x{v}: appears negative {n}x vs positive {p}x (try False)")
        else:
            bias_hints.append(f"x{v}: balanced ({p} positive, {n} negative)")
    if bias_hints:
        sections.append("Polarity hints:\n  " + "\n  ".join(bias_hints))

    sections.append(
        "\nUse these hints to guide your search. Start with forced assignments "
        "(unit clauses), then pure literals, then tackle the most constrained "
        "variables using the polarity hints above.\n"
        f"\nProvide the final assignment as: {{1: True, 2: False, ...}} "
        f"with all {num_vars} variables."
    )

    return base_prompt + "\n".join(sections)


def _simplify_formula(
    formula: CNFFormula, partial: dict[int, bool]
) -> CNFFormula:
    """Simplify a CNF formula given a partial truth assignment.

    Removes clauses satisfied by the partial assignment and removes
    falsified literals from remaining clauses. This is the standard
    unit propagation simplification used in DPLL.

    Args:
        formula: The CNF formula to simplify.
        partial: Partial assignment (var index -> bool).

    Returns:
        A simplified CNFFormula with satisfied clauses removed and
        falsified literals eliminated.
    """
    new_clauses: list[Clause] = []
    for clause in formula.clauses:
        # Check if any literal in the clause is satisfied
        satisfied = False
        remaining_lits: list[Literal] = []
        for lit in clause.literals:
            var_idx = lit.variable.index
            if var_idx in partial:
                val = partial[var_idx]
                if (lit.positive and val) or (not lit.positive and not val):
                    satisfied = True
                    break
                # Literal is falsified -- skip it
            else:
                remaining_lits.append(lit)
        if not satisfied:
            new_clauses.append(Clause(tuple(remaining_lits)))
    return CNFFormula(tuple(new_clauses))


def strategy_incremental(
    formula: CNFFormula, encoding: str = "structured"
) -> str:
    """Incremental prompt strategy: first-stage prompt for the most constrained variables.

    For larger formulas, this strategy breaks the problem into stages.
    The first call asks the LLM to assign only the most constrained
    variables. Subsequent calls (handled by the solver's incremental
    loop) simplify the formula with those assignments and ask for the
    next batch.

    This function generates the prompt for the first stage. The solver
    uses ``generate_incremental_continuation`` for subsequent stages.

    Args:
        formula: The CNF formula to solve.
        encoding: The encoding format to use.

    Returns:
        A prompt string for the first incremental stage.
    """
    stats = _compute_variable_stats(formula)
    most_constrained = stats["most_constrained"]
    num_vars = formula.num_variables

    # Select the top batch of variables to assign first.
    # Use roughly 1/3 of variables per stage, minimum 2, maximum 10.
    batch_size = max(2, min(10, num_vars // 3))
    batch_vars = [v for v, _c in most_constrained[:batch_size]]  # type: ignore[union-attr]

    base_prompt = encode_for_llm(formula, encoding)

    incremental_instructions = (
        f"\n\nINCREMENTAL SOLVING MODE\n"
        f"This formula has {num_vars} variables. To make it more manageable, "
        f"focus on assigning ONLY the following {len(batch_vars)} most "
        f"constrained variables first:\n"
        f"\n"
        f"  Target variables: {', '.join(f'x{v}' for v in batch_vars)}\n"
        f"\n"
        f"These variables appear in the most clauses and will have the "
        f"greatest simplification effect. Consider the clauses containing "
        f"these variables and choose values that satisfy as many clauses "
        f"as possible.\n"
        f"\n"
        f"Provide ONLY assignments for the target variables as a Python dict:\n"
        f"  {{{', '.join(f'{v}: True/False' for v in batch_vars)}}}\n"
    )

    return base_prompt + incremental_instructions


def generate_incremental_continuation(
    formula: CNFFormula,
    partial: dict[int, bool],
    encoding: str = "structured",
) -> Optional[str]:
    """Generate a continuation prompt for the incremental strategy.

    After the LLM assigns some variables, this function simplifies the
    formula with those assignments and generates a prompt for the
    remaining variables.

    Args:
        formula: The original (unsimplified) CNF formula.
        partial: The partial assignment from previous stages.
        encoding: The encoding format to use.

    Returns:
        A prompt string for the next stage, or None if all variables
        are already assigned.
    """
    all_vars = formula.get_variables()
    remaining_vars = frozenset(
        v for v in all_vars if v.index not in partial
    )

    if not remaining_vars:
        return None

    simplified = _simplify_formula(formula, partial)

    # If all clauses are satisfied, no need to continue
    if simplified.num_clauses == 0:
        return None

    remaining_indices = sorted(v.index for v in remaining_vars)
    assigned_str = ", ".join(
        f"x{k}={'True' if v else 'False'}" for k, v in sorted(partial.items())
    )

    base_prompt = encode_for_llm(simplified, encoding)

    continuation = (
        f"\n\nINCREMENTAL SOLVING -- CONTINUATION\n"
        f"Previously assigned variables: {assigned_str}\n"
        f"The formula has been simplified using those assignments.\n"
        f"\n"
        f"Remaining variables to assign: "
        f"{', '.join(f'x{i}' for i in remaining_indices)}\n"
        f"\n"
        f"Assign ALL remaining variables to satisfy the simplified formula.\n"
        f"\n"
        f"Provide assignments as a Python dict:\n"
        f"  {{{', '.join(f'{i}: True/False' for i in remaining_indices)}}}\n"
    )

    return base_prompt + continuation


# ---------------------------------------------------------------------------
# Prompt Strategy Registry
# ---------------------------------------------------------------------------

PROMPT_STRATEGIES: dict[str, Callable[[CNFFormula, str], str]] = {
    "baseline": strategy_baseline,
    "chain_of_thought": strategy_chain_of_thought,
    "constraint_highlight": strategy_constraint_highlight,
    "incremental": strategy_incremental,
}


def get_prompt(
    formula: CNFFormula,
    strategy: str = "baseline",
    encoding: str = "structured",
) -> str:
    """Generate a prompt for the LLM using the specified strategy and encoding.

    Dispatches to the appropriate strategy function from the registry.

    Args:
        formula: The CNF formula to solve.
        strategy: Prompt strategy name. One of "baseline", "chain_of_thought",
            "constraint_highlight", or "incremental".
        encoding: Encoding format name. One of "dimacs", "natural", "structured".

    Returns:
        A prompt string.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    strategy_fn = PROMPT_STRATEGIES.get(strategy)
    if strategy_fn is None:
        valid = ", ".join(sorted(PROMPT_STRATEGIES.keys()))
        raise ValueError(
            f"Unknown prompt strategy {strategy!r}. Valid strategies: {valid}"
        )
    return strategy_fn(formula, encoding)


# ---------------------------------------------------------------------------
# LLM Response Parsing
# ---------------------------------------------------------------------------


def parse_llm_response(
    response: str,
    expected_variables: frozenset[Variable],
) -> Optional[dict[int, bool]]:
    """Parse an LLM text response into a truth assignment.

    Handles multiple response formats:
    - Python dict literal: ``{1: True, 2: False, ...}``
    - JSON format: ``{"1": true, "2": false, ...}``
    - List format: ``x1=True, x2=False, ...``
    - Embedded in reasoning text (extracts dict from surrounding prose)

    Missing variables are filled with False (matching DPLL convention).
    Returns None only if parsing completely fails.

    Args:
        response: The raw text response from the LLM.
        expected_variables: The set of variables expected in the assignment.

    Returns:
        A complete truth assignment dict, or None if parsing failed.
    """
    expected_indices = frozenset(v.index for v in expected_variables)

    # Strategy 1: Extract Python dict literal from response
    assignment = _parse_python_dict(response, expected_indices)
    if assignment is not None:
        return assignment

    # Strategy 2: Extract JSON object from response
    assignment = _parse_json_dict(response, expected_indices)
    if assignment is not None:
        return assignment

    # Strategy 3: Extract x1=True, x2=False patterns
    assignment = _parse_list_format(response, expected_indices)
    if assignment is not None:
        return assignment

    return None


def _parse_python_dict(
    text: str, expected_indices: frozenset[int]
) -> Optional[dict[int, bool]]:
    """Extract a Python dict literal from LLM response text.

    Searches for patterns like ``{1: True, 2: False}`` in the text.

    Args:
        text: The LLM response text.
        expected_indices: Variable indices expected in the assignment.

    Returns:
        Parsed assignment dict, or None if extraction failed.
    """
    # Find all dict-like patterns in the text
    # Match { ... } blocks containing int: True/False patterns
    pattern = r"\{[^{}]*\b\d+\s*:\s*(?:True|False)\b[^{}]*\}"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            # Extract individual key-value pairs
            assignment = _extract_int_bool_pairs(match)
            if assignment and _has_coverage(assignment, expected_indices):
                return _fill_missing(assignment, expected_indices)
        except (ValueError, KeyError):
            continue

    return None


def _parse_json_dict(
    text: str, expected_indices: frozenset[int]
) -> Optional[dict[int, bool]]:
    """Extract a JSON object from LLM response text.

    Searches for patterns like ``{"1": true, "2": false}`` in the text.

    Args:
        text: The LLM response text.
        expected_indices: Variable indices expected in the assignment.

    Returns:
        Parsed assignment dict, or None if extraction failed.
    """
    # Find JSON-like objects with string keys and boolean values
    pattern = r'\{[^{}]*"?\d+"?\s*:\s*(?:true|false)\b[^{}]*\}'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        try:
            # Normalize to valid JSON: ensure keys are quoted strings
            normalized = re.sub(r"(\d+)\s*:", r'"\1":', match)
            # Normalize boolean values to JSON format
            normalized = re.sub(r"\bTrue\b", "true", normalized)
            normalized = re.sub(r"\bFalse\b", "false", normalized)
            parsed = json.loads(normalized)

            if isinstance(parsed, dict):
                assignment: dict[int, bool] = {}
                for k, v in parsed.items():
                    var_idx = int(k)
                    if isinstance(v, bool):
                        assignment[var_idx] = v
                    elif isinstance(v, (int, float)):
                        assignment[var_idx] = bool(v)

                if assignment and _has_coverage(assignment, expected_indices):
                    return _fill_missing(assignment, expected_indices)
        except (json.JSONDecodeError, ValueError, KeyError):
            continue

    return None


def _parse_list_format(
    text: str, expected_indices: frozenset[int]
) -> Optional[dict[int, bool]]:
    """Extract x1=True, x2=False patterns from LLM response text.

    Args:
        text: The LLM response text.
        expected_indices: Variable indices expected in the assignment.

    Returns:
        Parsed assignment dict, or None if extraction failed.
    """
    pattern = r"x(\d+)\s*=\s*(True|False|true|false|1|0)"
    matches = re.findall(pattern, text)

    if not matches:
        return None

    assignment: dict[int, bool] = {}
    for var_str, val_str in matches:
        var_idx = int(var_str)
        assignment[var_idx] = val_str.lower() in ("true", "1")

    if _has_coverage(assignment, expected_indices):
        return _fill_missing(assignment, expected_indices)

    return None


def _extract_int_bool_pairs(text: str) -> dict[int, bool]:
    """Extract integer-to-boolean pairs from a dict-like string.

    Args:
        text: A string containing patterns like "1: True, 2: False".

    Returns:
        Dict mapping variable indices to boolean values.
    """
    pattern = r"(\d+)\s*:\s*(True|False)"
    matches = re.findall(pattern, text)
    return {int(k): v == "True" for k, v in matches}


def _has_coverage(
    assignment: dict[int, bool], expected: frozenset[int]
) -> bool:
    """Check if an assignment covers at least one expected variable.

    We accept partial assignments (will fill missing with False),
    but require at least some overlap with expected variables
    to filter out spurious dict matches.

    Args:
        assignment: The parsed assignment.
        expected: Expected variable indices.

    Returns:
        True if there is meaningful overlap.
    """
    return bool(set(assignment.keys()) & expected)


def _fill_missing(
    assignment: dict[int, bool], expected: frozenset[int]
) -> dict[int, bool]:
    """Fill missing variables with False (DPLL convention).

    Args:
        assignment: Partial assignment.
        expected: All expected variable indices.

    Returns:
        Complete assignment with all expected variables.
    """
    result = dict(assignment)
    for idx in expected:
        if idx not in result:
            result[idx] = False
    return result


# ---------------------------------------------------------------------------
# LLMOracleSolver
# ---------------------------------------------------------------------------


class LLMOracleSolver:
    """SAT solver using an LLM as a probabilistic oracle.

    Implements the Solver protocol from ``p_equals_np.definitions``.
    Uses the correctness island pattern: stochastic generation (LLM)
    wrapped in deterministic verification (``CNFFormula.evaluate``).

    The solver encodes the CNF formula as a prompt, sends it to the
    LLM backend, parses the response into a truth assignment, and
    verifies the assignment. If verification fails, it retries up to
    ``max_attempts`` times.

    Attributes:
        timeout_seconds: Maximum wall-clock time for the solve call.
    """

    __slots__ = (
        "timeout_seconds",
        "_backend",
        "_encoding",
        "_strategy",
        "_max_attempts",
        "_attempts_made",
        "_successful_attempt",
        "_metrics",
        "_attempt_history",
        "_solve_metrics_history",
    )

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        encoding: str = "structured",
        strategy: str = "baseline",
        max_attempts: int = 5,
        timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize the LLM Oracle solver.

        Args:
            backend: LLM backend to use for generation. If None,
                attempts to create an AnthropicBackend; falls back
                to MockLLMBackend if the API is unavailable.
            encoding: Prompt encoding strategy. One of "dimacs",
                "natural", or "structured" (default).
            strategy: Prompt strategy for reasoning scaffolding. One of
                "baseline", "chain_of_thought", "constraint_highlight",
                or "incremental" (default "baseline").
            max_attempts: Maximum number of LLM calls per solve
                invocation (default 5).
            timeout_seconds: Wall-clock timeout in seconds (default 60).
        """
        if backend is not None:
            self._backend: LLMBackend = backend
        else:
            anthropic_backend = AnthropicBackend()
            if anthropic_backend._client is not None:
                self._backend = anthropic_backend
            else:
                self._backend = MockLLMBackend()

        if encoding not in _ENCODING_FUNCTIONS:
            valid = ", ".join(sorted(_ENCODING_FUNCTIONS.keys()))
            raise ValueError(
                f"Unknown encoding {encoding!r}. Valid encodings: {valid}"
            )
        if strategy not in PROMPT_STRATEGIES:
            valid = ", ".join(sorted(PROMPT_STRATEGIES.keys()))
            raise ValueError(
                f"Unknown strategy {strategy!r}. Valid strategies: {valid}"
            )
        self._encoding = encoding
        self._strategy = strategy
        self._max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self._attempts_made: int = 0
        self._successful_attempt: Optional[int] = None
        self._metrics: dict[str, object] = {}
        self._attempt_history: list[AttemptRecord] = []
        self._solve_metrics_history: list[OracleSolveMetrics] = []

    def solve(self, formula: CNFFormula) -> Optional[dict[int, bool]]:
        """Attempt to solve a CNF formula using the LLM oracle.

        Encodes the formula using the configured strategy and encoding,
        queries the LLM for a candidate assignment, parses the response,
        and verifies using ``formula.evaluate()``. Retries up to
        ``max_attempts`` times on failure.

        When the "incremental" strategy is selected, the solver uses a
        multi-stage approach: it asks the LLM to assign the most
        constrained variables first, simplifies the formula, and then
        asks for the remaining variables in subsequent rounds.

        The LLM's output is NEVER trusted. Only verified assignments
        (those passing ``formula.evaluate()``) are returned.

        Args:
            formula: A CNF formula to solve.

        Returns:
            A satisfying assignment ``dict[int, bool]`` if found,
            or None if all attempts failed or the formula is UNSAT.
        """
        self._attempts_made = 0
        self._successful_attempt = None
        self._attempt_history = []
        self._metrics = {
            "num_variables": formula.num_variables,
            "num_clauses": formula.num_clauses,
            "encoding": self._encoding,
            "strategy": self._strategy,
            "parse_failures": 0,
            "verify_failures": 0,
        }

        solve_start = time.monotonic()

        # Capture token counts before this solve (for delta calculation)
        tokens_before = self._get_backend_tokens()

        # Handle trivial case: empty formula is vacuously satisfiable
        if formula.num_clauses == 0:
            variables = formula.get_variables()
            result = {v.index: False for v in variables}
            self._successful_attempt = 0  # trivially solved, no LLM call
            self._record_solve_metrics(formula, solve_start, tokens_before)
            return result

        if self._strategy == "incremental":
            result = self._solve_incremental(formula)
        else:
            result = self._solve_standard(formula)

        self._record_solve_metrics(formula, solve_start, tokens_before)
        return result

    def _solve_standard(
        self, formula: CNFFormula
    ) -> Optional[dict[int, bool]]:
        """Standard solve loop with intelligent retry on failure.

        Implements the full retry framework:
        1. Failure diagnosis -- classify why an attempt failed and
           identify violated clauses for WRONG_ASSIGNMENT failures.
        2. Prompt modification -- include diagnostic feedback from the
           previous attempt in the next prompt.
        3. Strategy rotation -- cycle through baseline, chain-of-thought,
           and constraint-highlight strategies across attempts.
        4. Temperature escalation -- increase sampling randomness on
           subsequent attempts (0.3 -> 0.5 -> 0.7 -> 0.9 -> 1.0).

        Each attempt is logged with its strategy, temperature, failure
        mode, and outcome in ``_attempt_history``.

        Args:
            formula: The CNF formula to solve.

        Returns:
            A satisfying assignment, or None.
        """
        expected_variables = formula.get_variables()
        num_vars = formula.num_variables
        start_time = time.monotonic()
        retry_feedback: str = ""

        # Save the original backend temperature to restore after solving
        original_temperature: Optional[float] = getattr(
            self._backend, "temperature", None
        )

        try:
            for attempt in range(1, self._max_attempts + 1):
                elapsed = time.monotonic() - start_time
                if elapsed >= self.timeout_seconds:
                    logger.debug(
                        "Timeout after %.1fs at attempt %d", elapsed, attempt
                    )
                    break

                self._attempts_made = attempt
                attempt_index = attempt - 1

                # -- Strategy rotation --
                strategy = _get_retry_strategy(attempt_index, self._strategy)

                # -- Temperature escalation --
                temperature = _get_retry_temperature(attempt_index)
                _set_backend_temperature(self._backend, temperature)

                # -- Build prompt with retry feedback --
                prompt = get_prompt(formula, strategy, self._encoding)
                if retry_feedback:
                    prompt = prompt + retry_feedback

                logger.debug(
                    "Attempt %d: strategy=%s, temperature=%.1f",
                    attempt, strategy, temperature,
                )

                # -- Call the LLM (with per-attempt timing) --
                attempt_start = time.monotonic()
                response = self._backend.generate(prompt)

                # -- Diagnose the result --
                assignment: Optional[dict[int, bool]] = None
                if response is not None:
                    assignment = parse_llm_response(
                        response, expected_variables
                    )

                failure_mode, violated = diagnose_failure(
                    formula, response, assignment
                )
                attempt_elapsed = time.monotonic() - attempt_start

                # Build attempt record for history and logging
                record = AttemptRecord(
                    attempt_number=attempt,
                    strategy=strategy,
                    temperature=temperature,
                    failure_mode=None,  # set below if failed
                    assignment=assignment,
                    violated_clauses=violated,
                    response_text=(
                        response[:500] if response else None
                    ),
                    elapsed_seconds=attempt_elapsed,
                )

                # -- Check for success --
                if assignment is not None and formula.evaluate(assignment):
                    record.failure_mode = None
                    self._attempt_history.append(record)
                    self._successful_attempt = attempt
                    self._metrics["success"] = True
                    self._metrics["winning_strategy"] = strategy
                    self._metrics["winning_temperature"] = temperature
                    logger.info(
                        "Solved on attempt %d (strategy=%s, temp=%.1f)",
                        attempt, strategy, temperature,
                    )
                    return assignment

                # -- Record the failure --
                record.failure_mode = failure_mode
                self._attempt_history.append(record)

                if failure_mode == FailureMode.API_ERROR:
                    self._metrics["parse_failures"] = (  # type: ignore[assignment]
                        int(self._metrics["parse_failures"]) + 1  # type: ignore[arg-type]
                    )
                elif failure_mode == FailureMode.PARSE_ERROR:
                    self._metrics["parse_failures"] = (  # type: ignore[assignment]
                        int(self._metrics["parse_failures"]) + 1  # type: ignore[arg-type]
                    )
                elif failure_mode in (
                    FailureMode.WRONG_ASSIGNMENT,
                    FailureMode.INCOMPLETE_ASSIGNMENT,
                ):
                    self._metrics["verify_failures"] = (  # type: ignore[assignment]
                        int(self._metrics["verify_failures"]) + 1  # type: ignore[arg-type]
                    )

                logger.debug(
                    "Attempt %d failed: %s (%d violated clauses)",
                    attempt, failure_mode.value, len(violated),
                )

                # -- Build feedback for the next attempt --
                retry_feedback = build_retry_feedback(
                    failure_mode, record, num_vars
                )

        finally:
            # Restore original backend temperature
            if original_temperature is not None:
                _set_backend_temperature(self._backend, original_temperature)

        self._metrics["success"] = False
        return None

    def _solve_incremental(
        self, formula: CNFFormula
    ) -> Optional[dict[int, bool]]:
        """Incremental solve loop: assign variables in stages.

        Asks the LLM to assign the most constrained variables first,
        simplifies the formula, and continues until all variables are
        assigned or attempts are exhausted.

        Args:
            formula: The CNF formula to solve.

        Returns:
            A satisfying assignment, or None.
        """
        expected_variables = formula.get_variables()
        partial_assignment: dict[int, bool] = {}
        stage = 0
        self._metrics["incremental_stages"] = 0

        start_time = time.monotonic()

        # First stage prompt
        prompt: Optional[str] = get_prompt(
            formula, "incremental", self._encoding
        )

        while prompt is not None:
            elapsed = time.monotonic() - start_time
            if elapsed >= self.timeout_seconds:
                break
            if self._attempts_made >= self._max_attempts:
                break

            stage += 1
            self._metrics["incremental_stages"] = stage
            self._attempts_made += 1

            attempt_start = time.monotonic()
            response = self._backend.generate(prompt)
            attempt_elapsed = time.monotonic() - attempt_start

            if response is None:
                record = AttemptRecord(
                    attempt_number=self._attempts_made,
                    strategy="incremental",
                    temperature=getattr(self._backend, "temperature", 0.0),
                    failure_mode=FailureMode.API_ERROR,
                    elapsed_seconds=attempt_elapsed,
                )
                self._attempt_history.append(record)
                self._metrics["parse_failures"] = (  # type: ignore[assignment]
                    int(self._metrics["parse_failures"]) + 1  # type: ignore[arg-type]
                )
                break

            # Parse the partial response -- use all expected variables
            # so the parser accepts partial coverage
            stage_assignment = parse_llm_response(
                response, expected_variables
            )
            if stage_assignment is None:
                record = AttemptRecord(
                    attempt_number=self._attempts_made,
                    strategy="incremental",
                    temperature=getattr(self._backend, "temperature", 0.0),
                    failure_mode=FailureMode.PARSE_ERROR,
                    response_text=response[:500] if response else None,
                    elapsed_seconds=attempt_elapsed,
                )
                self._attempt_history.append(record)
                self._metrics["parse_failures"] = (  # type: ignore[assignment]
                    int(self._metrics["parse_failures"]) + 1  # type: ignore[arg-type]
                )
                break

            # Merge new assignments into partial (new values override)
            partial_assignment.update(stage_assignment)

            # Check if we have a complete assignment that works
            if len(partial_assignment) >= len(expected_variables):
                if formula.evaluate(partial_assignment):
                    record = AttemptRecord(
                        attempt_number=self._attempts_made,
                        strategy="incremental",
                        temperature=getattr(self._backend, "temperature", 0.0),
                        failure_mode=None,
                        assignment=partial_assignment,
                        response_text=response[:500] if response else None,
                        elapsed_seconds=attempt_elapsed,
                    )
                    self._attempt_history.append(record)
                    self._successful_attempt = self._attempts_made
                    self._metrics["success"] = True
                    return partial_assignment
                else:
                    _, violated = diagnose_failure(
                        formula, response, partial_assignment
                    )
                    record = AttemptRecord(
                        attempt_number=self._attempts_made,
                        strategy="incremental",
                        temperature=getattr(self._backend, "temperature", 0.0),
                        failure_mode=FailureMode.WRONG_ASSIGNMENT,
                        assignment=partial_assignment,
                        violated_clauses=violated,
                        response_text=response[:500] if response else None,
                        elapsed_seconds=attempt_elapsed,
                    )
                    self._attempt_history.append(record)
                    self._metrics["verify_failures"] = (  # type: ignore[assignment]
                        int(self._metrics["verify_failures"]) + 1  # type: ignore[arg-type]
                    )
                    break
            else:
                # Intermediate stage -- record as a successful parse step
                record = AttemptRecord(
                    attempt_number=self._attempts_made,
                    strategy="incremental",
                    temperature=getattr(self._backend, "temperature", 0.0),
                    failure_mode=None,
                    assignment=dict(stage_assignment),
                    response_text=response[:500] if response else None,
                    elapsed_seconds=attempt_elapsed,
                )
                self._attempt_history.append(record)

            # Generate continuation prompt for remaining variables
            prompt = generate_incremental_continuation(
                formula, partial_assignment, self._encoding
            )

        # Final fallback: if we have a partial assignment, fill missing
        # variables with False and check
        if partial_assignment:
            all_indices = frozenset(v.index for v in expected_variables)
            full = _fill_missing(partial_assignment, all_indices)
            if formula.evaluate(full):
                self._successful_attempt = self._attempts_made
                self._metrics["success"] = True
                return full
            self._metrics["verify_failures"] = (  # type: ignore[assignment]
                int(self._metrics["verify_failures"]) + 1  # type: ignore[arg-type]
            )

        self._metrics["success"] = False
        return None

    def name(self) -> str:
        """Return the human-readable solver name.

        Returns:
            A string identifying this solver.
        """
        return f"LLMOracle({self._strategy})"

    def complexity_claim(self) -> str:
        """State the honest complexity claim for this solver.

        The LLM oracle requires exponentially many calls in expectation
        for hard instances. Each call has polynomial verification cost,
        but the number of calls grows exponentially.

        Returns:
            A string describing the expected complexity.
        """
        return (
            "O(2^n / p(n)) expected LLM calls where p(n) is the probability "
            "of a correct guess; exponential for hard random 3-SAT instances"
        )

    @property
    def attempts_made(self) -> int:
        """Number of LLM calls made in the last solve invocation."""
        return self._attempts_made

    @property
    def successful_attempt(self) -> Optional[int]:
        """The attempt number that succeeded, or None if all failed."""
        return self._successful_attempt

    @property
    def metrics(self) -> dict[str, object]:
        """Metrics from the last solve invocation."""
        return dict(self._metrics)

    @property
    def attempt_history(self) -> list[AttemptRecord]:
        """Full history of attempts from the last solve invocation."""
        return list(self._attempt_history)

    # -- Metrics API --

    def _get_backend_tokens(self) -> int:
        """Read cumulative token count from the backend, if available.

        Returns:
            Total tokens (input + output) from the backend, or 0 if
            the backend does not track tokens.
        """
        input_tokens = getattr(self._backend, "total_input_tokens", 0) or 0
        output_tokens = getattr(self._backend, "total_output_tokens", 0) or 0
        return input_tokens + output_tokens

    def _record_solve_metrics(
        self,
        formula: CNFFormula,
        solve_start: float,
        tokens_before: int,
    ) -> None:
        """Build an OracleSolveMetrics record and append to history.

        Called at the end of every solve() invocation to capture the
        per-solve metrics snapshot.

        Args:
            formula: The formula that was solved.
            solve_start: Monotonic timestamp when solve() began.
            tokens_before: Token count from backend before this solve.
        """
        total_elapsed = time.monotonic() - solve_start
        tokens_after = self._get_backend_tokens()
        tokens_used = tokens_after - tokens_before

        # Extract per-attempt data from attempt history
        strategies: list[str] = []
        temperatures: list[float] = []
        failure_modes: list[str] = []
        clauses_violated: list[int] = []
        per_attempt_times: list[float] = []

        for record in self._attempt_history:
            strategies.append(record.strategy)
            temperatures.append(record.temperature)
            per_attempt_times.append(record.elapsed_seconds)
            clauses_violated.append(len(record.violated_clauses))
            if record.failure_mode is not None:
                failure_modes.append(record.failure_mode.value)

        metrics = OracleSolveMetrics(
            num_variables=formula.num_variables,
            num_clauses=formula.num_clauses,
            clause_ratio=formula.clause_variable_ratio,
            encoding=self._encoding,
            attempts_made=self._attempts_made,
            successful=self._successful_attempt is not None,
            successful_attempt=self._successful_attempt,
            strategies_used=tuple(strategies),
            temperatures_used=tuple(temperatures),
            failure_modes=tuple(failure_modes),
            clauses_violated_per_attempt=tuple(clauses_violated),
            total_elapsed_seconds=total_elapsed,
            per_attempt_seconds=tuple(per_attempt_times),
            total_tokens_used=tokens_used,
        )
        self._solve_metrics_history.append(metrics)

    def get_metrics(self) -> list[OracleSolveMetrics]:
        """Return all per-solve metrics accumulated since last reset.

        Metrics accumulate across multiple solve() calls. Use
        reset_metrics() to clear the history.

        Returns:
            List of OracleSolveMetrics, one per solve() call.
        """
        return list(self._solve_metrics_history)

    def get_aggregate_metrics(self) -> OracleAggregateMetrics:
        """Compute aggregate metrics from all accumulated solve records.

        Returns:
            OracleAggregateMetrics summarizing all solve() calls
            since the last reset.
        """
        return _compute_aggregate_metrics(self._solve_metrics_history)

    def reset_metrics(self) -> None:
        """Clear all accumulated solve metrics."""
        self._solve_metrics_history = []

    def export_metrics(
        self, filepath: str, include_aggregate: bool = True
    ) -> None:
        """Export accumulated metrics to a JSON file.

        Args:
            filepath: Output file path for the JSON export.
            include_aggregate: Whether to include aggregate summary.
        """
        export_metrics_json(
            self._solve_metrics_history, filepath, include_aggregate
        )
