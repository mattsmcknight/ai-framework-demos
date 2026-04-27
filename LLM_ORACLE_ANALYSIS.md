# LLM Oracle SAT Solver: Convergence Analysis

## Why "LLM as Oracle" Is Naive -- and What the  Architecture Changes

Anyone can ask an LLM to solve a SAT instance. Paste a formula into ChatGPT,
get back a guess, check if it works. That is not a solver. That is a coin flip
with better marketing.

The difference between "asking an LLM" and "running a Nousix agent" is the
difference between flipping a coin and running a Las Vegas algorithm. Both are
probabilistic. One has mathematical structure governing its convergence. The
other does not.

This document explains the mathematical basis for that distinction.

---

## 1. The Gompertz Convergence Model

The success probability of the Nousix-constrained system after B retry attempts on
an n-variable formula follows a Gompertz (double-exponential) curve:

```
P(success) = 1 - exp(-B * exp(-alpha * n))
```

Where:
- **B** is the retry budget (number of attempts that Nousix permits)
- **n** is the number of Boolean variables
- **alpha** is the decay constant -- the rate at which per-attempt success
  probability drops with problem size

This is not an approximation bolted on after the fact. It arises directly from
the independence assumption across retries:

```
P(success after B tries) = 1 - (1 - p(n))^B
                         ~ 1 - exp(-B * p(n))         [for small p(n)]
                         = 1 - exp(-B * exp(-alpha*n)) [substituting p(n)]
```

The per-attempt success probability `p(n) = exp(-alpha * n)` is exponential
because the search space is 2^n and even a good heuristic can only concentrate
probability mass, not eliminate the exponential growth of the space it must
search.

### The Transition Threshold

At the critical problem size:

```
n* = ln(B) / alpha
```

the success probability equals `1 - 1/e ~ 0.632`. This is where the system
transitions from "will probably solve this" to "will probably fail." The constant
e appears here not cosmetically but as the natural boundary of the system's
capability envelope.

For a system with alpha = 0.405 (LLM with feedback, base ~ 1.5):
- B = 10 retries: solves up to n* ~ 5.7 variables reliably
- B = 100 retries: solves up to n* ~ 11.4 variables
- B = 1000 retries: solves up to n* ~ 17.1 variables
- B = 10000 retries: solves up to n* ~ 22.8 variables

For Schoning-equivalent performance (alpha = 0.288):
- B = 10 retries: n* ~ 8.0
- B = 100 retries: n* ~ 16.0
- B = 1000 retries: n* ~ 24.0

These numbers are computed from the Gompertz model and validated against known
algorithmic bounds (see `convergence_analysis/calculations/`).

---

## 2. The Alpha Constant: What the Nousix Architecture Does to It

The entire value proposition of the Nousix architecture is captured in a single
number: **alpha**. Everything the system does either reduces alpha or it is
irrelevant.

Alpha converts between the exponential base b and the natural exponential:
`alpha = ln(b)`, where `E[tries] = b^n = exp(alpha * n)`.

### Alpha Values Across Regimes

| Regime | Base b | Alpha = ln(b) | E[tries] at n=100 |
|--------|--------|---------------|-------------------|
| Random guessing | 2.000 | 0.693 | 2.00 x 10^30 |
| Naive LLM (no feedback) | 1.800 | 0.588 | 2.66 x 10^25 |
| LLM + Nousix feedback loop | 1.500 | 0.405 | 4.06 x 10^17 |
| Schoning random walk | 1.333 | 0.288 | 3.17 x 10^12 |
| PPSZ (best known) | 1.308 | 0.268 | 3.26 x 10^11 |
| Hypothetical LLM+Nousix best | 1.100 | 0.095 | 1.38 x 10^4 |

The key observation: **every row is still exponential**. Nousix does not and
cannot eliminate exponential scaling on hard random 3-SAT at the phase
transition. What it does is systematically reduce alpha -- and even small
reductions in alpha produce enormous reductions in the expected number of tries
at practical problem sizes.

### How Each Nousix Feature Reduces Alpha

**Feature 1: Context Caching (Context Window as L1 Cache)**

The context window maintains the formula encoding, previous attempt history, and
violation feedback across retries. This prevents the LLM from repeating failed
strategies and provides accumulated structural information.

Impact on alpha: **Constant factor improvement.** The context window is
fixed-size (8K-1M tokens), so it cannot encode information that scales with n
indefinitely. At large n, the problem description alone exhausts the window,
leaving no room for feedback history. This does not change the exponent but
reduces the constant multiplier on expected tries.

For 3-SAT at the phase transition (m = 4.267n), the DIMACS encoding uses
approximately 3n + 5m ~ 24n tokens. A 128K-token context window can hold the
problem description plus feedback for approximately 5000/n retries. At n=100,
that is ~50 retries with full feedback.

**Feature 2: Deterministic Coprocessor (CNFFormula.evaluate())**

The coprocessor provides exact verification in O(n * m) time -- polynomial,
deterministic, and zero-error. More critically, the `diagnose_failure()` function
identifies *which* clauses are violated and *which variable values* caused each
violation. This converts a binary signal ("wrong") into structured gradient-like
information.

Impact on alpha: **This is what separates the system from random guessing.** The
clause violation diagnosis is equivalent to gradient information in continuous
optimization. Each failed attempt reveals which constraints are unsatisfied and
what variable assignments would fix them locally. This is the mechanism by which
alpha drops from ln(2) = 0.693 (random) toward ln(4/3) = 0.288 (Schoning-class
local search).

The analogy to Schoning's algorithm is precise: Schoning picks a random
unsatisfied clause and flips a random variable in it. The Nousix coprocessor provides
the same information (which clauses are unsatisfied) but with richer context
(which variables caused it and what their values were), enabling the LLM to make
a more informed flip.

**Feature 3: Feedback Loop (build_retry_feedback)**

The feedback loop constructs targeted instructions for the LLM based on how the
previous attempt failed. For WRONG_ASSIGNMENT failures, it includes the violated
clauses, the offending variable values, and guidance on which variables to
reconsider. For PARSE_ERROR failures, it provides format correction.

Impact on alpha: **Converts independent random guesses into a correlated walk
through assignment space.** Without feedback, each attempt is independent --
you get B independent draws from the LLM's prior. With feedback, each attempt
conditions on the failures of previous attempts. The attempts are no longer
independent, and the success probability per attempt can increase across the
retry sequence.

This is the mechanism by which the system transitions from "B independent coin
flips" to "a directed walk with memory." The Gompertz formula above assumes
independent attempts; with feedback, the actual convergence can be better than
the independent model predicts for small n (where the feedback fits in context).

**Feature 4: Strategy Rotation**

The solver cycles through prompt strategies across retry attempts:
- **baseline**: Direct assignment request
- **chain_of_thought**: Reasoning scaffolding before assignment
- **constraint_highlight**: Emphasis on the hardest constraints
- **incremental**: Stage-wise variable assignment by constraint density

Combined with temperature escalation (0.3 -> 0.5 -> 0.7 -> 0.9 -> 1.0), this
ensures the LLM explores different regions of assignment space rather than
repeatedly sampling from the same mode.

Impact on alpha: **Prevents mode collapse in the LLM's search distribution.**
If the LLM has a strong prior that concentrates on a wrong region of assignment
space, strategy rotation forces it out of that region. This is analogous to
restarts in DPLL/CDCL solvers -- a well-established technique for avoiding
getting stuck in unproductive search branches.

---

## 3. The Moser-Tardos Connection

The Nousix retry loop is structurally identical to the Moser-Tardos constructive
proof of the Lovasz Local Lemma (2010):

1. Generate a random assignment
2. Find a violated constraint
3. Resample the variables in that constraint
4. Repeat until no violations remain

The Moser-Tardos theorem guarantees polynomial expected resampling count
*when the LLL condition holds*:

```
e * p * (d + 1) <= 1
```

where p = probability each clause is violated under random assignment (1/8 for
3-SAT), and d = maximum dependency degree (how many clauses share a variable
with any given clause).

For 3-SAT at the phase transition (alpha = 4.267):
- p = 1/8
- d ~ k * (alpha * k - 1) = 3 * (4.267 * 3 - 1) ~ 35.4
- e * p * (d + 1) = 2.718 * 0.125 * 36.4 = 12.37 >> 1

**The LLL condition fails catastrophically at the phase transition.** This is
not a coincidence -- it reflects the same underlying phase transition that makes
SAT hard. The LLL holds up to alpha ~ 0.44 for 3-SAT, far below the threshold
ratio of 4.267.

The Nousix architecture operates in the gap between where Moser-Tardos guarantees
polynomial convergence and where the phase transition makes everything hard.
In this gap, the retry loop still converges (it is not divergent), but the
expected number of retries is exponential in n.

The Kolipaka-Szegedy (2011) entropy compression analysis shows that even beyond
the LLL boundary, the Moser-Tardos structure provides a specific exponential
convergence rate characterized by alpha -- rather than an arbitrary "it's just
hard."

---

## 4. The Information-Theoretic Floor

Even with a perfect oracle that extracts maximum information from each retry,
there is a lower bound on the number of retries required:

```
minimum retries >= n / log2(m)
```

where n = number of variables (bits of information needed to specify an
assignment) and m = number of clauses (the violation feedback identifies which
of m clauses failed, providing at most log2(m) bits per retry).

For 3-SAT with m = 4.267n:

| n | m | bits/retry | min retries |
|---|---|-----------|-------------|
| 10 | 42 | 5.4 | 1.8 |
| 20 | 85 | 6.4 | 3.1 |
| 50 | 213 | 7.7 | 6.5 |
| 100 | 426 | 8.7 | 11.5 |
| 200 | 853 | 9.7 | 20.6 |
| 500 | 2133 | 11.1 | 45.2 |
| 1000 | 4267 | 12.1 | 82.9 |

This floor scales as n/log(n) -- **sub-exponential but super-polynomial**. It
represents the theoretical ideal that no system (LLM-based or otherwise) can
beat with a generate-verify-retry architecture. The actual LLM is far less
efficient at extracting information from feedback, so practical performance is
much worse than this floor.

The n/log(n) floor is important because it establishes that the retry loop
architecture has a fundamental limit that sits between polynomial and
exponential -- it cannot be polynomial no matter how good the LLM gets, but it
is also not inherently required to be fully exponential.

---

## 5. The e-Governed Transition

The constant e = 2.71828... appears in the system in three mathematically
distinct ways:

**(a) As the base of the natural exponential in success probability:**
```
p(n) = exp(-alpha * n)
```
This is a change of base from b^n. alpha = ln(b).

**(b) As the transition threshold in the Gompertz formula:**
```
P(success | B tries) = 1 - exp(-B * exp(-alpha * n))
```
At n* = ln(B)/alpha, the success probability = 1 - 1/e ~ 0.632. This is the
natural boundary between "solvable within budget" and "budget-exceeded."

**(c) As the limit of marginal retry value:**
The compound growth (1 + 1/k)^k -> e describes the diminishing marginal value
of additional retries. After sufficient retries, each additional attempt adds
negligible probability of success -- the system hits e-governed diminishing
returns.

The e-governed transition is the operating specification for a Nousix deployment:
given a budget B and a target problem size n, the system either operates below
the transition threshold (high confidence of success) or above it (probabilistic
failure with well-characterized probability).

---

## 6. Why "LLM as Oracle" Alone Is Naive

A bare LLM call to solve SAT is a single stochastic sample from an uncalibrated
distribution over assignment space. It has the following deficiencies:

1. **No verification**: The LLM may hallucinate a "satisfying" assignment that
   does not actually satisfy the formula. Without deterministic verification,
   you cannot distinguish a correct answer from a confident wrong one.

2. **No feedback**: A single call provides no information about *why* the guess
   was wrong. There is no signal to guide improvement.

3. **No memory**: Each call is independent. The LLM cannot learn from its
   previous failures on this specific instance.

4. **No strategy diversity**: The LLM samples from a fixed distribution
   determined by its weights and temperature. There is no mechanism to explore
   different regions of assignment space systematically.

5. **No convergence guarantee**: With a single call, the probability of success
   on an n-variable formula is some unknown p(n), and there is no framework for
   improving it.

Nousix transforms this into a structured convergence process:

| Component | What It Provides | Mathematical Effect |
|-----------|-----------------|-------------------|
| Coprocessor verification | Zero-error correctness | Eliminates false positives |
| Clause violation diagnosis | Gradient-like feedback | Reduces alpha (improves base) |
| Context window history | Attempt memory | Prevents redundant exploration |
| Retry framework | Multiple independent draws | Gompertz convergence curve |
| Strategy rotation | Distribution diversity | Prevents mode collapse |
| Temperature escalation | Exploration control | Annealing-like coverage |
| Budget constraint | Bounded computation | Characterizable failure probability |

The difference is structural, not cosmetic. A single LLM call is a point sample
from a stochastic process. Nousix is a Las Vegas algorithm -- a
randomized algorithm that always produces a correct answer when it terminates,
with a well-characterized probability of termination within any given budget.

---

## 7. Implications for the P vs NP Demonstration

The LLM Oracle solver is the seventh solver in the P vs NP exploration suite.
Its failure mode is qualitatively different from the other six:

| Solver | What Grows Exponentially | Where It Manifests |
|--------|-------------------------|-------------------|
| BruteForce | Search space (2^n) | Enumeration time |
| DPLL | Backtracking tree depth | Decision/backtrack count |
| Algebraic | Polynomial degree (Groebner) | Memory (EXPSPACE) |
| Spectral | Cross-partition resolution | DPLL fallback invocations |
| LP Relaxation | Integrality gap | Rounding success probability |
| Structural | Treewidth / backdoor size | No structural shortcut |
| **LLM Oracle** | **Correct-assignment probability** | **Required LLM calls** |

The LLM Oracle's wall is information-theoretic rather than computational. It
most directly embodies the P vs NP verification asymmetry: verification is
O(n * m) (trivially polynomial), but generation of correct candidates requires
exponentially many attempts. You can literally watch the success rate drop as n
increases -- making the abstract P/NP gap tangible and measurable.

Nousix does not break through this wall. It characterizes the wall
with mathematical precision (the alpha constant, the Gompertz curve, the
e-governed transition) and provides the engineering framework to operate
productively within the wall's constraints. That is the value proposition:
not claiming to solve the unsolvable, but providing a rigorous system for
extracting maximum value from probabilistic inference within mathematically
understood limits.
