"""Structural approach: tractable subclasses, treewidth, and backdoor sets.

This is the most theoretically grounded experimental approach because it
identifies and exploits *proven* polynomial-time substructure within SAT.
Rather than hoping a general heuristic achieves polynomial time on all
inputs, this approach precisely delineates the P/NP boundary:

- **2-SAT** is in P: solvable in O(n + m) via implication graph + SCC.
- **Horn-SAT** is in P: solvable in O(n * m) via unit propagation.
- **Bounded treewidth**: SAT on formulas with treewidth k is solvable
  in O(2^k * n), which is polynomial when k is bounded.
- **Backdoor sets**: If a small set of variables exists such that every
  assignment to those variables leaves a tractable sub-formula, then
  SAT is solvable in O(2^|backdoor| * poly(n)).

The sharp transition: 2-SAT is in P, but 3-SAT is NP-complete. Adding
a single literal per clause crosses the complexity boundary. Random
3-SAT at the phase transition (ratio ~4.267) has large treewidth and
no small backdoor sets, which is precisely why general SAT remains hard.

Complexity claims:
    - 2-SAT: O(n + m) (proven, Aspvall-Plass-Tarjan 1979)
    - Horn-SAT: O(n * m) (proven, Dowling-Gallier 1984)
    - Bounded treewidth k: O(2^k * n) (FPT, Courcelle's theorem)
    - General 3-SAT: No structural shortcut found (expected)

Example:
    >>> from p_equals_np.sat_types import Variable, Literal, Clause, CNFFormula
    >>> # A simple 2-SAT instance: (x1 OR x2) AND (~x1 OR x2)
    >>> x1, x2 = Variable(1), Variable(2)
    >>> f = CNFFormula((
    ...     Clause((Literal(x1), Literal(x2))),
    ...     Clause((Literal(x1, False), Literal(x2))),
    ... ))
    >>> solver = StructuralSolver()
    >>> result = solver.solve(f)
    >>> result is not None
    True
"""

from __future__ import annotations

import itertools
from typing import Optional

from p_equals_np.sat_types import CNFFormula


# ---------------------------------------------------------------------------
# StructuralSolver
# ---------------------------------------------------------------------------


class StructuralSolver:
    """SAT solver exploiting tractable subclasses and structural properties.

    Implements the Solver protocol from ``p_equals_np.definitions``.
    Identifies and dispatches to polynomial-time algorithms for known
    tractable subclasses (2-SAT, Horn-SAT), and attempts structural
    shortcuts (bounded treewidth, small backdoor sets) for general instances.

    Attributes:
        detected_class: After solving, the structural class detected
            (e.g., "2-SAT", "Horn-SAT", "general").
        estimated_tw: After solving, the estimated treewidth upper bound.
        backdoor_found: After solving, the backdoor set found (if any).
    """

    __slots__ = ("detected_class", "estimated_tw", "backdoor_found")

    def __init__(self) -> None:
        """Initialize the structural solver."""
        self.detected_class: str = "unknown"
        self.estimated_tw: int = -1
        self.backdoor_found: Optional[set[int]] = None

    def solve(self, formula: CNFFormula) -> Optional[dict[int, bool]]:
        """Solve SAT by detecting and exploiting structural properties.

        Dispatches to the fastest applicable algorithm:
        1. If 2-SAT: solve via implication graph + SCC in O(n + m).
        2. If Horn-SAT: solve via unit propagation in O(n * m).
        3. Estimate treewidth and look for small backdoor sets.
        4. If backdoor found: enumerate backdoor assignments.
        5. Otherwise: return None (no structural shortcut).

        Args:
            formula: A CNF formula to solve.

        Returns:
            A satisfying assignment if found, or None if unsatisfiable
            or no structural shortcut applies.
        """
        int_clauses = _formula_to_int_clauses(formula)
        variables = _collect_variables(int_clauses)

        if detect_2sat(int_clauses):
            self.detected_class = "2-SAT"
            return solve_2sat(int_clauses, variables)

        if detect_horn_sat(int_clauses):
            self.detected_class = "Horn-SAT"
            return solve_horn_sat(int_clauses, variables)

        self.estimated_tw = estimate_treewidth(int_clauses)
        backdoors = find_backdoor_candidates(int_clauses, variables)
        if backdoors:
            self.backdoor_found = backdoors[0]
            self.detected_class = f"backdoor(size={len(backdoors[0])})"
            return _solve_via_backdoor(int_clauses, variables, backdoors[0])

        self.detected_class = "general"
        return None

    def name(self) -> str:
        """Return the human-readable solver name.

        Returns:
            A string identifying this solver and its strategy.
        """
        return "Structural (Treewidth + Backdoors + Tractable Subclasses)"

    def complexity_claim(self) -> str:
        """State the claimed complexity for each structural class.

        Returns:
            A string describing complexity per detected class.
        """
        return "O(n+m) for 2-SAT/Horn-SAT, O(2^k*n) for treewidth k"


# ---------------------------------------------------------------------------
# Subclass detection
# ---------------------------------------------------------------------------


def detect_2sat(clauses: list[frozenset[int]]) -> bool:
    """Detect whether a formula is a 2-SAT instance.

    A formula is 2-SAT if every clause has at most 2 literals.
    2-SAT is in P (Aspvall-Plass-Tarjan, 1979).

    Args:
        clauses: Formula in integer-literal representation.

    Returns:
        True if every clause has at most 2 literals.
    """
    return all(len(clause) <= 2 for clause in clauses)


def detect_horn_sat(clauses: list[frozenset[int]]) -> bool:
    """Detect whether a formula is a Horn-SAT instance.

    A formula is Horn if every clause has at most one positive literal.
    Horn-SAT is in P (Dowling-Gallier, 1984). In our integer-literal
    convention, positive literals are represented by positive integers.

    Args:
        clauses: Formula in integer-literal representation.

    Returns:
        True if every clause has at most one positive literal.
    """
    for clause in clauses:
        positive_count = sum(1 for lit in clause if lit > 0)
        if positive_count > 1:
            return False
    return True


# ---------------------------------------------------------------------------
# 2-SAT solver: Implication graph + Tarjan's SCC
# ---------------------------------------------------------------------------


def solve_2sat(
    clauses: list[frozenset[int]],
    variables: frozenset[int],
) -> Optional[dict[int, bool]]:
    """Solve a 2-SAT formula using implication graph and SCC decomposition.

    Algorithm (Aspvall-Plass-Tarjan, 1979):
    1. Build implication graph: clause (a OR b) becomes edges ~a->b and ~b->a.
    2. Find all strongly connected components using Tarjan's algorithm.
    3. If any variable x and ~x are in the same SCC, the formula is UNSAT.
    4. Otherwise, assign variables based on topological order of SCCs:
       a variable is True if its positive literal's SCC comes after its
       negative literal's SCC in reverse topological order.

    Complexity: O(n + m) where n = variables, m = clauses.

    Args:
        clauses: Formula in integer-literal representation (each clause
            must have at most 2 literals).
        variables: Set of all variable indices.

    Returns:
        A satisfying assignment, or None if unsatisfiable.
    """
    # An empty clause is trivially unsatisfiable
    if any(len(c) == 0 for c in clauses):
        return None

    graph, all_nodes = _build_implication_graph(clauses, variables)
    scc_id = _tarjan_scc(graph, all_nodes)

    # Check satisfiability: x and ~x must be in different SCCs
    for var in variables:
        if scc_id.get(var, -1) == scc_id.get(-var, -2):
            return None

    # Assign based on topological order of SCCs.
    # Tarjan's completes SCCs in reverse topological order: SCC 0 is a
    # sink (last in topological order), higher IDs are closer to sources.
    # A variable x is True if x's SCC is closer to sinks (lower ID)
    # than ~x's SCC, meaning x comes AFTER ~x in topological order.
    assignment: dict[int, bool] = {}
    for var in variables:
        pos_scc = scc_id.get(var, -1)
        neg_scc = scc_id.get(-var, -2)
        assignment[var] = pos_scc < neg_scc

    return assignment


def _build_implication_graph(
    clauses: list[frozenset[int]],
    variables: frozenset[int],
) -> tuple[dict[int, list[int]], set[int]]:
    """Build the implication graph for a 2-SAT formula.

    For clause (a OR b), add edges: ~a -> b and ~b -> a.
    For unit clause (a), add edge: ~a -> a.

    Args:
        clauses: 2-SAT clauses in integer-literal form.
        variables: All variable indices.

    Returns:
        A tuple of (adjacency_list, all_node_set).
    """
    all_nodes: set[int] = set()
    for var in variables:
        all_nodes.add(var)
        all_nodes.add(-var)

    graph: dict[int, list[int]] = {node: [] for node in all_nodes}

    for clause in clauses:
        lits = list(clause)
        if len(lits) == 0:
            continue
        if len(lits) == 1:
            a = lits[0]
            graph[-a].append(a)
        else:
            a, b = lits[0], lits[1]
            graph[-a].append(b)
            graph[-b].append(a)

    return graph, all_nodes


def _tarjan_scc(
    graph: dict[int, list[int]],
    all_nodes: set[int],
) -> dict[int, int]:
    """Find strongly connected components using Tarjan's algorithm.

    Uses an iterative implementation to avoid Python recursion limits
    on large graphs. Returns a mapping from each node to its SCC id.
    SCCs are numbered in the order they are completed (reverse topological).

    Args:
        graph: Adjacency list representation.
        all_nodes: Set of all nodes in the graph.

    Returns:
        A dict mapping each node to its SCC identifier.
    """
    index_counter = [0]
    node_index: dict[int, int] = {}
    node_lowlink: dict[int, int] = {}
    on_stack: dict[int, bool] = {}
    stack: list[int] = []
    scc_id: dict[int, int] = {}
    scc_counter = [0]

    for node in all_nodes:
        if node not in node_index:
            _tarjan_iterative(
                node, graph, index_counter, node_index,
                node_lowlink, on_stack, stack, scc_id, scc_counter,
            )

    return scc_id


def _tarjan_iterative(
    start: int,
    graph: dict[int, list[int]],
    index_counter: list[int],
    node_index: dict[int, int],
    node_lowlink: dict[int, int],
    on_stack: dict[int, bool],
    stack: list[int],
    scc_id: dict[int, int],
    scc_counter: list[int],
) -> None:
    """Iterative Tarjan's SCC for one connected component.

    Uses an explicit call stack to avoid Python recursion depth limits.
    Each frame on the call stack tracks the current node and iterator
    position over its neighbors.

    Args:
        start: Starting node for this traversal.
        graph: Adjacency list.
        index_counter: Mutable counter for DFS discovery order.
        node_index: Discovery index per node.
        node_lowlink: Low-link value per node.
        on_stack: Whether each node is currently on the SCC stack.
        stack: The SCC stack.
        scc_id: Output mapping from node to SCC id.
        scc_counter: Mutable counter for SCC numbering.
    """
    # Call stack: (node, neighbor_iterator, returned_from_child)
    call_stack: list[tuple[int, int]] = []

    # Initialize start node
    node_index[start] = index_counter[0]
    node_lowlink[start] = index_counter[0]
    index_counter[0] += 1
    on_stack[start] = True
    stack.append(start)
    call_stack.append((start, 0))

    while call_stack:
        node, neighbor_idx = call_stack[-1]
        neighbors = graph.get(node, [])

        if neighbor_idx < len(neighbors):
            # Advance to next neighbor
            call_stack[-1] = (node, neighbor_idx + 1)
            w = neighbors[neighbor_idx]

            if w not in node_index:
                # Unvisited: push new frame
                node_index[w] = index_counter[0]
                node_lowlink[w] = index_counter[0]
                index_counter[0] += 1
                on_stack[w] = True
                stack.append(w)
                call_stack.append((w, 0))
            elif on_stack.get(w, False):
                # On stack: update lowlink using index (textbook Tarjan's)
                if node_index[w] < node_lowlink[node]:
                    node_lowlink[node] = node_index[w]
        else:
            # All neighbors processed: check for SCC root
            if node_lowlink[node] == node_index[node]:
                current_scc = scc_counter[0]
                scc_counter[0] += 1
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc_id[w] = current_scc
                    if w == node:
                        break

            # Pop this frame and update parent's lowlink
            call_stack.pop()
            if call_stack:
                parent = call_stack[-1][0]
                if node_lowlink[node] < node_lowlink[parent]:
                    node_lowlink[parent] = node_lowlink[node]


# ---------------------------------------------------------------------------
# Horn-SAT solver: unit propagation
# ---------------------------------------------------------------------------


def solve_horn_sat(
    clauses: list[frozenset[int]],
    variables: frozenset[int],
) -> Optional[dict[int, bool]]:
    """Solve a Horn-SAT formula via iterative unit propagation.

    Horn clauses have at most one positive literal. The algorithm:
    1. Initialize all variables to False.
    2. Find unit clauses (single positive literal) and set them True.
    3. Simplify: remove satisfied clauses, shorten others.
    4. Repeat until fixed point.
    5. If an empty clause exists, return UNSAT.

    Complexity: O(n * m) where n = variables, m = clauses.

    Args:
        clauses: Formula in integer-literal representation (each clause
            must have at most one positive literal).
        variables: Set of all variable indices.

    Returns:
        A satisfying assignment, or None if unsatisfiable.
    """
    assignment: dict[int, bool] = {var: False for var in variables}
    working = list(clauses)

    changed = True
    while changed:
        changed = False

        # Find unit clauses with a single positive literal
        for clause in working:
            if len(clause) == 1:
                lit = next(iter(clause))
                var = abs(lit)
                value = lit > 0
                if assignment[var] != value:
                    assignment[var] = value
                    changed = True

        # Simplify clauses
        working = _simplify_all(working, assignment)

        # Check for empty clause (conflict)
        if any(len(c) == 0 for c in working):
            return None

        if not working:
            break

    # Verify remaining clauses are satisfied
    if not _verify_assignment(clauses, assignment):
        return None

    return assignment


def _simplify_all(
    clauses: list[frozenset[int]],
    assignment: dict[int, bool],
) -> list[frozenset[int]]:
    """Remove satisfied clauses and falsified literals.

    Args:
        clauses: Current clause set.
        assignment: Current variable assignments.

    Returns:
        Simplified clause list.
    """
    result: list[frozenset[int]] = []
    for clause in clauses:
        satisfied = False
        remaining: set[int] = set()
        for lit in clause:
            var = abs(lit)
            val = assignment.get(var)
            if val is not None:
                lit_true = (lit > 0 and val) or (lit < 0 and not val)
                if lit_true:
                    satisfied = True
                    break
            else:
                remaining.add(lit)
        if not satisfied:
            result.append(frozenset(remaining))
    return result


def _verify_assignment(
    clauses: list[frozenset[int]],
    assignment: dict[int, bool],
) -> bool:
    """Check that an assignment satisfies all clauses.

    Args:
        clauses: The original clause set.
        assignment: Complete variable assignment.

    Returns:
        True if every clause is satisfied.
    """
    for clause in clauses:
        satisfied = False
        for lit in clause:
            var = abs(lit)
            val = assignment.get(var, False)
            if (lit > 0 and val) or (lit < 0 and not val):
                satisfied = True
                break
        if not satisfied:
            return False
    return True


# ---------------------------------------------------------------------------
# Variable Interaction Graph and Treewidth
# ---------------------------------------------------------------------------


def compute_vig(clauses: list[frozenset[int]]) -> dict[int, set[int]]:
    """Compute the Variable Interaction Graph (VIG) as an adjacency list.

    In the VIG, two variables share an edge if they appear together in
    at least one clause. The VIG captures the constraint structure of
    the formula and is used for treewidth estimation.

    Args:
        clauses: Formula in integer-literal representation.

    Returns:
        Adjacency list mapping each variable to its neighbors.
    """
    adjacency: dict[int, set[int]] = {}

    for clause in clauses:
        vars_in_clause = [abs(lit) for lit in clause]
        for var in vars_in_clause:
            if var not in adjacency:
                adjacency[var] = set()

        for i, v1 in enumerate(vars_in_clause):
            for v2 in vars_in_clause[i + 1:]:
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)

    return adjacency


def estimate_treewidth(clauses: list[frozenset[int]]) -> int:
    """Estimate treewidth upper bound using greedy min-degree heuristic.

    Treewidth measures how "tree-like" the variable interaction graph is.
    Formulas with bounded treewidth k can be solved in O(2^k * n) time.

    The min-degree heuristic iteratively eliminates the vertex with fewest
    neighbors, adding edges between all its neighbors (fill-in), and tracks
    the maximum degree at elimination. This gives an upper bound on the
    true treewidth.

    Complexity: O(n^2) where n = number of variables (polynomial).

    Args:
        clauses: Formula in integer-literal representation.

    Returns:
        An upper bound on the treewidth (non-negative integer).
        Returns 0 for formulas with no variables.
    """
    vig = compute_vig(clauses)
    if not vig:
        return 0

    # Work on a mutable copy
    adj: dict[int, set[int]] = {v: set(neighbors) for v, neighbors in vig.items()}
    remaining = set(adj.keys())
    max_degree_at_elimination = 0

    while remaining:
        # Find vertex with minimum degree among remaining
        min_var = min(remaining, key=lambda v: len(adj[v] & remaining))
        neighbors = adj[min_var] & remaining
        degree = len(neighbors)

        if degree > max_degree_at_elimination:
            max_degree_at_elimination = degree

        # Add fill-in edges between all pairs of neighbors
        neighbor_list = list(neighbors)
        for i, n1 in enumerate(neighbor_list):
            for n2 in neighbor_list[i + 1:]:
                adj[n1].add(n2)
                adj[n2].add(n1)

        remaining.remove(min_var)

    return max_degree_at_elimination


# ---------------------------------------------------------------------------
# Backdoor Sets
# ---------------------------------------------------------------------------


def find_backdoor_candidates(
    clauses: list[frozenset[int]],
    variables: frozenset[int],
    max_size: int = 5,
) -> list[set[int]]:
    """Find small backdoor sets to tractable subclasses.

    A backdoor set S is a set of variables such that for EVERY truth
    assignment to S, the resulting simplified formula is in a tractable
    class (2-SAT or Horn-SAT). If such a set of size k exists, SAT can
    be solved in O(2^k * poly(n)) time.

    This function searches for backdoor sets by exhaustive enumeration
    of subsets up to max_size. The search is polynomial for fixed max_size
    but becomes expensive as max_size grows.

    Args:
        clauses: Formula in integer-literal representation.
        variables: Set of all variable indices.
        max_size: Maximum backdoor set size to search for (default 5).

    Returns:
        A list of backdoor sets found (may be empty). Each set contains
        variable indices. The list is sorted by set size (smallest first).
    """
    var_list = sorted(variables)
    found: list[set[int]] = []

    for size in range(1, min(max_size + 1, len(var_list) + 1)):
        for subset in itertools.combinations(var_list, size):
            backdoor = set(subset)
            if _is_backdoor(clauses, backdoor):
                found.append(backdoor)
                # Return first found at each size (optimization)
                return found

    return found


def _is_backdoor(
    clauses: list[frozenset[int]],
    backdoor: set[int],
) -> bool:
    """Check if a variable set is a backdoor to tractable subclasses.

    Tests all 2^|backdoor| assignments to the backdoor variables. For each
    assignment, simplifies the formula and checks if the result is 2-SAT
    or Horn-SAT.

    Args:
        clauses: Formula in integer-literal representation.
        backdoor: Set of variable indices to test.

    Returns:
        True if every assignment to backdoor yields a tractable formula.
    """
    backdoor_list = sorted(backdoor)

    for values in itertools.product((False, True), repeat=len(backdoor_list)):
        simplified = _simplify_for_assignment(
            clauses, dict(zip(backdoor_list, values))
        )

        # Check if simplified formula is tractable
        if not detect_2sat(simplified) and not detect_horn_sat(simplified):
            return False

    return True


def _simplify_for_assignment(
    clauses: list[frozenset[int]],
    assignment: dict[int, bool],
) -> list[frozenset[int]]:
    """Simplify clauses given a partial assignment.

    Removes satisfied clauses and falsified literals, matching the DPLL
    simplification semantics.

    Args:
        clauses: Clause set.
        assignment: Partial variable assignment.

    Returns:
        Simplified clause list.
    """
    result: list[frozenset[int]] = []
    for clause in clauses:
        satisfied = False
        remaining: set[int] = set()
        for lit in clause:
            var = abs(lit)
            if var in assignment:
                lit_true = (lit > 0) == assignment[var]
                if lit_true:
                    satisfied = True
                    break
            else:
                remaining.add(lit)
        if not satisfied:
            result.append(frozenset(remaining))
    return result


def _solve_via_backdoor(
    clauses: list[frozenset[int]],
    variables: frozenset[int],
    backdoor: set[int],
) -> Optional[dict[int, bool]]:
    """Solve SAT by enumerating assignments to a backdoor set.

    For each of the 2^|backdoor| assignments, simplifies the formula
    and solves the resulting tractable sub-formula.

    Args:
        clauses: Formula in integer-literal representation.
        variables: All variable indices.
        backdoor: Set of backdoor variable indices.

    Returns:
        A satisfying assignment, or None if unsatisfiable.
    """
    backdoor_list = sorted(backdoor)
    remaining_vars = variables - backdoor

    for values in itertools.product((False, True), repeat=len(backdoor_list)):
        partial = dict(zip(backdoor_list, values))
        simplified = _simplify_for_assignment(clauses, partial)

        # Try tractable solvers on the simplified formula
        sub_vars = _collect_variables(simplified) & remaining_vars
        result = _solve_tractable(simplified, sub_vars)

        if result is not None:
            # Merge backdoor assignment with sub-formula solution
            full = dict(partial)
            full.update(result)
            # Fill in any unassigned variables
            for var in variables:
                if var not in full:
                    full[var] = False
            return full

    return None


def _solve_tractable(
    clauses: list[frozenset[int]],
    variables: frozenset[int],
) -> Optional[dict[int, bool]]:
    """Attempt to solve a formula using tractable sub-solvers.

    Tries 2-SAT solver first, then Horn-SAT solver. Returns None if
    neither applies or if the formula is unsatisfiable.

    Args:
        clauses: Simplified clause set.
        variables: Variables remaining in the formula.

    Returns:
        A satisfying assignment, or None.
    """
    if not clauses:
        return {var: False for var in variables}

    if detect_2sat(clauses):
        return solve_2sat(clauses, variables)

    if detect_horn_sat(clauses):
        return solve_horn_sat(clauses, variables)

    return None


# ---------------------------------------------------------------------------
# Internal conversion helpers (matching dpll.py conventions)
# ---------------------------------------------------------------------------


def _formula_to_int_clauses(formula: CNFFormula) -> list[frozenset[int]]:
    """Convert a CNFFormula to integer-literal clause representation.

    Positive literal for variable i -> +i, negated literal -> -i.

    Args:
        formula: A CNFFormula with Literal/Clause objects.

    Returns:
        A list of frozensets of int literals.
    """
    result: list[frozenset[int]] = []
    for clause in formula.clauses:
        int_lits: set[int] = set()
        for literal in clause.literals:
            idx = literal.variable.index
            int_lits.add(idx if literal.positive else -idx)
        result.append(frozenset(int_lits))
    return result


def _collect_variables(clauses: list[frozenset[int]]) -> frozenset[int]:
    """Collect all variable indices from a clause set.

    Args:
        clauses: List of int-literal frozensets.

    Returns:
        A frozenset of variable indices (always positive).
    """
    variables: set[int] = set()
    for clause in clauses:
        for literal in clause:
            variables.add(abs(literal))
    return frozenset(variables)
