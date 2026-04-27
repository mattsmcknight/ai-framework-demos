"""Spectral approach to SAT via Variable Interaction Graph eigenvalues.

Theoretical basis: Build a Variable Interaction Graph (VIG) from a CNF
formula where vertices are variables and edges connect variables that
co-occur in a clause. The spectral properties of the VIG's Laplacian
matrix (eigenvalues, Fiedler vector) are used to partition variables
and guide a divide-and-conquer solving strategy.

Key concepts:
    - **Variable Interaction Graph (VIG)**: Undirected graph where each
      variable is a vertex and an edge connects two variables if they
      co-occur in at least one clause. Edge weights count co-occurrences.
    - **Laplacian matrix**: L = D - A, where D is the degree matrix and
      A is the adjacency matrix. The Laplacian encodes the graph structure.
    - **Algebraic connectivity**: The second-smallest eigenvalue of L
      (lambda_2). Measures how well-connected the graph is.
    - **Fiedler vector**: The eigenvector corresponding to lambda_2.
      Its sign pattern provides a natural bipartition of the graph.

Why it fails: Spectral methods characterize the *structure* of variable
interactions but satisfiability depends on literal *signs* (polarities),
which are lost in the VIG. Two formulas with identical VIGs can differ
in satisfiability. Graph properties are not satisfiability properties.
Spectral methods help guide search heuristics but cannot replace it.

Complexity: O(n^3) for eigenvalue computation + exponential for
cross-partition clause resolution. Not polynomial overall.
"""

from __future__ import annotations

import math
from typing import Optional

from p_equals_np.sat_types import CNFFormula
from p_equals_np.dpll import DPLLSolver


# ---------------------------------------------------------------------------
# SpectralSolver
# ---------------------------------------------------------------------------


class SpectralSolver:
    """SAT solver using spectral properties of the Variable Interaction Graph.

    Implements the Solver protocol from ``p_equals_np.definitions``.
    Constructs the VIG, computes its Laplacian's spectral properties,
    uses the Fiedler vector to partition variables, then applies
    divide-and-conquer with DPLL fallback on sub-problems.

    Attributes:
        timeout_seconds: Maximum wall-clock seconds for DPLL fallback.
    """

    __slots__ = ("timeout_seconds",)

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        """Initialize the spectral solver.

        Args:
            timeout_seconds: Timeout passed to the DPLL fallback solver.
        """
        self.timeout_seconds = timeout_seconds

    def solve(self, formula: CNFFormula) -> Optional[dict[int, bool]]:
        """Solve SAT using spectral partitioning and divide-and-conquer.

        Partitions variables via the Fiedler vector, solves sub-problems
        for each partition independently, then handles cross-partition
        clauses. Falls back to DPLL for sub-problems and cross-partition
        resolution.

        Args:
            formula: A CNF formula to solve.

        Returns:
            A satisfying assignment, or None if unsatisfiable.
        """
        var_indices = sorted(v.index for v in formula.get_variables())
        if not var_indices:
            return {} if formula.evaluate({}) else None

        if len(var_indices) <= 3:
            return self._dpll_fallback(formula)

        partition_a, partition_b = spectral_partition(formula)
        if not partition_a or not partition_b:
            return self._dpll_fallback(formula)

        return self._divide_and_conquer(
            formula, var_indices, partition_a, partition_b
        )

    def name(self) -> str:
        """Return the human-readable solver name.

        Returns:
            The string identifying this solver.
        """
        return "Spectral (VIG Eigenvalues)"

    def complexity_claim(self) -> str:
        """State the claimed time complexity of this solver.

        Returns:
            A string describing the complexity.
        """
        return "O(n^3) for eigenvalues + exponential for cross-partition"

    # --- Private helpers ---

    def _dpll_fallback(self, formula: CNFFormula) -> Optional[dict[int, bool]]:
        """Solve a formula directly using DPLL as fallback.

        Args:
            formula: A CNF formula.

        Returns:
            A satisfying assignment or None.
        """
        solver = DPLLSolver(timeout_seconds=self.timeout_seconds)
        return solver.solve(formula)

    def _divide_and_conquer(
        self,
        formula: CNFFormula,
        var_indices: list[int],
        partition_a: set[int],
        partition_b: set[int],
    ) -> Optional[dict[int, bool]]:
        """Solve by partitioning variables and combining sub-solutions.

        Classifies clauses into partition-local and cross-partition.
        Solves local clauses for each partition, then checks whether
        the combined assignment satisfies cross-partition clauses.
        Falls back to full DPLL if cross-partition resolution fails.

        Args:
            formula: The original CNF formula.
            var_indices: All variable indices in the formula.
            partition_a: Variable indices in partition A.
            partition_b: Variable indices in partition B.

        Returns:
            A satisfying assignment or None.
        """
        clauses_a, clauses_b, cross_clauses = _classify_clauses(
            formula, partition_a, partition_b
        )

        solution_a = self._solve_subproblem(clauses_a, partition_a)
        if solution_a is None and clauses_a:
            return self._dpll_fallback(formula)

        solution_b = self._solve_subproblem(clauses_b, partition_b)
        if solution_b is None and clauses_b:
            return self._dpll_fallback(formula)

        combined = {}
        if solution_a is not None:
            combined.update(solution_a)
        if solution_b is not None:
            combined.update(solution_b)

        for v in var_indices:
            if v not in combined:
                combined[v] = False

        if formula.evaluate(combined):
            return combined

        # Cross-partition clauses invalidate combined solution; fall back.
        return self._dpll_fallback(formula)

    def _solve_subproblem(
        self,
        clauses: list[tuple[tuple[int, bool], ...]],
        partition_vars: set[int],
    ) -> Optional[dict[int, bool]]:
        """Solve the sub-problem restricted to one partition's clauses.

        Args:
            clauses: Clause tuples of (var_index, positive) pairs.
            partition_vars: Variable indices in this partition.

        Returns:
            A partial assignment, or None if unsatisfiable.
        """
        if not clauses:
            return {v: False for v in partition_vars}

        sub_formula = _build_subformula(clauses)
        return self._dpll_fallback(sub_formula)


# ---------------------------------------------------------------------------
# VIG construction
# ---------------------------------------------------------------------------


def formula_to_vig(formula: CNFFormula) -> list[list[float]]:
    """Build the Variable Interaction Graph adjacency matrix from a CNF formula.

    Creates an undirected weighted graph where vertices are variables
    (indexed 0..n-1 corresponding to variable indices sorted ascending)
    and edge weights count co-occurrences in clauses.

    Args:
        formula: A CNF formula.

    Returns:
        An n x n adjacency matrix where entry [i][j] counts the
        number of clauses in which variables i and j co-occur.
    """
    var_indices = sorted(v.index for v in formula.get_variables())
    n = len(var_indices)
    if n == 0:
        return []

    index_map = {v: i for i, v in enumerate(var_indices)}
    adjacency = [[0.0] * n for _ in range(n)]

    for clause in formula.clauses:
        clause_vars = [lit.variable.index for lit in clause.literals]
        for a_pos in range(len(clause_vars)):
            for b_pos in range(a_pos + 1, len(clause_vars)):
                i = index_map[clause_vars[a_pos]]
                j = index_map[clause_vars[b_pos]]
                adjacency[i][j] += 1.0
                adjacency[j][i] += 1.0

    return adjacency


# ---------------------------------------------------------------------------
# Laplacian computation
# ---------------------------------------------------------------------------


def compute_laplacian(adjacency: list[list[float]]) -> list[list[float]]:
    """Compute the Laplacian matrix L = D - A from an adjacency matrix.

    The Laplacian is symmetric positive semi-definite with smallest
    eigenvalue 0 (eigenvector: all-ones for connected components).

    Args:
        adjacency: An n x n symmetric adjacency matrix.

    Returns:
        The n x n Laplacian matrix.
    """
    n = len(adjacency)
    laplacian = [[0.0] * n for _ in range(n)]

    for i in range(n):
        degree = sum(adjacency[i])
        for j in range(n):
            if i == j:
                laplacian[i][j] = degree
            else:
                laplacian[i][j] = -adjacency[i][j]

    return laplacian


# ---------------------------------------------------------------------------
# Eigenvalue computation (symmetric tridiagonal QL with implicit shifts)
# ---------------------------------------------------------------------------


def eigenvalues(matrix: list[list[float]]) -> list[float]:
    """Compute all eigenvalues of a symmetric matrix.

    Reduces to tridiagonal form via Householder reflections, then
    applies the QL algorithm with implicit Wilkinson shifts. O(n^3).

    Args:
        matrix: An n x n symmetric matrix.

    Returns:
        A sorted list of eigenvalues (ascending).

    Raises:
        ValueError: If the matrix is empty or not square.
    """
    n = len(matrix)
    if n == 0:
        raise ValueError("Cannot compute eigenvalues of empty matrix")
    for row in matrix:
        if len(row) != n:
            raise ValueError("Matrix must be square")

    if n == 1:
        return [matrix[0][0]]

    if n == 2:
        return _eigenvalues_2x2(matrix)

    diag, off = _householder_tridiagonalize(matrix)
    _tql2_eigenvalues(diag, off)
    return sorted(diag)


def _eigenvalues_2x2(matrix: list[list[float]]) -> list[float]:
    """Compute eigenvalues of a 2x2 symmetric matrix analytically.

    Args:
        matrix: A 2x2 symmetric matrix.

    Returns:
        Sorted list of two eigenvalues.
    """
    a = matrix[0][0]
    b = matrix[0][1]
    d = matrix[1][1]

    trace = a + d
    det = a * d - b * b
    disc = trace * trace - 4.0 * det
    disc = max(disc, 0.0)  # Guard against floating-point negativity
    sqrt_disc = math.sqrt(disc)

    e1 = (trace - sqrt_disc) / 2.0
    e2 = (trace + sqrt_disc) / 2.0
    return sorted([e1, e2])


def _householder_tridiagonalize(
    matrix: list[list[float]],
) -> tuple[list[float], list[float]]:
    """Reduce a symmetric matrix to tridiagonal form via Householder reflections.

    Follows the EISPACK/LAPACK convention. Produces a symmetric
    tridiagonal matrix T such that Q^T A Q = T, where Q is
    orthogonal (not explicitly formed here).

    Args:
        matrix: An n x n symmetric matrix (not modified).

    Returns:
        (diag, off) where diag has n elements and off has n elements
        (off[0] is unused, off[1..n-1] are the sub-diagonal).
    """
    n = len(matrix)
    a = [row[:] for row in matrix]
    diag = [0.0] * n
    off = [0.0] * n

    for i in range(n - 1, 0, -1):
        # Householder reduction of row/column i
        scale = sum(abs(a[i][k]) for k in range(i))

        if scale < 1e-15:
            off[i] = a[i][i - 1]
            diag[i] = 0.0
            continue

        # Scale the vector for numerical stability
        h = 0.0
        for k in range(i):
            a[i][k] /= scale
            h += a[i][k] * a[i][k]

        f = a[i][i - 1]
        g = -math.copysign(math.sqrt(h), f)
        off[i] = scale * g
        h -= f * g
        a[i][i - 1] = f - g

        # Form u / H and store in the vacated part of a[i]
        f = 0.0
        for j in range(i):
            a[j][i] = a[i][j] / h
            g_val = 0.0
            for k in range(j + 1):
                g_val += a[j][k] * a[i][k]
            for k in range(j + 1, i):
                g_val += a[k][j] * a[i][k]
            off[j] = g_val / h
            f += off[j] * a[i][j]

        hh = f / (h + h)

        for j in range(i):
            f = a[i][j]
            g_val = off[j] - hh * f
            off[j] = g_val
            for k in range(j + 1):
                a[j][k] -= f * off[k] + g_val * a[i][k]

    off[0] = 0.0
    for i in range(n):
        diag[i] = a[i][i]

    return diag, off


def _tql2_eigenvalues(diag: list[float], off: list[float]) -> None:
    """Compute eigenvalues of a symmetric tridiagonal matrix (QL algorithm).

    Modifies diag in place to contain eigenvalues. This is the
    well-known TQL2 algorithm from EISPACK, using implicit shifts
    for cubic convergence.

    Args:
        diag: Diagonal elements (modified in place to eigenvalues).
        off: Off-diagonal elements (off[0] unused, off[1..n-1] are
            sub-diagonal; destroyed during computation).
    """
    n = len(diag)

    # Shift off-diagonal indices so off[i] = sub-diagonal e[i]
    for i in range(1, n):
        off[i - 1] = off[i]
    off[n - 1] = 0.0

    for l_idx in range(n):
        iteration_count = 0

        while True:
            # Find small sub-diagonal element
            m = l_idx
            while m < n - 1:
                threshold = 1e-12 * (abs(diag[m]) + abs(diag[m + 1]))
                if abs(off[m]) <= threshold:
                    break
                m += 1

            if m == l_idx:
                break

            iteration_count += 1
            if iteration_count > 200:
                break  # Prevent infinite loop on pathological input

            # Wilkinson shift from trailing 2x2 block
            g = (diag[l_idx + 1] - diag[l_idx]) / (2.0 * off[l_idx])
            r = math.sqrt(g * g + 1.0)
            g = diag[m] - diag[l_idx] + off[l_idx] / (
                g + math.copysign(r, g)
            )

            s = 1.0
            c = 1.0
            p = 0.0

            # QL transformation from m-1 down to l_idx
            for i in range(m - 1, l_idx - 1, -1):
                f = s * off[i]
                b = c * off[i]

                if abs(f) >= abs(g):
                    c = g / f
                    r = math.sqrt(c * c + 1.0)
                    off[i + 1] = f * r
                    s = 1.0 / r
                    c = c * s
                else:
                    s = f / g
                    r = math.sqrt(s * s + 1.0)
                    off[i + 1] = g * r
                    c = 1.0 / r
                    s = s * c

                g = diag[i + 1] - p
                r = (diag[i] - g) * s + 2.0 * c * b
                p = s * r
                diag[i + 1] = g + p
                g = c * r - b

            diag[l_idx] = diag[l_idx] - p
            off[l_idx] = g
            off[m] = 0.0


# ---------------------------------------------------------------------------
# Eigenvector computation (inverse iteration)
# ---------------------------------------------------------------------------


def eigenvector(
    matrix: list[list[float]], eigenvalue: float
) -> list[float]:
    """Compute the eigenvector for a given eigenvalue via inverse iteration.

    Uses the shifted inverse iteration method: repeatedly solves
    (A - sigma*I) x = b, where sigma is close to the target eigenvalue.
    The iterates converge to the eigenvector corresponding to the
    eigenvalue nearest sigma.

    Args:
        matrix: An n x n symmetric matrix.
        eigenvalue: The eigenvalue whose eigenvector is sought.

    Returns:
        The unit eigenvector as a list of floats.

    Raises:
        ValueError: If the matrix is empty.
    """
    n = len(matrix)
    if n == 0:
        raise ValueError("Cannot compute eigenvector of empty matrix")

    # Small perturbation avoids exact singularity
    shift = eigenvalue + 1e-10
    shifted = _shifted_matrix(matrix, shift)

    # Initial vector with broken symmetry
    b = [1.0 / math.sqrt(n)] * n
    b[0] += 0.1

    for _ in range(50):
        x = _solve_system(shifted, b)
        if x is None:
            shift += 1e-8
            shifted = _shifted_matrix(matrix, shift)
            continue

        norm = _vector_norm(x)
        if norm < 1e-15:
            break
        b = [xi / norm for xi in x]

    return b


def _shifted_matrix(
    matrix: list[list[float]], shift: float
) -> list[list[float]]:
    """Create (A - shift * I).

    Args:
        matrix: An n x n matrix.
        shift: Scalar to subtract from diagonal.

    Returns:
        The shifted matrix.
    """
    n = len(matrix)
    result = [row[:] for row in matrix]
    for i in range(n):
        result[i][i] -= shift
    return result


def _solve_system(
    matrix: list[list[float]], rhs: list[float]
) -> Optional[list[float]]:
    """Solve Ax = b via Gaussian elimination with partial pivoting.

    Args:
        matrix: An n x n matrix (not modified).
        rhs: Right-hand side vector.

    Returns:
        Solution vector, or None if singular.
    """
    n = len(rhs)
    a = [row[:] for row in matrix]
    b = rhs[:]

    for col in range(n):
        max_row = col
        max_val = abs(a[col][col])
        for row in range(col + 1, n):
            if abs(a[row][col]) > max_val:
                max_val = abs(a[row][col])
                max_row = row

        if max_val < 1e-15:
            return None

        a[col], a[max_row] = a[max_row], a[col]
        b[col], b[max_row] = b[max_row], b[col]

        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for k in range(col, n):
                a[row][k] -= factor * a[col][k]
            b[row] -= factor * b[col]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(a[i][i]) < 1e-15:
            return None
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= a[i][j] * x[j]
        x[i] /= a[i][i]

    return x


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------


def spectral_features(formula: CNFFormula) -> dict[str, float]:
    """Compute spectral features of a CNF formula's Variable Interaction Graph.

    Extracts four key spectral properties:
    - algebraic_connectivity: lambda_2 of the Laplacian
    - spectral_radius: largest eigenvalue of the adjacency matrix
    - spectral_gap: lambda_2 / lambda_max (normalized connectivity)
    - eigenvalue_ratio: lambda_max / lambda_2 (inverse measure)

    Args:
        formula: A CNF formula.

    Returns:
        A dict mapping feature names to float values.
    """
    var_count = formula.num_variables
    if var_count <= 1:
        return _trivial_features()

    adjacency = formula_to_vig(formula)
    laplacian = compute_laplacian(adjacency)
    lap_eigs = eigenvalues(laplacian)

    lambda_2 = lap_eigs[1] if len(lap_eigs) > 1 else 0.0
    lambda_max = lap_eigs[-1] if lap_eigs else 0.0

    adj_eigs = eigenvalues(adjacency)
    adj_radius = max(abs(e) for e in adj_eigs) if adj_eigs else 0.0

    gap = lambda_2 / lambda_max if lambda_max > 1e-12 else 0.0
    ratio = lambda_max / lambda_2 if lambda_2 > 1e-12 else float("inf")

    return {
        "algebraic_connectivity": lambda_2,
        "spectral_radius": adj_radius,
        "spectral_gap": gap,
        "eigenvalue_ratio": ratio,
    }


def _trivial_features() -> dict[str, float]:
    """Return zero-valued spectral features for trivial formulas.

    Returns:
        Dict with all spectral features set to 0.0.
    """
    return {
        "algebraic_connectivity": 0.0,
        "spectral_radius": 0.0,
        "spectral_gap": 0.0,
        "eigenvalue_ratio": 0.0,
    }


# ---------------------------------------------------------------------------
# Spectral partition
# ---------------------------------------------------------------------------


def spectral_partition(
    formula: CNFFormula,
) -> tuple[set[int], set[int]]:
    """Partition variables using the Fiedler vector of the VIG Laplacian.

    The Fiedler vector (eigenvector of lambda_2) provides a natural
    bipartition: variables with non-negative components go to partition A,
    negative to partition B. This minimizes the edge cut between
    partitions (spectral bisection).

    Args:
        formula: A CNF formula.

    Returns:
        (partition_a, partition_b) where each is a set of variable indices.
    """
    var_indices = sorted(v.index for v in formula.get_variables())
    n = len(var_indices)

    if n <= 1:
        return set(var_indices), set()

    adjacency = formula_to_vig(formula)
    laplacian = compute_laplacian(adjacency)
    eigs = eigenvalues(laplacian)

    lambda_2 = eigs[1] if len(eigs) > 1 else 0.0

    if lambda_2 < 1e-12:
        return _partition_by_components(adjacency, var_indices)

    fiedler = eigenvector(laplacian, lambda_2)

    partition_a: set[int] = set()
    partition_b: set[int] = set()

    for i, var_idx in enumerate(var_indices):
        if i < len(fiedler) and fiedler[i] >= 0:
            partition_a.add(var_idx)
        else:
            partition_b.add(var_idx)

    # Ensure neither partition is empty
    if not partition_a and partition_b:
        moved = partition_b.pop()
        partition_a.add(moved)
    elif not partition_b and partition_a:
        moved = partition_a.pop()
        partition_b.add(moved)

    return partition_a, partition_b


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _vector_norm(v: list[float]) -> float:
    """Compute the Euclidean norm of a vector.

    Args:
        v: A list of floats.

    Returns:
        The L2 norm.
    """
    return math.sqrt(sum(x * x for x in v))


def _partition_by_components(
    adjacency: list[list[float]], var_indices: list[int]
) -> tuple[set[int], set[int]]:
    """Partition variables by connected components when graph is disconnected.

    Uses BFS to find components, assigns first component to A, rest to B.

    Args:
        adjacency: The adjacency matrix.
        var_indices: Sorted variable indices.

    Returns:
        (partition_a, partition_b).
    """
    n = len(var_indices)
    visited = [False] * n
    components: list[list[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        component: list[int] = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in range(n):
                if not visited[neighbor] and adjacency[node][neighbor] > 0:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(component)

    if len(components) <= 1:
        mid = n // 2
        return (
            {var_indices[i] for i in range(mid)},
            {var_indices[i] for i in range(mid, n)},
        )

    partition_a = {var_indices[i] for i in components[0]}
    partition_b: set[int] = set()
    for comp in components[1:]:
        partition_b.update(var_indices[i] for i in comp)

    return partition_a, partition_b


def _classify_clauses(
    formula: CNFFormula,
    partition_a: set[int],
    partition_b: set[int],
) -> tuple[
    list[tuple[tuple[int, bool], ...]],
    list[tuple[tuple[int, bool], ...]],
    list[tuple[tuple[int, bool], ...]],
]:
    """Classify clauses into partition-local and cross-partition groups.

    Args:
        formula: The CNF formula.
        partition_a: Variable indices in partition A.
        partition_b: Variable indices in partition B.

    Returns:
        (clauses_a, clauses_b, cross_clauses).
    """
    clauses_a: list[tuple[tuple[int, bool], ...]] = []
    clauses_b: list[tuple[tuple[int, bool], ...]] = []
    cross_clauses: list[tuple[tuple[int, bool], ...]] = []

    for clause in formula.clauses:
        lit_tuples = tuple(
            (lit.variable.index, lit.positive) for lit in clause.literals
        )
        var_set = {lit.variable.index for lit in clause.literals}
        in_a = bool(var_set & partition_a)
        in_b = bool(var_set & partition_b)

        if in_a and not in_b:
            clauses_a.append(lit_tuples)
        elif in_b and not in_a:
            clauses_b.append(lit_tuples)
        else:
            cross_clauses.append(lit_tuples)

    return clauses_a, clauses_b, cross_clauses


def _build_subformula(
    clauses: list[tuple[tuple[int, bool], ...]],
) -> CNFFormula:
    """Convert clause tuples back into a CNFFormula.

    Args:
        clauses: List of clauses as tuples of (var_index, positive) pairs.

    Returns:
        A CNFFormula object.
    """
    from p_equals_np.sat_types import Clause, Literal, Variable

    formula_clauses: list[Clause] = []
    for clause_data in clauses:
        literals = tuple(
            Literal(Variable(var_idx), positive=pos)
            for var_idx, pos in clause_data
        )
        formula_clauses.append(Clause(literals))

    return CNFFormula(tuple(formula_clauses))
