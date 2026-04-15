from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp


@dataclass
class MaxCutResult:
    """Container for QAOA Max-Cut results."""

    best_bitstring: str
    best_cut_value: int
    exact_bitstring: str
    exact_cut_value: int
    approximation_ratio: float
    optimal_parameters: np.ndarray
    counts: Dict[str, int]


def maxcut_value(bitstring: str, graph: nx.Graph) -> int:
    """Compute the Max-Cut objective value for a bitstring assignment.

    Args:
        bitstring: Binary string assigning each node to one of two partitions.
        graph: Undirected graph.

    Returns:
        Number of edges crossing the cut.
    """
    value = 0
    for u, v in graph.edges():
        if bitstring[u] != bitstring[v]:
            value += 1
    return value


def brute_force_maxcut(graph: nx.Graph) -> Tuple[str, int]:
    """Compute the exact Max-Cut solution by exhaustive search.

    Args:
        graph: Undirected graph.

    Returns:
        Tuple of (best_bitstring, best_cut_value).
    """
    n = graph.number_of_nodes()
    best_string = None
    best_value = -1

    for bits in itertools.product("01", repeat=n):
        bitstring = "".join(bits)
        value = maxcut_value(bitstring, graph)
        if value > best_value:
            best_value = value
            best_string = bitstring

    return best_string, best_value


def build_maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """Build the Max-Cut cost Hamiltonian as a SparsePauliOp.

    For Max-Cut, a common cost Hamiltonian is:
        C = sum_{(i,j) in E} 0.5 * (I - Z_i Z_j)

    Since constant identity shifts do not affect optimization over bitstring
    probabilities for comparison purposes, we include the full operator here.

    Args:
        graph: Undirected graph.

    Returns:
        SparsePauliOp representing the Max-Cut cost Hamiltonian.
    """
    n = graph.number_of_nodes()
    paulis: List[Tuple[str, float]] = []

    # Constant term: 0.5 * |E| * I
    paulis.append(("I" * n, 0.5 * graph.number_of_edges()))

    # Interaction terms: -0.5 * Z_i Z_j
    for u, v in graph.edges():
        label = ["I"] * n
        label[n - 1 - u] = "Z"
        label[n - 1 - v] = "Z"
        paulis.append(("".join(label), -0.5))

    return SparsePauliOp.from_list(paulis)


def expectation_value(
    params: np.ndarray,
    ansatz: QAOAAnsatz,
    cost_hamiltonian: SparsePauliOp,
    estimator: StatevectorEstimator,
) -> float:
    """Evaluate the negative expected cost for classical minimization.

    Args:
        params: QAOA parameter vector.
        ansatz: Parameterized QAOA circuit.
        cost_hamiltonian: Max-Cut cost Hamiltonian.
        estimator: Estimator primitive.

    Returns:
        Negative expectation value of the cost Hamiltonian.
    """
    pub = (ansatz, [cost_hamiltonian], [params])
    result = estimator.run([pub]).result()
    energy = result[0].data.evs[0]
    return -float(energy)


def sample_bitstrings(
    ansatz: QAOAAnsatz,
    params: np.ndarray,
    shots: int = 2048,
) -> Dict[str, int]:
    """Sample measurement outcomes from the optimized QAOA circuit.

    Args:
        ansatz: Parameterized QAOA circuit.
        params: Optimized QAOA parameters.
        shots: Number of circuit shots.

    Returns:
        Dictionary mapping bitstrings to counts.
    """
    sampler = StatevectorSampler()
    measured = ansatz.copy()
    measured.measure_all()

    job = sampler.run([(measured, [params])], shots=shots)
    result = job.result()[0]

    # Convert sampler output into a standard counts dictionary.
    counts = result.data.meas.get_counts()

    # Qiskit bit order is often reversed relative to node indexing.
    fixed_counts = {}
    for bitstring, count in counts.items():
        fixed_counts[bitstring[::-1]] = fixed_counts.get(bitstring[::-1], 0) + count

    return fixed_counts


def solve_maxcut_qaoa(
    graph: nx.Graph,
    reps: int = 2,
    shots: int = 2048,
    seed: int = 42,
) -> MaxCutResult:
    """Solve Max-Cut approximately with QAOA.

    Args:
        graph: Input graph.
        reps: QAOA depth p.
        shots: Number of measurement shots.
        seed: Random seed for initialization.

    Returns:
        MaxCutResult containing QAOA and exact-solver results.
    """
    rng = np.random.default_rng(seed)

    cost_hamiltonian = build_maxcut_hamiltonian(graph)

    # flatten=True is preferred for performance when binding many parameters
    ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps, flatten=True)

    estimator = StatevectorEstimator()

    initial_params = rng.uniform(0.0, 2.0 * math.pi, ansatz.num_parameters)

    opt_result = minimize(
        expectation_value,
        initial_params,
        args=(ansatz, cost_hamiltonian, estimator),
        method="COBYLA",
        options={"maxiter": 200},
    )

    optimal_params = np.asarray(opt_result.x, dtype=float)
    counts = sample_bitstrings(ansatz, optimal_params, shots=shots)

    best_qaoa_bitstring = max(counts, key=counts.get)
    best_qaoa_cut = maxcut_value(best_qaoa_bitstring, graph)

    exact_bitstring, exact_cut = brute_force_maxcut(graph)

    approximation_ratio = (
        best_qaoa_cut / exact_cut if exact_cut > 0 else 1.0
    )

    return MaxCutResult(
        best_bitstring=best_qaoa_bitstring,
        best_cut_value=best_qaoa_cut,
        exact_bitstring=exact_bitstring,
        exact_cut_value=exact_cut,
        approximation_ratio=approximation_ratio,
        optimal_parameters=optimal_params,
        counts=counts,
    )


def print_results(result: MaxCutResult) -> None:
    """Pretty-print QAOA and exact Max-Cut results."""
    print("\n===== QAOA Max-Cut Results =====")
    print(f"Best sampled bitstring : {result.best_bitstring}")
    print(f"Best sampled cut value : {result.best_cut_value}")
    print(f"Exact optimal bitstring: {result.exact_bitstring}")
    print(f"Exact optimal cut value: {result.exact_cut_value}")
    print(f"Approximation ratio    : {result.approximation_ratio:.4f}")
    print(f"Optimal parameters     : {np.round(result.optimal_parameters, 4)}")

    print("\nTop sampled bitstrings:")
    sorted_counts = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)
    for bitstring, count in sorted_counts[:10]:
        print(
            f"  {bitstring} | count={count:4d} | cut={result.best_cut_value if bitstring == result.best_bitstring else 'n/a'}"
        )


def build_example_graph() -> nx.Graph:
    """Create a small example graph suitable for NISQ-scale QAOA tests."""
    g = nx.Graph()
    g.add_nodes_from(range(4))
    g.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 2),
        ]
    )
    return g


if __name__ == "__main__":
    graph = build_example_graph()

    print("Graph nodes:", list(graph.nodes()))
    print("Graph edges:", list(graph.edges()))

    result = solve_maxcut_qaoa(
        graph=graph,
        reps=2,      # QAOA depth p
        shots=2048,
        seed=42,
    )
    print_results(result)