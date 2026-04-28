from __future__ import annotations

import itertools
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from objects import ExperimentSummary, SingleRunResult
from helper import generate_full_report
from noise_models import (
    make_custom_noise_model, 
    make_noisy_simulator, 
    make_noisy_simulator_from_ibm_backend,
)
from builder import (
    build_connected_random_graph,
    build_random_graph,
    build_weighted_random_graph,
)
from helper import (
    generate_full_report, 
    plot_best_vs_average_ratio,
    plot_graph_structure,
    plot_qaoa_circuit,
    plot_classical_vs_quantum_maxcut,
)

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except Exception:
    AER_AVAILABLE = False

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    IBM_RUNTIME_AVAILABLE = True
except Exception:
    IBM_RUNTIME_AVAILABLE = False


# ---------------------------------------------------------------------
# Max-Cut utilities
# ---------------------------------------------------------------------

def maxcut_value(bitstring: str, graph: nx.Graph) -> int:
    """Compute the Max-Cut value for a given bitstring assignment."""
    value = 0
    for u, v in graph.edges():
        if bitstring[u] != bitstring[v]:
            value += 1
    return value

def average_cut_from_counts(counts: Dict[str, int], graph: nx.Graph) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0

    return sum(
        maxcut_value(bitstring, graph) * count
        for bitstring, count in counts.items()
    ) / total


def brute_force_maxcut(graph: nx.Graph) -> Tuple[str, int]:
    """Compute the exact Max-Cut solution by exhaustive search."""
    n = graph.number_of_nodes()
    best_bitstring = ""
    best_value = -1

    for bits in itertools.product("01", repeat=n):
        bitstring = "".join(bits)
        value = maxcut_value(bitstring, graph)
        if value > best_value:
            best_value = value
            best_bitstring = bitstring

    return best_bitstring, best_value


def build_maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """Construct the Max-Cut cost Hamiltonian.

    Hamiltonian:
        C = sum_(i,j in E) 0.5 * (I - Z_i Z_j)

    Returns:
        SparsePauliOp representation of the cost operator.
    """
    n = graph.number_of_nodes()
    terms: List[Tuple[str, float]] = []

    # Constant identity contribution
    terms.append(("I" * n, 0.5 * graph.number_of_edges()))

    # Edge interaction terms
    for u, v in graph.edges():
        label = ["I"] * n
        label[n - 1 - u] = "Z"
        label[n - 1 - v] = "Z"
        terms.append(("".join(label), -0.5))

    return SparsePauliOp.from_list(terms)


def make_qaoa_ansatz(graph: nx.Graph, reps: int) -> Tuple[QAOAAnsatz, SparsePauliOp]:
    """Create the QAOA ansatz and corresponding Max-Cut Hamiltonian."""
    cost_hamiltonian = build_maxcut_hamiltonian(graph)
    ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps, flatten=True)
    return ansatz, cost_hamiltonian


# ---------------------------------------------------------------------
# Parameter optimization
# ---------------------------------------------------------------------

def qaoa_objective(
    params: np.ndarray,
    ansatz: QAOAAnsatz,
    cost_hamiltonian: SparsePauliOp,
    estimator: StatevectorEstimator,
) -> float:
    """Objective for classical optimization: negative expected cost."""
    pub = (ansatz, [cost_hamiltonian], [params])
    result = estimator.run([pub]).result()
    energy = result[0].data.evs[0]
    return -float(energy)


def optimize_qaoa_parameters(
    graph: nx.Graph,
    reps: int,
    seed: int = 42,
    maxiter: int = 200,
) -> Tuple[QAOAAnsatz, SparsePauliOp, np.ndarray, object]:
    """Optimize QAOA parameters using a noiseless estimator."""
    rng = np.random.default_rng(seed)

    ansatz, cost_hamiltonian = make_qaoa_ansatz(graph, reps=reps)
    estimator = StatevectorEstimator()

    initial_params = rng.uniform(0.0, 2.0 * math.pi, size=ansatz.num_parameters)

    opt_result = minimize(
        qaoa_objective,
        initial_params,
        args=(ansatz, cost_hamiltonian, estimator),
        method="COBYLA",
        options={"maxiter": maxiter},
    )

    optimal_params = np.asarray(opt_result.x, dtype=float)
    return ansatz, cost_hamiltonian, optimal_params, opt_result


# ---------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------

def reverse_bitstring_counts(counts: Dict[str, int]) -> Dict[str, int]:
    """Reverse Qiskit's bitstring order to align with node indexing."""
    fixed: Dict[str, int] = {}
    for bitstring, count in counts.items():
        reversed_bitstring = bitstring[::-1]
        fixed[reversed_bitstring] = fixed.get(reversed_bitstring, 0) + count
    return fixed


def extract_best_bitstring(counts: Dict[str, int], graph: nx.Graph) -> Tuple[str, int]:
    """Find the best cut value among sampled bitstrings."""
    best_string = ""
    best_cut = -1

    for bitstring, _count in counts.items():
        cut = maxcut_value(bitstring, graph)
        if cut > best_cut:
            best_cut = cut
            best_string = bitstring

    return best_string, best_cut


def sample_noiseless(
    ansatz: QAOAAnsatz,
    params: np.ndarray,
    shots: int,
) -> Dict[str, int]:
    """Sample using the local statevector sampler."""
    sampler = StatevectorSampler()
    circ = ansatz.copy()
    circ.measure_all()

    job = sampler.run([(circ, [params])], shots=shots)
    result = job.result()[0]
    counts = result.data.meas.get_counts()
    return reverse_bitstring_counts(counts)


def sample_with_aer_backend(
    ansatz: QAOAAnsatz,
    params: np.ndarray,
    shots: int,
    simulator: "AerSimulator",
    seed: int = 42,
) -> Dict[str, int]:
    """Sample a bound QAOA circuit using an Aer simulator."""
    circ = ansatz.assign_parameters(params)
    measured = circ.copy()
    measured.measure_all()

    transpiled = transpile(
        measured,
        backend=simulator,
        seed_transpiler=seed,
        optimization_level=3,
    )

    result = simulator.run(
        transpiled,
        shots=shots,
        seed_simulator=seed,
    ).result()

    counts = result.get_counts()
    return reverse_bitstring_counts(counts)


# ---------------------------------------------------------------------
# IBM backend helpers
# ---------------------------------------------------------------------

def get_runtime_service() -> "QiskitRuntimeService":
    """Initialize Qiskit Runtime service."""
    if not IBM_RUNTIME_AVAILABLE:
        raise ImportError(
            "qiskit-ibm-runtime is not installed. "
            "Install it with: pip install qiskit-ibm-runtime"
        )
    return QiskitRuntimeService()


def get_hardware_backend(
    backend_name: Optional[str] = None,
):
    """Get an IBM Quantum backend.

    If backend_name is None, choose the least busy operational backend.
    """
    service = get_runtime_service()

    if backend_name is not None:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(operational=True, simulator=False)

    return service, backend


def sample_on_ibm_hardware(
    ansatz: QAOAAnsatz,
    params: np.ndarray,
    backend,
    shots: int,
    seed: int = 42,
) -> Dict[str, int]:
    """Run the optimized circuit on real IBM Quantum hardware via SamplerV2."""
    if not IBM_RUNTIME_AVAILABLE:
        raise ImportError(
            "qiskit-ibm-runtime is not installed. "
            "Install it with: pip install qiskit-ibm-runtime"
        )

    circ = ansatz.assign_parameters(params)
    circ.measure_all()

    isa_circ = transpile(
        circ,
        backend=backend,
        seed_transpiler=seed,
        optimization_level=3,
    )

    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circ], shots=shots)
    result = job.result()[0]

    if hasattr(result.data, "meas"):
        counts = result.data.meas.get_counts()
    elif hasattr(result.data, "c"):
        counts = result.data.c.get_counts()
    else:
        raise RuntimeError(
            "Could not extract counts from hardware result. "
            "Inspect result.data to adjust measurement key."
        )

    return reverse_bitstring_counts(counts)


# ---------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------

def run_single_setting(
    graph: nx.Graph,
    reps: int,
    setting: str,
    exact_cut_value: int,
    shots: int = 2048,
    seed: int = 42,
    maxiter: int = 200,
    backend_name: Optional[str] = None,
    noise_type: str = None,
    noisy_use_backend_topology: bool = False,
    save_dir: str = "results"
) -> SingleRunResult:
    """Run one experiment for one QAOA depth and one execution setting.

    Args:
        graph: Input graph.
        reps: QAOA depth p.
        setting: One of {"noiseless", "noisy", "hardware"}.
        exact_cut_value: Exact Max-Cut value.
        shots: Number of shots.
        seed: Random seed.
        maxiter: Maximum optimizer iterations.
        backend_name: IBM backend name for noisy/hardware settings.
        noisy_use_backend_topology: Whether noisy simulation should also enforce
            backend coupling-map constraints.

    Returns:
        SingleRunResult for the selected setting and depth.
    """
    start = time.perf_counter()

    ansatz, _hamiltonian, optimal_params, opt_result = optimize_qaoa_parameters(
        graph=graph,
        reps=reps,
        seed=seed,
        maxiter=maxiter,
    )
    plot_qaoa_circuit(
        ansatz,
        outdir=os.path.join(save_dir,"qaoa_circuit"),
        filename=f"qaoa_circuit_{reps}_{setting}.png"
    )

    if setting == "noiseless":
        counts = sample_noiseless(ansatz, optimal_params, shots=shots)

    elif setting == "noisy":
        _service, backend = get_hardware_backend(backend_name=backend_name)

        # simulator = make_noisy_simulator_from_ibm_backend(
        #     backend=backend,
        #     enforce_backend_topology=noisy_use_backend_topology,
        # )

        simulator = make_noisy_simulator(type=noise_type)

        counts = sample_with_aer_backend(
            ansatz=ansatz,
            params=optimal_params,
            shots=shots,
            simulator=simulator,
            seed=seed,
        )

    elif setting == "hardware":
        _service, backend = get_hardware_backend(backend_name=backend_name)
        counts = sample_on_ibm_hardware(
            ansatz=ansatz,
            params=optimal_params,
            backend=backend,
            shots=shots,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown setting: {setting}")

    runtime_seconds = time.perf_counter() - start
    best_bitstring, best_cut_value = extract_best_bitstring(counts, graph)
    
    best_approximation_ratio = (
        best_cut_value / exact_cut_value if exact_cut_value > 0 else 1.0
    )

    average_cut_value = average_cut_from_counts(counts, graph)

    average_approximation_ratio = (
        average_cut_value / exact_cut_value if exact_cut_value > 0 else 1.0
    )

    return SingleRunResult(
        setting=setting,
        depth=reps,
        best_bitstring=best_bitstring,
        best_cut_value=best_cut_value,
        exact_cut_value=exact_cut_value,
        best_approximation_ratio=best_approximation_ratio,
        average_cut_value=average_cut_value,
        average_approximation_ratio=average_approximation_ratio,
        runtime_seconds=runtime_seconds,
        shots=shots,
        optimizer_success=bool(opt_result.success),
        optimizer_fun=float(opt_result.fun),
        optimal_parameters=[float(x) for x in optimal_params],
        counts=counts,
    )


def run_full_experiment(
    graph: nx.Graph,
    depths: List[int],
    shots: int = 2048,
    seed: int = 42,
    maxiter: int = 200,
    noise_type: Optional[str] = None,
    include_noisy: bool = True,
    include_hardware: bool = False,
    backend_name: Optional[str] = None,
    noisy_use_backend_topology: bool = False,
    save_dir: str = "results",
) -> ExperimentSummary:
    """Run the full Section 4 experiment sweep."""
    exact_bitstring, exact_cut_value = brute_force_maxcut(graph)

    settings = ["noiseless"]
    if include_noisy:
        settings.append("noisy")
    if include_hardware:
        settings.append("hardware")

    results: List[SingleRunResult] = []

    for reps in depths:
        for setting in settings:
            print(f"\nRunning setting='{setting}', depth p={reps}")
            try:
                run_result = run_single_setting(
                    graph=graph,
                    reps=reps,
                    setting=setting,
                    exact_cut_value=exact_cut_value,
                    shots=shots,
                    seed=seed,
                    maxiter=maxiter,
                    backend_name=backend_name,
                    noise_type=noise_type,
                    noisy_use_backend_topology=noisy_use_backend_topology,
                    save_dir=save_dir,
                )
                results.append(run_result)
            except Exception as exc:
                print(f"Skipped {setting} at p={reps} due to error: {exc}")

    return ExperimentSummary(
        graph_edges=list(graph.edges()),
        exact_bitstring=exact_bitstring,
        exact_cut_value=exact_cut_value,
        results=results,
    )


# ---------------------------------------------------------------------
# Convenience output
# ---------------------------------------------------------------------

def print_summary(summary: ExperimentSummary) -> None:
    """Print a readable experiment summary."""
    print("\n" + "=" * 72)
    print("MAX-CUT QAOA EXPERIMENT SUMMARY")
    print("=" * 72)
    print(f"Graph edges        : {summary.graph_edges}")
    print(f"Exact best string  : {summary.exact_bitstring}")
    print(f"Exact cut value    : {summary.exact_cut_value}")

    print("\nResults:")
    for r in summary.results:
        print("-" * 72)
        print(f"Setting            : {r.setting}")
        print(f"Depth p            : {r.depth}")
        print(f"Best bitstring     : {r.best_bitstring}")
        print(f"Best cut value     : {r.best_cut_value}")
        print(f"Best Approximation ratio: {r.best_approximation_ratio:.4f}")
        print(f"Avg Approximation ratio: {r.average_approximation_ratio:.4f}")
        print(f"Runtime (sec)      : {r.runtime_seconds:.4f}")
        print(f"Optimizer success  : {r.optimizer_success}")
        print(f"Objective value    : {r.optimizer_fun:.6f}")

        top_counts = sorted(r.counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top sampled states :")
        for bitstring, count in top_counts:
            print(f"  {bitstring}: {count}")

    print("=" * 72)


def results_as_table(summary: ExperimentSummary) -> List[dict]:
    """Convert results into a table-friendly list of dictionaries."""
    rows = []
    for r in summary.results:
        rows.append(
            {
                "setting": r.setting,
                "depth": r.depth,
                "best_bitstring": r.best_bitstring,
                "best_cut_value": r.best_cut_value,
                "exact_cut_value": r.exact_cut_value,
                "best_approximation_ratio": round(r.best_approximation_ratio, 4),
                "average_approximation_ratio": round(r.average_approximation_ratio, 4),
                "runtime_seconds": round(r.runtime_seconds, 4),
                "shots": r.shots,
                "optimizer_success": r.optimizer_success,
            }
        )
    return rows


# ---------------------------------------------------------------------
# Example graphs
# ---------------------------------------------------------------------

def build_example_graph_4() -> nx.Graph:
    """Small 4-node graph."""
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


# def build_example_graph_6() -> nx.Graph:
#     """Slightly larger 6-node graph."""
#     g = nx.Graph()
#     g.add_nodes_from(range(6))
#     g.add_edges_from(
#         [
#             (0, 1), (0, 2), (1, 3), (2, 3),
#             (3, 4), (4, 5), (2, 5), (1, 4),
#         ]
#     )
#     return g

def build_dense_graph_6() -> nx.Graph:
    """Dense 6-node graph (challenging Max-Cut instance)."""
    g = nx.Graph()
    g.add_nodes_from(range(6))

    g.add_edges_from([
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 5),
        (2, 3), (2, 4), (2, 5),
        (3, 5),
        (4, 5),
    ])

    return g


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # graph = build_dense_graph_6()
    nodes = 7
    noise_type = "all"

    save_dir = f"results_v{nodes}_n{noise_type}"

    graph = build_connected_random_graph(
        n_nodes=nodes,
        edge_prob=0.7,
        seed=42,
    )
    plot_graph_structure(graph, outdir=save_dir)

    depths = [1, 2, 4, 8, 16]

    summary = run_full_experiment(
        graph=graph,
        depths=depths,
        shots=2048,
        seed=42,
        maxiter=200,
        noise_type=noise_type,
        include_noisy=True,
        include_hardware=True,
        backend_name="ibm_kingston",
        noisy_use_backend_topology=False,  # set True only if you want coupling-map constraints too
        save_dir=save_dir,
    )

    print_summary(summary)
    generate_full_report(summary, outdir=save_dir)
    # generate_depth_tradeoff_plots(
    #     summary=summary,
    #     outdir=save_dir,
    # )
    plot_best_vs_average_ratio(
        summary=summary,
        outdir=save_dir,
    )
    plot_classical_vs_quantum_maxcut(
        summary=summary,
        outdir=save_dir,
    )

    print("\nCompact results table:")
    for row in results_as_table(summary):
        print(row)