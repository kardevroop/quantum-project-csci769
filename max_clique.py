"""
QAOA Max-Clique experiment framework.

Supports:
1. Random connected graph generation
2. Exact Max-Clique using NetworkX Bron-Kerbosch-style clique search
3. Noiseless simulation
4. Noisy simulation with selectable custom noise models
5. Optional IBM hardware execution
6. Best and average approximation-ratio tracking
7. Depth sweep for accumulated-noise analysis

Expected local files:
    from objects import ExperimentSummary, SingleRunResult
    from helper import generate_full_report

Required SingleRunResult fields:
    setting, depth, best_bitstring, best_cut_value, exact_cut_value,
    approximation_ratio, average_cut_value, average_approximation_ratio,
    runtime_seconds, shots, optimizer_success, optimizer_fun,
    optimal_parameters, counts
"""

from __future__ import annotations

import itertools
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from objects import ExperimentSummary, SingleRunResult
from helper import generate_full_report, plot_best_vs_average_ratio

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        ReadoutError,
        depolarizing_error,
        thermal_relaxation_error,
    )
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
# Graph generation
# ---------------------------------------------------------------------

def build_connected_random_graph(
    n_nodes: int,
    edge_prob: float = 0.5,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate a connected random graph."""
    rng = random.Random(seed)

    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))

    nodes = list(range(n_nodes))
    rng.shuffle(nodes)

    # Random spanning tree to guarantee connectivity.
    for i in range(1, n_nodes):
        u = nodes[i]
        v = rng.choice(nodes[:i])
        graph.add_edge(u, v)

    # Add extra random edges.
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if not graph.has_edge(i, j) and rng.random() < edge_prob:
                graph.add_edge(i, j)

    return graph


# ---------------------------------------------------------------------
# Max-Clique utilities
# ---------------------------------------------------------------------

def selected_nodes_from_bitstring(bitstring: str) -> List[int]:
    """Return selected nodes where bit is 1."""
    return [i for i, bit in enumerate(bitstring) if bit == "1"]


def is_clique(bitstring: str, graph: nx.Graph) -> bool:
    """Check whether selected nodes form a clique."""
    selected = selected_nodes_from_bitstring(bitstring)

    for i, u in enumerate(selected):
        for v in selected[i + 1:]:
            if not graph.has_edge(u, v):
                return False

    return True


def clique_size(bitstring: str) -> int:
    """Return number of selected nodes."""
    return bitstring.count("1")


def exact_max_clique(graph: nx.Graph) -> Tuple[str, int, List[int]]:
    """Compute exact maximum clique using NetworkX Bron-Kerbosch-style search."""
    max_clique_nodes = max(nx.find_cliques(graph), key=len)

    bitstring = ["0"] * graph.number_of_nodes()
    for node in max_clique_nodes:
        bitstring[node] = "1"

    return "".join(bitstring), len(max_clique_nodes), sorted(max_clique_nodes)


def extract_best_clique_from_counts(
    counts: Dict[str, int],
    graph: nx.Graph,
) -> Tuple[str, int, List[int]]:
    """Find largest valid clique among sampled bitstrings."""
    best_bitstring = ""
    best_size = 0

    for bitstring in counts:
        if is_clique(bitstring, graph):
            size = clique_size(bitstring)
            if size > best_size:
                best_size = size
                best_bitstring = bitstring

    return best_bitstring, best_size, selected_nodes_from_bitstring(best_bitstring)


def average_valid_clique_size_from_counts(
    counts: Dict[str, int],
    graph: nx.Graph,
) -> float:
    """Average valid clique size over sampled distribution.

    Invalid cliques contribute 0.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    weighted_sum = 0.0
    for bitstring, count in counts.items():
        if is_clique(bitstring, graph):
            weighted_sum += clique_size(bitstring) * count

    return weighted_sum / total


def valid_sample_rate_from_counts(
    counts: Dict[str, int],
    graph: nx.Graph,
) -> float:
    """Fraction of samples that are valid cliques."""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    valid = sum(
        count for bitstring, count in counts.items()
        if is_clique(bitstring, graph)
    )

    return valid / total


# ---------------------------------------------------------------------
# Max-Clique Hamiltonian
# ---------------------------------------------------------------------

def _add_term(coeffs: Dict[str, float], pauli: str, coeff: float) -> None:
    coeffs[pauli] = coeffs.get(pauli, 0.0) + coeff


def build_max_clique_hamiltonian(
    graph: nx.Graph,
    penalty: Optional[float] = None,
) -> SparsePauliOp:
    """Build Max-Clique QUBO Hamiltonian.

    Objective:
        maximize sum_i x_i - A * sum_{(i,j) not in E} x_i x_j

    where x_i = 1 means node i is selected.
    """
    n = graph.number_of_nodes()

    if penalty is None:
        penalty = float(n)

    coeffs: Dict[str, float] = {}
    identity = "I" * n

    def z_label(*qubits: int) -> str:
        label = ["I"] * n
        for q in qubits:
            label[n - 1 - q] = "Z"
        return "".join(label)

    # Reward: x_i = (I - Z_i) / 2
    for i in range(n):
        _add_term(coeffs, identity, 0.5)
        _add_term(coeffs, z_label(i), -0.5)

    # Penalty for selecting non-edge pairs:
    # -A x_i x_j = -A/4 (I - Z_i - Z_j + Z_i Z_j)
    for i, j in itertools.combinations(range(n), 2):
        if graph.has_edge(i, j):
            continue

        _add_term(coeffs, identity, -penalty / 4.0)
        _add_term(coeffs, z_label(i), penalty / 4.0)
        _add_term(coeffs, z_label(j), penalty / 4.0)
        _add_term(coeffs, z_label(i, j), -penalty / 4.0)

    terms = [(p, c) for p, c in coeffs.items() if abs(c) > 1e-12]
    return SparsePauliOp.from_list(terms)


def make_qaoa_ansatz(
    graph: nx.Graph,
    reps: int,
    penalty: Optional[float] = None,
) -> Tuple[QAOAAnsatz, SparsePauliOp]:
    """Create QAOA ansatz for Max-Clique."""
    hamiltonian = build_max_clique_hamiltonian(graph, penalty=penalty)
    ansatz = QAOAAnsatz(
        cost_operator=hamiltonian,
        reps=reps,
        flatten=True,
    )
    return ansatz, hamiltonian


# ---------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------

def qaoa_objective(
    params: np.ndarray,
    ansatz: QAOAAnsatz,
    hamiltonian: SparsePauliOp,
    estimator: StatevectorEstimator,
) -> float:
    """Negative expected objective for scipy minimization."""
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run([pub]).result()
    expectation = result[0].data.evs[0]
    return -float(expectation)


def optimize_max_clique_qaoa(
    graph: nx.Graph,
    reps: int,
    penalty: Optional[float] = None,
    seed: int = 42,
    maxiter: int = 300,
) -> Tuple[QAOAAnsatz, SparsePauliOp, np.ndarray, object]:
    """Optimize QAOA parameters using noiseless estimator."""
    rng = np.random.default_rng(seed)

    ansatz, hamiltonian = make_qaoa_ansatz(
        graph=graph,
        reps=reps,
        penalty=penalty,
    )

    estimator = StatevectorEstimator()
    initial_params = rng.uniform(
        0.0,
        2.0 * math.pi,
        size=ansatz.num_parameters,
    )

    opt_result = minimize(
        qaoa_objective,
        initial_params,
        args=(ansatz, hamiltonian, estimator),
        method="COBYLA",
        options={"maxiter": maxiter},
    )

    return ansatz, hamiltonian, np.asarray(opt_result.x, dtype=float), opt_result


# ---------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------

SINGLE_QUBIT_GATES = ["u1", "u2", "u3", "h", "x", "sx", "rx", "ry", "rz"]
TWO_QUBIT_GATES = ["cx"]


def create_depolarizing_noise(
    p1: float = 0.001,
    p2: float = 0.01,
) -> "NoiseModel":
    """Create depolarizing noise model."""
    noise_model = NoiseModel()

    error_1 = depolarizing_error(p1, 1)
    error_2 = depolarizing_error(p2, 2)

    noise_model.add_all_qubit_quantum_error(error_1, SINGLE_QUBIT_GATES)
    noise_model.add_all_qubit_quantum_error(error_2, TWO_QUBIT_GATES)

    return noise_model


def create_thermal_noise(
    t1: float = 50e3,
    t2: float = 70e3,
    gate_time_1q: float = 50,
    gate_time_2q: float = 300,
) -> "NoiseModel":
    """Create thermal relaxation noise model.

    Times are in ns.
    """
    noise_model = NoiseModel()

    error_1 = thermal_relaxation_error(t1, t2, gate_time_1q)

    error_2_single = thermal_relaxation_error(t1, t2, gate_time_2q)
    error_2 = error_2_single.tensor(error_2_single)

    noise_model.add_all_qubit_quantum_error(error_1, SINGLE_QUBIT_GATES)
    noise_model.add_all_qubit_quantum_error(error_2, TWO_QUBIT_GATES)

    return noise_model


def create_readout_noise(
    prob_0to1: float = 0.02,
    prob_1to0: float = 0.02,
) -> "NoiseModel":
    """Create readout-only noise model."""
    noise_model = NoiseModel()

    readout_error = ReadoutError([
        [1.0 - prob_0to1, prob_0to1],
        [prob_1to0, 1.0 - prob_1to0],
    ])

    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model


def create_combined_noise(
    p1: float = 0.001,
    p2: float = 0.01,
    t1: float = 50e3,
    t2: float = 70e3,
    gate_time_1q: float = 50,
    gate_time_2q: float = 300,
    prob_0to1: float = 0.02,
    prob_1to0: float = 0.02,
) -> "NoiseModel":
    """Create combined depolarizing + thermal + readout noise."""
    noise_model = NoiseModel()

    depol_1 = depolarizing_error(p1, 1)
    depol_2 = depolarizing_error(p2, 2)

    thermal_1 = thermal_relaxation_error(t1, t2, gate_time_1q)

    thermal_2_single = thermal_relaxation_error(t1, t2, gate_time_2q)
    thermal_2 = thermal_2_single.tensor(thermal_2_single)

    readout_error = ReadoutError([
        [1.0 - prob_0to1, prob_0to1],
        [prob_1to0, 1.0 - prob_1to0],
    ])

    noise_model.add_all_qubit_quantum_error(
        depol_1.compose(thermal_1),
        SINGLE_QUBIT_GATES,
    )
    noise_model.add_all_qubit_quantum_error(
        depol_2.compose(thermal_2),
        TWO_QUBIT_GATES,
    )
    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


def make_noise_model(
    noise_type: str = "combined",
    noise_scale: float = 1.0,
) -> "NoiseModel":
    """Create selected noise model.

    noise_type:
        "depolarizing", "thermal", "readout", "combined"

    Increasing noise_scale increases noise strength.
    """
    if not AER_AVAILABLE:
        raise ImportError("qiskit-aer is not installed.")

    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive.")

    if noise_type == "depolarizing":
        return create_depolarizing_noise(
            p1=min(1.0, 0.001 * noise_scale),
            p2=min(1.0, 0.01 * noise_scale),
        )

    if noise_type == "thermal":
        return create_thermal_noise(
            t1=50e3 / noise_scale,
            t2=70e3 / noise_scale,
            gate_time_1q=50,
            gate_time_2q=300,
        )

    if noise_type == "readout":
        return create_readout_noise(
            prob_0to1=min(0.5, 0.02 * noise_scale),
            prob_1to0=min(0.5, 0.02 * noise_scale),
        )

    if noise_type == "combined":
        return create_combined_noise(
            p1=min(1.0, 0.001 * noise_scale),
            p2=min(1.0, 0.01 * noise_scale),
            t1=50e3 / noise_scale,
            t2=70e3 / noise_scale,
            gate_time_1q=50,
            gate_time_2q=300,
            prob_0to1=min(0.5, 0.02 * noise_scale),
            prob_1to0=min(0.5, 0.02 * noise_scale),
        )

    raise ValueError(f"Unknown noise_type={noise_type}")


def make_noisy_simulator(
    noise_type: str = "combined",
    noise_scale: float = 1.0,
) -> "AerSimulator":
    """Create Aer simulator with selected custom noise model."""
    if not AER_AVAILABLE:
        raise ImportError("qiskit-aer is not installed.")

    return AerSimulator(
        noise_model=make_noise_model(
            noise_type=noise_type,
            noise_scale=noise_scale,
        )
    )


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------

def reverse_bitstring_counts(counts: Dict[str, int]) -> Dict[str, int]:
    """Reverse Qiskit bitstrings to match graph node indexing."""
    fixed: Dict[str, int] = {}

    for bitstring, count in counts.items():
        reversed_bitstring = bitstring[::-1]
        fixed[reversed_bitstring] = fixed.get(reversed_bitstring, 0) + count

    return fixed


def sample_noiseless(
    ansatz: QAOAAnsatz,
    params: np.ndarray,
    shots: int,
) -> Dict[str, int]:
    """Sample noiselessly using StatevectorSampler."""
    sampler = StatevectorSampler()

    circ = ansatz.copy()
    circ.measure_all()

    job = sampler.run([(circ, [params])], shots=shots)
    result = job.result()[0]

    counts = result.data.meas.get_counts()
    return reverse_bitstring_counts(counts)


def sample_with_aer(
    ansatz: QAOAAnsatz,
    params: np.ndarray,
    simulator: "AerSimulator",
    shots: int,
    seed: int = 42,
) -> Dict[str, int]:
    """Sample with Aer simulator."""
    circ = ansatz.assign_parameters(params)
    measured = circ.copy()
    measured.measure_all()

    tqc = transpile(
        measured,
        backend=simulator,
        seed_transpiler=seed,
        optimization_level=3,
    )

    result = simulator.run(
        tqc,
        shots=shots,
        seed_simulator=seed,
    ).result()

    return reverse_bitstring_counts(result.get_counts())


# ---------------------------------------------------------------------
# Optional IBM hardware
# ---------------------------------------------------------------------

def get_runtime_service() -> "QiskitRuntimeService":
    """Initialize IBM Runtime service."""
    if not IBM_RUNTIME_AVAILABLE:
        raise ImportError(
            "qiskit-ibm-runtime is not installed. "
            "Install it with: pip install qiskit-ibm-runtime"
        )

    return QiskitRuntimeService()


def get_hardware_backend(backend_name: Optional[str] = None):
    """Get IBM backend."""
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
    """Sample optimized circuit on IBM hardware."""
    if not IBM_RUNTIME_AVAILABLE:
        raise ImportError("qiskit-ibm-runtime is not installed.")

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
        raise RuntimeError("Could not extract counts from hardware result.")

    return reverse_bitstring_counts(counts)


# ---------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------

def run_single_setting(
    graph: nx.Graph,
    reps: int,
    setting: str,
    exact_clique_size: int,
    penalty: Optional[float] = None,
    shots: int = 2048,
    seed: int = 42,
    maxiter: int = 300,
    backend_name: Optional[str] = None,
    noise_type: str = "combined",
    noise_scale: float = 1.0,
) -> SingleRunResult:
    """Run one Max-Clique QAOA setting for one depth."""
    start = time.perf_counter()

    ansatz, _hamiltonian, optimal_params, opt_result = optimize_max_clique_qaoa(
        graph=graph,
        reps=reps,
        penalty=penalty,
        seed=seed,
        maxiter=maxiter,
    )

    if setting == "noiseless":
        counts = sample_noiseless(
            ansatz=ansatz,
            params=optimal_params,
            shots=shots,
        )

    elif setting == "noisy":
        simulator = make_noisy_simulator(
            noise_type=noise_type,
            noise_scale=noise_scale,
        )

        counts = sample_with_aer(
            ansatz=ansatz,
            params=optimal_params,
            simulator=simulator,
            shots=shots,
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
        raise ValueError(f"Unknown setting={setting}")

    runtime_seconds = time.perf_counter() - start

    best_bitstring, best_clique_size, best_nodes = extract_best_clique_from_counts(
        counts=counts,
        graph=graph,
    )

    best_approximation_ratio = (
        best_clique_size / exact_clique_size
        if exact_clique_size > 0
        else 1.0
    )

    average_clique_size = average_valid_clique_size_from_counts(
        counts=counts,
        graph=graph,
    )

    average_approximation_ratio = (
        average_clique_size / exact_clique_size
        if exact_clique_size > 0
        else 1.0
    )

    valid_rate = valid_sample_rate_from_counts(
        counts=counts,
        graph=graph,
    )

    print(
        f"{setting} | p={reps} | "
        f"best_clique={best_clique_size} | "
        f"best_ratio={best_approximation_ratio:.4f} | "
        f"avg_clique={average_clique_size:.4f} | "
        f"avg_ratio={average_approximation_ratio:.4f} | "
        f"valid_rate={valid_rate:.4f} | "
        f"runtime={runtime_seconds:.4f}s | "
        f"best_nodes={best_nodes}"
    )

    return SingleRunResult(
        setting=setting,
        depth=reps,
        best_bitstring=best_bitstring,

        # Reused names so existing helper plots work:
        best_cut_value=best_clique_size,
        exact_cut_value=exact_clique_size,
        approximation_ratio=best_approximation_ratio,
        average_cut_value=average_clique_size,
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
    penalty: Optional[float] = None,
    shots: int = 2048,
    seed: int = 42,
    maxiter: int = 300,
    include_noisy: bool = True,
    include_hardware: bool = False,
    backend_name: Optional[str] = None,
    noise_type: str = "combined",
    noise_scale: float = 1.0,
) -> ExperimentSummary:
    """Run full Max-Clique QAOA sweep over noiseless/noisy/hardware settings."""
    exact_bitstring, exact_clique_size, exact_nodes = exact_max_clique(graph)

    print("\nExact Max-Clique solution")
    print(f"  Exact bitstring   : {exact_bitstring}")
    print(f"  Exact clique size : {exact_clique_size}")
    print(f"  Exact nodes       : {exact_nodes}")

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
                result = run_single_setting(
                    graph=graph,
                    reps=reps,
                    setting=setting,
                    exact_clique_size=exact_clique_size,
                    penalty=penalty,
                    shots=shots,
                    seed=seed,
                    maxiter=maxiter,
                    backend_name=backend_name,
                    noise_type=noise_type,
                    noise_scale=noise_scale,
                )
                results.append(result)

            except Exception as exc:
                print(f"Skipped setting={setting}, p={reps} due to error: {exc}")

    return ExperimentSummary(
        graph_edges=list(graph.edges()),
        exact_bitstring=exact_bitstring,
        exact_cut_value=exact_clique_size,
        results=results,
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_summary(summary: ExperimentSummary) -> None:
    """Print compact Max-Clique summary."""
    print("\n" + "=" * 72)
    print("QAOA MAX-CLIQUE SUMMARY")
    print("=" * 72)

    print(f"Graph edges              : {summary.graph_edges}")
    print(f"Exact clique bitstring   : {summary.exact_bitstring}")
    print(f"Exact maximum clique size: {summary.exact_cut_value}")

    for result in summary.results:
        print("-" * 72)
        print(f"Setting                  : {result.setting}")
        print(f"Depth p                  : {result.depth}")
        print(f"Best bitstring           : {result.best_bitstring}")
        print(f"Best clique size         : {result.best_cut_value}")
        print(f"Best approximation ratio : {result.approximation_ratio:.4f}")
        print(f"Average clique size      : {result.average_cut_value:.4f}")
        print(f"Average approximation    : {result.average_approximation_ratio:.4f}")
        print(f"Runtime                  : {result.runtime_seconds:.4f}s")

    print("=" * 72)


def results_as_table(summary: ExperimentSummary) -> List[dict]:
    """Convert summary to simple rows."""
    rows = []

    for result in summary.results:
        rows.append(
            {
                "setting": result.setting,
                "depth": result.depth,
                "best_bitstring": result.best_bitstring,
                "best_clique_size": result.best_cut_value,
                "average_clique_size": round(result.average_cut_value, 4),
                "exact_clique_size": result.exact_cut_value,
                "best_approximation_ratio": round(result.approximation_ratio, 4),
                "average_approximation_ratio": round(
                    result.average_approximation_ratio,
                    4,
                ),
                "runtime_seconds": round(result.runtime_seconds, 4),
                "shots": result.shots,
                "optimizer_success": result.optimizer_success,
            }
        )

    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    nodes = 6
    save_dir = f"results_max_clique_n{nodes}"

    graph = build_connected_random_graph(
        n_nodes=nodes,
        edge_prob=0.5,
        seed=42,
    )

    print("Graph nodes:", list(graph.nodes()))
    print("Graph edges:", list(graph.edges()))
    print("Number of edges:", graph.number_of_edges())

    depths = list(range(1, 21))

    summary = run_full_experiment(
        graph=graph,
        depths=depths,
        penalty=None,              # defaults to n_nodes
        shots=4096,
        seed=42,
        maxiter=300,
        include_noisy=True,
        include_hardware=True,    # set True when ready for IBM hardware
        backend_name="ibm_kingston",
        noise_type="combined",     # depolarizing, thermal, readout, combined
        noise_scale=5.0,           # increase to 10.0 for stronger noise
    )

    print_summary(summary)

    generate_full_report(
        summary=summary,
        outdir=save_dir,
    )
    plot_best_vs_average_ratio(
        summary=summary,
        outdir=save_dir,
    )

    print("\nCompact results table:")
    for row in results_as_table(summary):
        print(row)