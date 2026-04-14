from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise import (
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)


AER_AVAILABLE = True


def make_custom_noise_model() -> NoiseModel:
    """Build a simple custom noise model for QAOA experiments."""
    noise_model = NoiseModel()

    # 1-qubit gate noise
    one_qubit_error = depolarizing_error(0.001, 1)

    # 2-qubit gate noise
    two_qubit_error = depolarizing_error(0.01, 2)

    # Readout noise
    readout_error = ReadoutError([
        [0.98, 0.02],  # P(measure 0 | true 0), P(measure 1 | true 0)
        [0.03, 0.97],  # P(measure 0 | true 1), P(measure 1 | true 1)
    ])

    # Apply to common 1-qubit gates used after transpilation
    for gate in ["h", "x", "sx", "rx", "rz"]:
        noise_model.add_all_qubit_quantum_error(one_qubit_error, gate)

    # Apply to entangling gates
    noise_model.add_all_qubit_quantum_error(two_qubit_error, "cx")

    # Apply readout noise to all qubits
    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


def make_noisy_simulator() -> AerSimulator:
    """Create an Aer simulator with a custom noise model."""
    noise_model = make_custom_noise_model()
    return AerSimulator(noise_model=noise_model)


def make_noisy_simulator_from_ibm_backend(
    backend,
    enforce_backend_topology: bool = False,
) -> "AerSimulator":
    """Create a noisy simulator using an IBM backend-derived noise model.

    Args:
        backend: IBM backend object.
        enforce_backend_topology: If True, also enforce the backend coupling map
            and basis gates. If False, use only the backend-derived noise model.

    Returns:
        Configured AerSimulator.
    """
    if not AER_AVAILABLE:
        raise ImportError(
            "qiskit-aer is not installed. Install it with: pip install qiskit-aer"
        )

    noise_model = NoiseModel.from_backend(backend)

    if enforce_backend_topology:
        return AerSimulator(
            noise_model=noise_model,
            coupling_map=backend.coupling_map,
            basis_gates=noise_model.basis_gates,
        )

    return AerSimulator(noise_model=noise_model)
