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


def make_noisy_simulator(type=None) -> AerSimulator:
    """Create an Aer simulator with a custom noise model."""
    if type is None:
        noise_model = make_custom_noise_model()
    elif type == "depolarizing":
        noise_model = create_depolarizing_noise()
    elif type == "thermal":
        noise_model = create_thermal_noise()
    elif type == "readout":
        noise_model = create_readout_error()
    elif type == "all":
        noise_model = create_combined_noise()
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

# Create Depolarizing Noise Model
def create_depolarizing_noise(p1=0.001, p2=0.01):
    """
    Create a depolarizing noise model

    @param p1: single qubit depolarizing probability
    @param p2: two qubit depolarizing probability
    @return: NoiseModel
    """
    noise_model = NoiseModel()

    # Single-qubit error
    error_1 = depolarizing_error(p1, 1)

    # Two-qubit error
    error_2 = depolarizing_error(p2, 2)

    # Apply to all basis gates
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    return noise_model


# Create Thermal Relaxation Noise Model
def create_thermal_noise(t1=50e3, t2=70e3, gate_time_1q=50, gate_time_2q=300):
    """
    Create thermal relaxation noise model

    @param t1: T1 relaxation time (in ns)
    @param t2: T2 dephasing time (in ns)
    @param gate_time_1q: 1-qubit gate time (ns)
    @param gate_time_2q: 2-qubit gate time (ns)
    @return: NoiseModel
    """
    noise_model = NoiseModel()

    # Single-qubit thermal error
    error_1 = thermal_relaxation_error(t1, t2, gate_time_1q)

    # Two-qubit thermal error (tensor product)
    error_2 = error_1.tensor(error_1)

    # Apply errors
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    return noise_model


# Create Readout Error
def create_readout_error(prob_0to1=0.02, prob_1to0=0.02):
    """
    Create readout error model

    @param prob_0to1: probability of reading 1 when actual is 0
    @param prob_1to0: probability of reading 0 when actual is 1
    @return: NoiseModel with readout error
    """
    noise_model = NoiseModel()

    readout_error = ReadoutError([
        [1 - prob_0to1, prob_0to1],
        [prob_1to0, 1 - prob_1to0]
    ])

    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


# Combined Noise Model (Recommended for experiments)
def create_combined_noise():
    """
    Combine depolarizing + thermal + readout noise
    """
    noise_model = NoiseModel()

    # Depolarizing
    depol_1 = depolarizing_error(0.001, 1)
    depol_2 = depolarizing_error(0.01, 2)

    # Thermal
    t1 = 50e3
    t2 = 70e3
    error_thermal_1 = thermal_relaxation_error(t1, t2, 50)
    error_thermal_2 = error_thermal_1.tensor(error_thermal_1)

    # Readout
    readout_error = ReadoutError([
        [0.98, 0.02],
        [0.02, 0.98]
    ])

    # Add errors
    noise_model.add_all_qubit_quantum_error(depol_1.compose(error_thermal_1), ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(depol_2.compose(error_thermal_2), ['cx'])
    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model