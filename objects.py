from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass
class SingleRunResult:
    """Result for a single experiment setting and QAOA depth."""

    setting: str
    depth: int
    best_bitstring: str
    best_cut_value: int
    exact_cut_value: int
    approximation_ratio: float
    runtime_seconds: float
    shots: int
    optimizer_success: bool
    optimizer_fun: float
    optimal_parameters: list[float]
    counts: dict[str, int]


@dataclass
class ExperimentSummary:
    """Summary of a complete Max-Cut experiment across settings."""

    graph_edges: list[tuple[int, int]]
    exact_bitstring: str
    exact_cut_value: int
    results: list[SingleRunResult]