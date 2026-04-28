import networkx as nx
import random


def build_random_graph(
    n_nodes: int,
    edge_prob: float = 0.5,
    seed: int | None = None,
    ensure_connected: bool = True,
) -> nx.Graph:
    """Generate a random Erdős–Rényi graph.

    Args:
        n_nodes: Number of nodes.
        edge_prob: Probability of edge creation (0–1).
        seed: Random seed for reproducibility.
        ensure_connected: If True, regenerate until graph is connected.

    Returns:
        Random graph.
    """
    rng = seed

    while True:
        g = nx.erdos_renyi_graph(n=n_nodes, p=edge_prob, seed=rng)

        if not ensure_connected:
            return g

        if nx.is_connected(g):
            return g

        # change seed slightly to avoid infinite loop
        if rng is not None:
            rng += 1


def build_connected_random_graph(
    n_nodes: int,
    edge_prob: float = 0.7,
    seed: int | None = None,
) -> nx.Graph:
    """Generate a connected random graph.

    Strategy:
        1. Create a random spanning tree (ensures connectivity)
        2. Add extra edges with probability edge_prob

    Args:
        n_nodes: Number of nodes.
        edge_prob: Probability of adding extra edges.
        seed: Random seed.

    Returns:
        Connected graph.
    """
    rng = random.Random(seed)

    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))

    # Step 1: random spanning tree
    nodes = list(range(n_nodes))
    rng.shuffle(nodes)

    for i in range(1, n_nodes):
        u = nodes[i]
        v = rng.choice(nodes[:i])
        g.add_edge(u, v)

    # Step 2: add random edges
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if not g.has_edge(i, j) and rng.random() < edge_prob:
                g.add_edge(i, j)

    return g


def build_weighted_random_graph(
    n_nodes: int,
    edge_prob: float = 0.4,
    weight_range: tuple = (0.5, 2.0),
    seed: int | None = None,
) -> nx.Graph:
    """Generate a weighted random graph."""
    rng = random.Random(seed)

    g = build_connected_random_graph(
        n_nodes=n_nodes,
        edge_prob=edge_prob,
        seed=seed,
    )

    for u, v in g.edges():
        g[u][v]["weight"] = rng.uniform(*weight_range)

    return g