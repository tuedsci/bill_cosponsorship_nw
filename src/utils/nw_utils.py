"""
Module for network utilities.

Author: Tue Nguyen
"""

import networkx as nx
import pandas as pd


def summarize_network(G: nx.Graph, as_df=True) -> dict | pd.DataFrame:
    """
    Summarize a network with optimized computations.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    connected = nx.is_connected(G)
    components = list(nx.connected_components(G))

    metrics = {
        # Basic info
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": nx.density(G),
        "avg_degree": 2 * n_edges / n_nodes,
        # Connectivity
        "is_connected": connected,
        "n_cc": len(components),
        "largest_cc": len(max(components, key=len)),
        # Clustering
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "triangles": sum(nx.triangles(G).values()) / 3,
        # Mixing
        "degree_assortativity": nx.degree_assortativity_coefficient(G),
        # Distance
        "diameter": nx.diameter(G) if connected else None,
        "avg_shortest_path": nx.average_shortest_path_length(G) if connected else None,
    }

    if as_df:
        return pd.DataFrame(metrics, index=[0])

    return metrics
