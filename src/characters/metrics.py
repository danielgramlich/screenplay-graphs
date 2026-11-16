
"""
metrics.py

Metric computation for a sequence of weighted undirected graphs.

Matches the paper's operational definitions:
- W: total edge weight.
- D: cast Jaccard distance (provided externally; depends only on V_t).
- C: Freeman-style centralization using strength, normalized by (|V|-1)*s_max.
- L and Delta: computed on the largest connected component (LCC), unweighted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import networkx as nx


@dataclass(frozen=True)
class MetricRow:
    t: int
    W: float
    D: float
    C: float
    L: float
    Delta: float
    n_nodes: int
    n_edges: int


def strength_centralization(G: nx.Graph) -> float:
    """
    Freeman-style centralization (operational, as in the paper):

    For strengths s_i and s_max:
      S = sum_i (s_max - s_i)
      C = S / ((n-1)*s_max), if n>2 and s_max>0, else 0.
    """
    n = G.number_of_nodes()
    if n <= 2:
        return 0.0
    strengths = [float(s) for _, s in G.degree(weight="weight")]
    if not strengths:
        return 0.0
    s_max = max(strengths)
    if s_max <= 0:
        return 0.0
    S = sum((s_max - s) for s in strengths)
    denom = (n - 1) * s_max
    return float(S / denom) if denom > 0 else 0.0


def lcc_diameter_and_avg_path(G: nx.Graph) -> Tuple[float, float]:
    """
    Compute diameter (Delta) and average path length (L) on the largest connected component (LCC),
    using unweighted distances (topological), as described in the paper.

    If the LCC has <2 vertices, returns (0,0).
    """
    if G.number_of_nodes() <= 1:
        return 0.0, 0.0

    comps = list(nx.connected_components(G))
    if not comps:
        return 0.0, 0.0

    lcc_nodes = max(comps, key=len)
    if len(lcc_nodes) < 2:
        return 0.0, 0.0

    lcc = G.subgraph(lcc_nodes).copy()

    try:
        diam = float(nx.diameter(lcc))
        avgp = float(nx.average_shortest_path_length(lcc))
    except Exception:
        return 0.0, 0.0

    return diam, avgp


def total_edge_weight(G: nx.Graph) -> float:
    return float(G.size(weight="weight")) if G.number_of_edges() > 0 else 0.0


def compute_metrics_for_scene_aligned_series(
    graphs: Sequence[nx.Graph],
    cast_jaccard: Sequence[float],
) -> List[MetricRow]:
    """
    Compute metrics for a sequence of graphs where index aligns with scenes (1..T).
    """
    if len(graphs) != len(cast_jaccard):
        raise ValueError("graphs and cast_jaccard must have the same length for scene-aligned methods")

    rows: List[MetricRow] = []
    for i, (G, D) in enumerate(zip(graphs, cast_jaccard), start=1):
        W = total_edge_weight(G)
        C = strength_centralization(G)
        Delta, L = lcc_diameter_and_avg_path(G)
        rows.append(
            MetricRow(
                t=i,
                W=W,
                D=float(D),
                C=C,
                L=L,
                Delta=Delta,
                n_nodes=G.number_of_nodes(),
                n_edges=G.number_of_edges(),
            )
        )
    return rows


def compute_metrics_for_single_graph(G: nx.Graph) -> MetricRow:
    """Metrics for a single (film-level) graph. D is set to 0.0 by convention."""
    W = total_edge_weight(G)
    C = strength_centralization(G)
    Delta, L = lcc_diameter_and_avg_path(G)
    return MetricRow(
        t=1,
        W=W,
        D=0.0,
        C=C,
        L=L,
        Delta=Delta,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
    )
