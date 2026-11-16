
"""
methods.py

Graph construction methods ("baselines") to compare with the paper's main rule.

All methods use the SAME:
- scene split,
- character normalization,
- speaker turn sequence extraction.

This is critical for fairness: differences should come only from the edge rule / temporal aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import networkx as nx
from collections import Counter, defaultdict


@dataclass(frozen=True)
class MethodSpec:
    id: str
    label: str
    family: str
    temporal_unit: str
    description: str
    params: Dict[str, object]


# ==========================================================
#   HELPERS
# ==========================================================

def _add_weighted_edge(G: nx.Graph, u: str, v: str, w: float) -> None:
    if u == v:
        return
    if G.has_edge(u, v):
        G[u][v]["weight"] += float(w)
    else:
        G.add_edge(u, v, weight=float(w))


def _sum_graphs(graphs: Sequence[nx.Graph]) -> nx.Graph:
    out = nx.Graph()
    for g in graphs:
        for u, v, data in g.edges(data=True):
            _add_weighted_edge(out, u, v, data.get("weight", 1.0))
    return out


def aggregate_over_all_scenes(scene_graphs: Sequence[nx.Graph]) -> nx.Graph:
    """Aggregate a list of per-scene graphs into one global graph (edge weights summed)."""
    return _sum_graphs(scene_graphs)


# ==========================================================
#   PER-SCENE EDGE RULES
# ==========================================================

def graphs_turn_adjacency_per_scene(speakers_per_scene: Sequence[Sequence[str]]) -> List[nx.Graph]:
    """
    Paper main rule (adjacency of turns):

    For each scene t, for each consecutive pair (a_l, a_{l+1}), if distinct:
      w_{ij}(t) += 1
    """
    graphs: List[nx.Graph] = []
    for speakers in speakers_per_scene:
        G = nx.Graph()
        for i in range(len(speakers) - 1):
            a, b = speakers[i], speakers[i + 1]
            if a != b:
                _add_weighted_edge(G, a, b, 1.0)
        graphs.append(G)
    return graphs


def graphs_copresence_min_turns_per_scene(turn_counts_per_scene: Sequence[Counter]) -> List[nx.Graph]:
    """
    Paper variant A (copresence clique, weighted):

    For a scene t with cast V_t and c_i(t) = number of turns of i in scene t:
      for each pair i != j in V_t:
        w^A_ij(t) = min(c_i(t), c_j(t))
    """
    graphs: List[nx.Graph] = []
    for counts in turn_counts_per_scene:
        G = nx.Graph()
        cast = sorted(counts.keys())
        for i in range(len(cast)):
            for j in range(i + 1, len(cast)):
                u, v = cast[i], cast[j]
                w = min(counts[u], counts[v])
                if w > 0:
                    _add_weighted_edge(G, u, v, float(w))
        graphs.append(G)
    return graphs


def graphs_copresence_binary_per_scene(turn_counts_per_scene: Sequence[Counter]) -> List[nx.Graph]:
    """
    Copresence clique (binary):

    For each scene t, connect all pairs of speakers with weight 1.
    """
    graphs: List[nx.Graph] = []
    for counts in turn_counts_per_scene:
        G = nx.Graph()
        cast = sorted(counts.keys())
        for i in range(len(cast)):
            for j in range(i + 1, len(cast)):
                _add_weighted_edge(G, cast[i], cast[j], 1.0)
        graphs.append(G)
    return graphs


def graphs_turn_window_k_per_scene(speakers_per_scene: Sequence[Sequence[str]], k: int = 2) -> List[nx.Graph]:
    """
    Paper variant B (k=2 with decay):
      For d in {1,2} connect {a_l, a_{l+d}} (when distinct) with increment 1/d.

    Generalized to any k>=1:
      For d in {1..k}: add increment 1/d.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    graphs: List[nx.Graph] = []
    for speakers in speakers_per_scene:
        G = nx.Graph()
        n = len(speakers)
        for l in range(n):
            for d in range(1, k + 1):
                r = l + d
                if r >= n:
                    break
                a, b = speakers[l], speakers[r]
                if a != b:
                    _add_weighted_edge(G, a, b, 1.0 / float(d))
        graphs.append(G)
    return graphs


# ==========================================================
#   TEMPORAL AGGREGATIONS
# ==========================================================

def graphs_cumulative(scene_graphs: Sequence[nx.Graph]) -> List[nx.Graph]:
    """Rochat-like cumulative growth: snapshot t aggregates scenes 1..t."""
    cum = nx.Graph()
    out: List[nx.Graph] = []
    for g in scene_graphs:
        for u, v, data in g.edges(data=True):
            _add_weighted_edge(cum, u, v, data.get("weight", 1.0))
        out.append(cum.copy())
    return out


def graphs_sliding_window(scene_graphs: Sequence[nx.Graph], window_size: int = 5) -> List[nx.Graph]:
    """
    Tang-style snapshots: each snapshot t aggregates the last 'window_size' scenes.
    window_size=1 equals the raw per-scene graphs.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    out: List[nx.Graph] = []
    for t in range(len(scene_graphs)):
        start = max(0, t - window_size + 1)
        out.append(_sum_graphs(scene_graphs[start : t + 1]))
    return out


def graphs_event_decay_end_of_scene(
    speakers_per_scene: Sequence[Sequence[str]],
    *,
    half_life_events: int = 50,
) -> List[nx.Graph]:
    """
    Jones-style (approx.) event-based dynamic network with exponential forgetting.

    We treat each adjacency event between consecutive turns as an "interaction event" at event-time τ.
    Edge weights decay exponentially with distance in events.

    Discrete formulation using half-life H (in number of events):
      alpha = 0.5 ** (1/H)
      weight_ij(τ) = alpha * weight_ij(τ-1) + 1_{event ij at τ}

    Implementation uses lazy decay per edge:
      store (w, last_tau) for each edge; when updating or emitting snapshot at τ, apply alpha^(τ-last_tau).

    Snapshots are emitted at the end of each scene (after processing its events).
    """
    if half_life_events <= 0:
        raise ValueError("half_life_events must be > 0")
    alpha = 0.5 ** (1.0 / float(half_life_events))

    edge_state: Dict[Tuple[str, str], Tuple[float, int]] = {}  # (u,v)->(w,last_tau)
    tau = 0

    def _decay_to_now(w: float, last_tau: int, now_tau: int) -> float:
        if now_tau <= last_tau:
            return w
        return w * (alpha ** (now_tau - last_tau))

    def _update_edge(u: str, v: str, now_tau: int) -> None:
        nonlocal edge_state
        if u == v:
            return
        key = (u, v) if u < v else (v, u)
        if key in edge_state:
            w, last = edge_state[key]
            w = _decay_to_now(w, last, now_tau)
            w += 1.0
            edge_state[key] = (w, now_tau)
        else:
            edge_state[key] = (1.0, now_tau)

    def _emit_snapshot(now_tau: int) -> nx.Graph:
        G = nx.Graph()
        for (u, v), (w, last) in edge_state.items():
            w_now = _decay_to_now(w, last, now_tau)
            if w_now > 0:
                G.add_edge(u, v, weight=float(w_now))
        return G

    snapshots: List[nx.Graph] = []
    for speakers in speakers_per_scene:
        # process events in this scene
        for i in range(len(speakers) - 1):
            a, b = speakers[i], speakers[i + 1]
            if a == b:
                continue
            tau += 1
            _update_edge(a, b, tau)

        # snapshot at end of scene (after all events)
        snapshots.append(_emit_snapshot(tau))

    return snapshots


def graphs_multilayer_collapse(
    layer_a: Sequence[nx.Graph],
    layer_b: Sequence[nx.Graph],
    *,
    beta_a: float = 1.0,
    beta_b: float = 1.0,
) -> List[nx.Graph]:
    """
    Kivelä / Mourchid-inspired (simplified) multilayer collapse.

    We build two layers (e.g., dialogue adjacency and copresence) and collapse them:
      w = beta_a * w_a + beta_b * w_b
    """
    if len(layer_a) != len(layer_b):
        raise ValueError("layers must have the same number of snapshots")

    out: List[nx.Graph] = []
    for ga, gb in zip(layer_a, layer_b):
        G = nx.Graph()
        for u, v, data in ga.edges(data=True):
            _add_weighted_edge(G, u, v, beta_a * float(data.get("weight", 1.0)))
        for u, v, data in gb.edges(data=True):
            _add_weighted_edge(G, u, v, beta_b * float(data.get("weight", 1.0)))
        out.append(G)
    return out


# ==========================================================
#   METHOD REGISTRY
# ==========================================================

def get_default_methods(*, tang_window_size: int = 5, half_life_events: int = 50) -> List[MethodSpec]:
    """
    Registry of baseline methods aligned with the paper's related work categories.

    You can add/remove methods without touching the runner.
    """
    return [
        MethodSpec(
            id="turn_adj_scene",
            label="Adjacência de turnos (por cena)",
            family="Proposto (regra principal)",
            temporal_unit="scene",
            description="Conecta pares consecutivos distintos e soma 1 ao peso.",
            params={},
        ),
        MethodSpec(
            id="copresence_min_turns_scene",
            label="Copresença min(turnos) (por cena)",
            family="Agarwal et al. (baseline típico de copresença)",
            temporal_unit="scene",
            description="Clique do elenco da cena com w_ij(t)=min(c_i(t), c_j(t)).",
            params={},
        ),
        MethodSpec(
            id="turn_window_k2_scene",
            label="Janela de turnos k=2 com decaimento (por cena)",
            family="Variação de interação (janela curta)",
            temporal_unit="scene",
            description="Para d∈{1,2}, conecta {a_l, a_{l+d}} com incremento 1/d.",
            params={"k": 2},
        ),
        MethodSpec(
            id="aggregate_film_copresence_min_turns",
            label="Grafo agregado do filme (copresença min-turnos)",
            family="Moretti (grafo agregado – aproximação via copresença)",
            temporal_unit="film",
            description="Soma a copresença min-turnos ao longo de todas as cenas, gerando um grafo único.",
            params={},
        ),
        MethodSpec(
            id="cumulative_turn_adj",
            label="Cumulativo até a cena t (adjacência)",
            family="Rochat (evolução / crescimento cumulativo)",
            temporal_unit="scene_cumulative",
            description="Snapshot t agrega cenas 1..t (pesos somados) a partir da regra de adjacência.",
            params={},
        ),
        MethodSpec(
            id=f"tang_window_w{tang_window_size}",
            label=f"Janela deslizante de cenas (w={tang_window_size})",
            family="Tang et al. (snapshots por janela)",
            temporal_unit="scene_window",
            description="Snapshot t agrega as últimas w cenas (pesos somados) a partir da regra de adjacência.",
            params={"window_size": tang_window_size},
        ),
        MethodSpec(
            id=f"event_decay_half{half_life_events}",
            label=f"Rede dinâmica por eventos com decaimento (half-life={half_life_events} eventos)",
            family="Jones et al. (dinâmica por eventos – aproximação)",
            temporal_unit="scene_event_decay",
            description="Eventos são alternâncias consecutivas; pesos decaem exponencialmente; snapshot no fim de cada cena.",
            params={"half_life_events": half_life_events},
        ),
        MethodSpec(
            id="multilayer_collapse_dialogue+copresence",
            label="Multicamadas (colapso: diálogo + copresença)",
            family="Kivelä / Mourchid (multicamadas – aproximação textual)",
            temporal_unit="scene_multilayer",
            description="Colapsa duas camadas: adjacência (diálogo) + copresença min-turnos.",
            params={"beta_dialogue": 1.0, "beta_copresence": 1.0},
        ),
    ]
