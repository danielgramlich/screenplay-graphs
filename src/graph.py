"""Construcao do Grafo de Transicao Emocional (G2) e extracao de metricas."""

import json
import logging

import networkx as nx
import numpy as np
import pandas as pd

from src.utils import EMOTIONS, EMOTIONS_LIST, GRAPHS, ensure_dirs

logger = logging.getLogger(__name__)


def build_transition_matrix(emotions: list[str]) -> np.ndarray:
    """Constroi matriz de transicao 6x6 a partir de sequencia de emocoes dominantes.

    Args:
        emotions: Lista de emocoes dominantes por cena (strings).

    Returns:
        Matriz 6x6 numpy (linhas=origem, colunas=destino).
    """
    n = len(EMOTIONS_LIST)
    idx = {e: i for i, e in enumerate(EMOTIONS_LIST)}
    T = np.zeros((n, n), dtype=float)

    for i in range(len(emotions) - 1):
        e_from = emotions[i]
        e_to = emotions[i + 1]
        if e_from in idx and e_to in idx:
            T[idx[e_from]][idx[e_to]] += 1

    return T


def build_digraph(T: np.ndarray) -> nx.DiGraph:
    """Constroi DiGraph do NetworkX a partir da matriz de transicao."""
    G = nx.DiGraph()
    for i, e_from in enumerate(EMOTIONS_LIST):
        G.add_node(e_from)
        for j, e_to in enumerate(EMOTIONS_LIST):
            if T[i][j] > 0:
                G.add_edge(e_from, e_to, weight=T[i][j])
    return G


def compute_metrics(T: np.ndarray, G: nx.DiGraph) -> dict:
    """Extrai todas as metricas do grafo G2.

    Returns:
        Dict com todas as metricas computadas.
    """
    total = T.sum()
    if total == 0:
        return _empty_metrics()

    # Normalizar para probabilidades
    P = T / total

    # Shannon entropy
    nonzero = P[P > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))

    # PageRank estacionario
    try:
        pagerank = nx.pagerank(G, weight="weight")
    except nx.PowerIterationFailedConvergence:
        pagerank = {e: 1.0 / len(EMOTIONS_LIST) for e in EMOTIONS_LIST}

    # Reciprocidade ponderada
    recip_num = 0.0
    for i in range(len(EMOTIONS_LIST)):
        for j in range(len(EMOTIONS_LIST)):
            if i != j:
                recip_num += min(T[i][j], T[j][i])
    recip_denom = T.sum() - np.trace(T)
    weighted_reciprocity = recip_num / recip_denom if recip_denom > 0 else 0.0

    # Autoloop ratio
    autoloop_ratio = np.trace(T) / total

    # Out-strength por no
    idx = {e: i for i, e in enumerate(EMOTIONS_LIST)}
    out_strength = {e: float(T[idx[e]].sum()) for e in EMOTIONS_LIST}

    # Betweenness centrality (pesos invertidos = distancia)
    G_inv = G.copy()
    for u, v, d in G_inv.edges(data=True):
        d["distance"] = 1.0 / d["weight"] if d["weight"] > 0 else float("inf")
    betweenness = nx.betweenness_centrality(G_inv, weight="distance")

    # Indice de polarizacao
    positive = {"joy", "trust", "surprise"}
    negative = {"sadness", "anger", "fear"}
    cross_transitions = 0.0
    for e_from in EMOTIONS_LIST:
        for e_to in EMOTIONS_LIST:
            i, j = idx[e_from], idx[e_to]
            if (e_from in positive and e_to in negative) or \
               (e_from in negative and e_to in positive):
                cross_transitions += T[i][j]
    polarization = cross_transitions / total

    # Indice de recuperacao
    recovery_sum = 0.0
    for neg_e in ["sadness", "fear", "anger"]:
        recovery_sum += T[idx[neg_e]][idx["joy"]]
    recovery_index = recovery_sum / total

    return {
        "entropy": float(entropy),
        "autoloop_ratio": float(autoloop_ratio),
        "weighted_reciprocity": float(weighted_reciprocity),
        "polarization_index": float(polarization),
        "recovery_index": float(recovery_index),
        **{f"pagerank_{e}": float(pagerank.get(e, 0)) for e in EMOTIONS_LIST},
        **{f"out_strength_{e}": float(out_strength[e]) for e in EMOTIONS_LIST},
        **{f"betweenness_{e}": float(betweenness.get(e, 0)) for e in EMOTIONS_LIST},
    }


def _empty_metrics() -> dict:
    """Retorna metricas zeradas para filmes sem transicoes."""
    result = {
        "entropy": 0.0,
        "autoloop_ratio": 0.0,
        "weighted_reciprocity": 0.0,
        "polarization_index": 0.0,
        "recovery_index": 0.0,
    }
    for e in EMOTIONS_LIST:
        result[f"pagerank_{e}"] = 0.0
        result[f"out_strength_{e}"] = 0.0
        result[f"betweenness_{e}"] = 0.0
    return result


def build_all_graphs() -> dict[str, dict]:
    """Constroi G2 para todos os filmes com emocoes anotadas."""
    ensure_dirs()
    results = {}
    emotion_files = sorted(EMOTIONS.glob("*.csv"))

    if not emotion_files:
        logger.warning("Nenhum arquivo de emocoes encontrado em %s", EMOTIONS)
        return results

    for path in emotion_files:
        title = path.stem
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            logger.warning("%s: CSV vazio, pulando", title)
            continue
        if "dominant_emotion" not in df.columns or len(df) < 2:
            logger.warning("%s: dados insuficientes, pulando", title)
            continue
        emotions_seq = df["dominant_emotion"].tolist()

        T = build_transition_matrix(emotions_seq)
        G = build_digraph(T)
        metrics = compute_metrics(T, G)

        result = {
            "transition_matrix": T.tolist(),
            "emotions": EMOTIONS_LIST,
            "num_scenes": len(emotions_seq),
            "metrics": metrics,
        }

        out_path = GRAPHS / f"{title}.json"
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results[title] = result
        logger.info("%s: G2 construido (%d transicoes)", title, int(T.sum()))

    return results
