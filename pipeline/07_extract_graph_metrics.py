#!/usr/bin/env python3
"""Script 12: Grafos refinados de valencia discretizada (v5, v3, v3q, v3_act3)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np
import pandas as pd
import networkx as nx

from src.utils import EMOTIONS, PROCESSED, ensure_dirs, FEATURES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# === Discretizacao ===

def to_v5(c):
    if c >= 0.3: return "v5_high"
    if c >= 0.05: return "v5_pos"
    if c > -0.05: return "v5_neu"
    if c > -0.3: return "v5_neg"
    return "v5_low"

V5_STATES = ["v5_high", "v5_pos", "v5_neu", "v5_neg", "v5_low"]

def to_v3(c):
    if c > 0.05: return "pos"
    if c < -0.05: return "neg"
    return "neu"

V3_STATES = ["pos", "neu", "neg"]

def to_v3q(c, q33, q66):
    """Tercis relativos ao filme."""
    if c >= q66: return "high"
    if c <= q33: return "low"
    return "mid"

V3Q_STATES = ["high", "mid", "low"]


# === Grafo ===

def build_tm(seq, states):
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    T = np.zeros((n, n))
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a in idx and b in idx:
            T[idx[a]][idx[b]] += 1
    return T


def graph_metrics(T, states, prefix):
    total = T.sum()
    if total == 0:
        return {}
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    P = T / total
    nz = P[P > 0]

    result = {
        f"{prefix}_entropy": float(-np.sum(nz * np.log2(nz))),
        f"{prefix}_autoloop": float(np.trace(T) / total),
    }

    # Reciprocidade
    recip = sum(min(T[i][j], T[j][i]) for i in range(n) for j in range(n) if i != j)
    denom = total - np.trace(T)
    result[f"{prefix}_reciprocity"] = float(recip / denom) if denom > 0 else 0.0

    # PageRank
    G = nx.DiGraph()
    for s in states:
        G.add_node(s)
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            if T[i][j] > 0:
                G.add_edge(s1, s2, weight=T[i][j])
    try:
        pr = nx.pagerank(G, weight="weight")
    except:
        pr = {s: 1.0 / n for s in states}

    for s in states:
        result[f"{prefix}_pr_{s}"] = float(pr.get(s, 0))

    # Betweenness
    G_inv = G.copy()
    for u, v, d in G_inv.edges(data=True):
        d["dist"] = 1.0 / d["weight"] if d["weight"] > 0 else 1e6
    bw = nx.betweenness_centrality(G_inv, weight="dist")
    for s in states:
        result[f"{prefix}_bw_{s}"] = float(bw.get(s, 0))

    # Specific transitions (neg->pos for "relief")
    if "neg" in idx and "pos" in idx:
        result[f"{prefix}_neg_to_pos"] = float(T[idx["neg"]][idx["pos"]] / total)
        result[f"{prefix}_pos_to_neg"] = float(T[idx["pos"]][idx["neg"]] / total)

    return result


def compute_all_graph_variants(valence: np.ndarray) -> dict:
    n = len(valence)
    if n < 6:
        return {}

    feats = {}

    # V5 original
    seq_v5 = [to_v5(c) for c in valence]
    T_v5 = build_tm(seq_v5, V5_STATES)
    feats.update(graph_metrics(T_v5, V5_STATES, "gv5"))

    # V3 fixo
    seq_v3 = [to_v3(c) for c in valence]
    T_v3 = build_tm(seq_v3, V3_STATES)
    feats.update(graph_metrics(T_v3, V3_STATES, "gv3"))

    # V3 quantis (tercis relativos)
    q33, q66 = np.percentile(valence, [33.3, 66.7])
    seq_v3q = [to_v3q(c, q33, q66) for c in valence]
    T_v3q = build_tm(seq_v3q, V3Q_STATES)
    feats.update(graph_metrics(T_v3q, V3Q_STATES, "gv3q"))

    # V3 fixo so do ultimo terco
    act3 = valence[2 * n // 3:]
    if len(act3) >= 4:
        seq_v3a3 = [to_v3(c) for c in act3]
        T_v3a3 = build_tm(seq_v3a3, V3_STATES)
        feats.update(graph_metrics(T_v3a3, V3_STATES, "gv3a3"))

    return feats


def main():
    ensure_dirs()
    rows = []
    emotion_files = sorted(EMOTIONS.glob("*.csv"))

    for path in emotion_files:
        title = path.stem
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if "vader_compound" not in df.columns or len(df) < 6:
            continue

        v = df["vader_compound"].values
        feats = compute_all_graph_variants(v)
        if feats:
            rows.append({"title": title, **feats})

    result = pd.DataFrame(rows)
    out = FEATURES / "emotional_graphs" / "refined_graph_features.csv"
    result.to_csv(out, index=False)
    logging.info("Salvo: %d filmes, %d colunas -> %s", len(result), len(result.columns), out)

    # Summary
    variants = set(c.split("_")[0] for c in result.columns if c != "title")
    print(f"\n{len(result)} filmes")
    for v in sorted(variants):
        cols = [c for c in result.columns if c.startswith(v)]
        print(f"  {v}: {len(cols)} features")


if __name__ == "__main__":
    main()
