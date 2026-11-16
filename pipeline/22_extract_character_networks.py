#!/usr/bin/env python3
"""Etapa 22: Extrair redes de personagens e features de todos os filmes.

Para cada roteiro:
1. Identifica personagens (ALL-CAPS + heurística de diálogo)
2. Agrupa nomes similares (Levenshtein)
3. Extrai turnos de fala por cena
4. Constrói grafo de co-ocorrência por adjacência de turno
5. Calcula métricas estruturais, temporais e de protagonista

Famílias de features:
  F1 (estruturais): n_chars, density, max_strength, betweenness, clustering, modularity
  F2 (temporais): média/max/std de peso e centralização por cena, entropia, tendência
  F3 (protagonista): força relativa, presença, ranking de betweenness
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

from src.utils import RAW, FEATURES_CHARACTERS, ensure_dirs
from src.characters.sbsi_core import (
    ExtractionConfig, discover_and_validate_characters,
    split_script_into_scenes, extract_speaker_turns,
)
from src.characters.methods import graphs_turn_adjacency_per_scene

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CONFIG = ExtractionConfig()


# ═══════════════════════════════════════════════════════════════
# MÉTRICAS DO GRAFO DO FILME
# ═══════════════════════════════════════════════════════════════

def film_graph_features(scene_graphs, speakers_per_scene, canonical_set):
    """Calcula features F1 + F2 + F3 para um filme inteiro.

    Args:
        scene_graphs: lista de nx.Graph (um por cena)
        speakers_per_scene: lista de listas de speakers
        canonical_set: set de nomes canônicos

    Returns:
        dict com todas as features, ou None se dados insuficientes
    """
    # Grafo agregado do filme
    film_G = nx.Graph()
    for G in scene_graphs:
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1)
            if film_G.has_edge(u, v):
                film_G[u][v]["weight"] += w
            else:
                film_G.add_edge(u, v, weight=w)

    n_nodes = film_G.number_of_nodes()
    n_edges = film_G.number_of_edges()

    if n_nodes < 2:
        return None

    feats = {}

    # ── F1: Estruturais (grafo agregado) ──

    feats["F1_n_chars"] = n_nodes
    feats["F1_n_edges"] = n_edges
    feats["F1_density"] = nx.density(film_G)

    # Força (strength = soma dos pesos das arestas)
    strengths = {}
    for n in film_G.nodes():
        strengths[n] = sum(d["weight"] for _, _, d in film_G.edges(n, data=True))
    feats["F1_max_strength"] = max(strengths.values()) if strengths else 0

    # Centralização de força (Freeman)
    if n_nodes > 1 and feats["F1_max_strength"] > 0:
        s_max = feats["F1_max_strength"]
        sum_diff = sum(s_max - s for s in strengths.values())
        feats["F1_strength_centralization"] = sum_diff / ((n_nodes - 1) * s_max)
    else:
        feats["F1_strength_centralization"] = 0

    # Betweenness (usando peso invertido como distância)
    try:
        G_dist = film_G.copy()
        for u, v, d in G_dist.edges(data=True):
            d["dist"] = 1.0 / d["weight"] if d["weight"] > 0 else 1e6
        bw = nx.betweenness_centrality(G_dist, weight="dist")
        feats["F1_max_betweenness"] = max(bw.values()) if bw else 0
    except Exception:
        feats["F1_max_betweenness"] = 0

    # Clustering
    try:
        feats["F1_avg_clustering"] = nx.average_clustering(film_G, weight="weight")
    except Exception:
        feats["F1_avg_clustering"] = 0

    # Modularidade (Louvain)
    try:
        communities = nx.community.louvain_communities(film_G, weight="weight", seed=42)
        feats["F1_modularity"] = nx.community.modularity(film_G, communities, weight="weight")
        feats["F1_n_communities"] = len(communities)
    except Exception:
        feats["F1_modularity"] = 0
        feats["F1_n_communities"] = 1

    # Diâmetro (maior componente conexa)
    try:
        lcc = max(nx.connected_components(film_G), key=len)
        lcc_G = film_G.subgraph(lcc)
        feats["F1_diameter"] = nx.diameter(lcc_G)
    except Exception:
        feats["F1_diameter"] = 0

    # ── F2: Temporais (por cena) ──

    n_scenes = len(scene_graphs)
    feats["F2_n_scenes"] = n_scenes

    # Peso total por cena
    W_series = []
    C_series = []  # centralização por cena
    for G in scene_graphs:
        w = sum(d.get("weight", 1) for _, _, d in G.edges(data=True))
        W_series.append(w)

        # Centralização por cena
        nn = G.number_of_nodes()
        if nn > 1:
            s_per_node = {n: sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True))
                          for n in G.nodes()}
            s_max = max(s_per_node.values()) if s_per_node else 0
            if s_max > 0:
                sum_d = sum(s_max - s for s in s_per_node.values())
                C_series.append(sum_d / ((nn - 1) * s_max))
            else:
                C_series.append(0)
        else:
            C_series.append(0)

    W = np.array(W_series)
    C = np.array(C_series)

    feats["F2_W_mean"] = float(np.mean(W))
    feats["F2_W_max"] = float(np.max(W))
    feats["F2_W_std"] = float(np.std(W))
    feats["F2_C_mean"] = float(np.mean(C))
    feats["F2_C_max"] = float(np.max(C))
    feats["F2_C_std"] = float(np.std(C))

    # Skewness e kurtosis do peso
    from scipy.stats import skew, kurtosis
    feats["F2_W_skew"] = float(skew(W)) if len(W) > 2 else 0
    feats["F2_W_kurtosis"] = float(kurtosis(W)) if len(W) > 2 else 0

    # Tendência da centralização (slope)
    if len(C) > 2:
        from scipy.stats import linregress
        slope, _, _, _, _ = linregress(np.arange(len(C)), C)
        feats["F2_C_trend"] = float(slope)
    else:
        feats["F2_C_trend"] = 0

    # Jaccard cast distance (mudança de elenco entre cenas)
    cast_sets = [set(sp) for sp in speakers_per_scene]
    D_series = []
    for i in range(1, len(cast_sets)):
        a, b = cast_sets[i - 1], cast_sets[i]
        if a or b:
            union = a | b
            inter = a & b
            D_series.append(1 - len(inter) / len(union) if union else 0)
        else:
            D_series.append(0)

    if D_series:
        feats["F2_D_mean"] = float(np.mean(D_series))
        feats["F2_D_max"] = float(np.max(D_series))
        feats["F2_D_std"] = float(np.std(D_series))
        feats["F2_D_peaks_ratio"] = float(sum(1 for d in D_series if d > 0.8) / len(D_series))
    else:
        feats["F2_D_mean"] = feats["F2_D_max"] = feats["F2_D_std"] = feats["F2_D_peaks_ratio"] = 0

    # Entropia do peso
    W_pos = W[W > 0]
    if len(W_pos) > 1:
        p = W_pos / W_pos.sum()
        feats["F2_W_entropy"] = float(-np.sum(p * np.log2(p)))
    else:
        feats["F2_W_entropy"] = 0

    # ── F3: Protagonista ──

    if strengths:
        protag = max(strengths, key=strengths.get)
        protag_strength = strengths[protag]
        total_strength = sum(strengths.values())

        feats["F3_protag_name"] = protag
        feats["F3_protag_strength_frac"] = protag_strength / total_strength if total_strength > 0 else 0

        # Presença do protagonista (em quantas cenas aparece)
        protag_scenes = sum(1 for sp in speakers_per_scene if protag in sp)
        feats["F3_protag_presence"] = protag_scenes / n_scenes if n_scenes > 0 else 0

        # Desvio padrão da força (concentração de interações)
        feats["F3_strength_std"] = float(np.std(list(strengths.values())))

        # Ranking de betweenness do protagonista
        if bw and protag in bw:
            sorted_bw = sorted(bw.values(), reverse=True)
            protag_rank = sorted_bw.index(bw[protag]) + 1
            feats["F3_protag_betw_rank"] = protag_rank / n_nodes
        else:
            feats["F3_protag_betw_rank"] = 0
    else:
        feats["F3_protag_name"] = ""
        feats["F3_protag_strength_frac"] = 0
        feats["F3_protag_presence"] = 0
        feats["F3_strength_std"] = 0
        feats["F3_protag_betw_rank"] = 0

    return feats


# ═══════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def process_film(path):
    """Processa um roteiro e retorna features de rede de personagens."""
    title = path.stem
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"title": title, "parse_ok": False, "error": str(e)}

    if len(text) < 500:
        return {"title": title, "parse_ok": False, "error": "roteiro muito curto"}

    try:
        name_mapping, canonical_set, raw_counts = discover_and_validate_characters(text, CONFIG)
    except Exception as e:
        return {"title": title, "parse_ok": False, "error": f"char_extract: {e}"}

    if len(canonical_set) < 2:
        return {"title": title, "parse_ok": False, "error": f"apenas {len(canonical_set)} personagens"}

    try:
        scenes = split_script_into_scenes(text)
    except Exception as e:
        return {"title": title, "parse_ok": False, "error": f"scene_split: {e}"}

    if len(scenes) < 3:
        return {"title": title, "parse_ok": False, "error": f"apenas {len(scenes)} cenas"}

    # Extrair turnos por cena
    speakers_per_scene = []
    for sc in scenes:
        speakers, _ = extract_speaker_turns(sc.body, name_mapping, canonical_set, CONFIG)
        speakers_per_scene.append(speakers)

    # Construir grafos por cena
    try:
        scene_graphs = graphs_turn_adjacency_per_scene(speakers_per_scene)
    except Exception as e:
        return {"title": title, "parse_ok": False, "error": f"graph_build: {e}"}

    # Calcular features
    feats = film_graph_features(scene_graphs, speakers_per_scene, canonical_set)
    if feats is None:
        return {"title": title, "parse_ok": False, "error": "grafo vazio"}

    feats["title"] = title
    feats["parse_ok"] = True
    feats["error"] = ""
    feats["n_canonical_chars"] = len(canonical_set)

    return feats


def main():
    ensure_dirs()

    raw_files = sorted(RAW.glob("*.txt"))
    logging.info("Filmes a processar: %d", len(raw_files))

    rows = []
    success = 0
    fail = 0

    for i, path in enumerate(raw_files):
        result = process_film(path)
        rows.append(result)

        if result.get("parse_ok"):
            success += 1
        else:
            fail += 1

        if (i + 1) % 50 == 0:
            logging.info("Progresso: %d/%d (sucesso=%d, falha=%d)",
                         i + 1, len(raw_files), success, fail)

    df = pd.DataFrame(rows)

    # Separar features numéricas (sem protag_name, parse_ok, error)
    meta_cols = ["title", "parse_ok", "error", "n_canonical_chars", "F3_protag_name"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Salvar
    out_path = FEATURES_CHARACTERS / "character_network_features.csv"
    df.to_csv(out_path, index=False)

    # Relatório
    logging.info("=" * 60)
    logging.info("  RESULTADO DA EXTRAÇÃO DE REDES DE PERSONAGENS")
    logging.info("=" * 60)
    logging.info("  Total: %d filmes", len(df))
    logging.info("  Sucesso: %d (%.1f%%)", success, success / len(df) * 100)
    logging.info("  Falha: %d (%.1f%%)", fail, fail / len(df) * 100)

    if success > 0:
        ok = df[df["parse_ok"] == True]
        logging.info("\n  Features por filme: %d", len(feature_cols))
        logging.info("  Personagens: mean=%.1f, median=%.0f, max=%.0f",
                     ok["n_canonical_chars"].mean(), ok["n_canonical_chars"].median(),
                     ok["n_canonical_chars"].max())

    if fail > 0:
        errors = df[df["parse_ok"] == False]["error"].value_counts().head(5)
        logging.info("\n  Erros mais comuns:")
        for err, n in errors.items():
            logging.info("    %s: %d", err[:50], n)

    logging.info("\n  Salvo: %s", out_path)


if __name__ == "__main__":
    main()
