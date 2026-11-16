#!/usr/bin/env python3
"""Etapa 26: Extrair personagens e emoção do MovieSum direto do XML.

O formato MovieSum tem personagens marcados com <character> tags,
então não precisa de heurística ALL-CAPS — extração direta.

Para cada filme MovieSum:
1. Extrai personagens e falas do XML
2. Identifica protagonista (mais falas)
3. Calcula VADER nas falas de cada personagem
4. Gera features de rede + emoção por personagem
"""

import sys
import re
import json
import logging
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import networkx as nx
from src.utils import FEATURES_CHARACTERS, METADATA, ensure_dirs
from src.sentiment import vader_scene

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_moviesum_xml(script_xml):
    """Extrai cenas, personagens e falas do XML MovieSum.

    Returns:
        list of dicts: [{heading, characters: [{name, dialogue}], descriptions}]
    """
    text = re.sub(r"</?script>", "", script_xml).strip()
    raw_scenes = re.findall(r"<scene>(.*?)</scene>", text, re.DOTALL)

    scenes = []
    for raw in raw_scenes:
        heading = ""
        h = re.search(r"<stage_direction>(.*?)</stage_direction>", raw)
        if h:
            heading = h.group(1).strip()

        # Extrair sequência de character + dialogue
        # Pattern: <character>NAME</character> seguido opcionalmente de
        # <parenthetical>...</parenthetical> e <dialogue>TEXT</dialogue>
        turns = []
        parts = re.split(r"(<character>.*?</character>)", raw)

        current_char = None
        for part in parts:
            char_match = re.match(r"<character>(.*?)</character>", part)
            if char_match:
                current_char = char_match.group(1).strip().upper()
                # Limpar: remover (V.O.), (O.S.), etc
                current_char = re.sub(r"\s*\(.*?\)\s*", "", current_char).strip()
                # Ignorar se é direção de cena (CUT TO:, FADE IN:, etc)
                if current_char.endswith(":") or current_char in ("CUT TO", "FADE IN",
                    "DISSOLVE TO", "SMASH CUT TO", "TIME CUT", "INTERCUT"):
                    current_char = None
            elif current_char:
                # Buscar diálogos nesta parte
                dialogues = re.findall(r"<dialogue>(.*?)</dialogue>", part, re.DOTALL)
                for d in dialogues:
                    d_clean = re.sub(r"<.*?>", "", d).strip()
                    if d_clean:
                        turns.append({"name": current_char, "dialogue": d_clean})
                # Se chegou nova descrição/cena, resetar
                if "<stage_direction>" in part or "<scene_description>" in part:
                    current_char = None

        scenes.append({"heading": heading, "turns": turns})

    return scenes


def build_character_features(scenes):
    """Constrói features de rede + emoção a partir das cenas parseadas."""
    # Contar falas por personagem
    char_counts = Counter()
    for sc in scenes:
        for turn in sc["turns"]:
            char_counts[turn["name"]] += 1

    # Filtrar personagens com >= 2 falas
    valid_chars = {c for c, n in char_counts.items() if n >= 2 and len(c) > 0}

    if len(valid_chars) < 2:
        return None

    # Construir rede de adjacência de turno por cena
    scene_graphs = []
    speakers_per_scene = []

    for sc in scenes:
        speakers = [t["name"] for t in sc["turns"] if t["name"] in valid_chars]
        speakers_per_scene.append(speakers)

        G = nx.Graph()
        for i in range(len(speakers) - 1):
            a, b = speakers[i], speakers[i + 1]
            if a != b:
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
        scene_graphs.append(G)

    # Grafo agregado
    film_G = nx.Graph()
    for G in scene_graphs:
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1)
            if film_G.has_edge(u, v):
                film_G[u][v]["weight"] += w
            else:
                film_G.add_edge(u, v, weight=w)

    if film_G.number_of_nodes() < 2:
        return None

    # Identificar protagonista
    strengths = {}
    for n in film_G.nodes():
        strengths[n] = sum(d["weight"] for _, _, d in film_G.edges(n, data=True))

    protag = max(strengths, key=strengths.get)
    n_nodes = film_G.number_of_nodes()
    n_edges = film_G.number_of_edges()

    feats = {}

    # ── F1: Estruturais ──
    feats["F1_n_chars"] = n_nodes
    feats["F1_n_edges"] = n_edges
    feats["F1_density"] = nx.density(film_G)
    feats["F1_max_strength"] = max(strengths.values())

    if n_nodes > 1 and feats["F1_max_strength"] > 0:
        s_max = feats["F1_max_strength"]
        feats["F1_strength_centralization"] = sum(s_max - s for s in strengths.values()) / ((n_nodes - 1) * s_max)
    else:
        feats["F1_strength_centralization"] = 0

    try:
        G_dist = film_G.copy()
        for u, v, d in G_dist.edges(data=True):
            d["dist"] = 1.0 / d["weight"] if d["weight"] > 0 else 1e6
        bw = nx.betweenness_centrality(G_dist, weight="dist")
        feats["F1_max_betweenness"] = max(bw.values()) if bw else 0
    except Exception:
        feats["F1_max_betweenness"] = 0
        bw = {}

    try:
        feats["F1_avg_clustering"] = nx.average_clustering(film_G, weight="weight")
    except Exception:
        feats["F1_avg_clustering"] = 0

    try:
        communities = nx.community.louvain_communities(film_G, weight="weight", seed=42)
        feats["F1_modularity"] = nx.community.modularity(film_G, communities, weight="weight")
        feats["F1_n_communities"] = len(communities)
    except Exception:
        feats["F1_modularity"] = 0
        feats["F1_n_communities"] = 1

    # ── F2: Temporais ──
    n_scenes = len(scenes)
    feats["F2_n_scenes"] = n_scenes

    W_series = [sum(d.get("weight", 1) for _, _, d in G.edges(data=True)) for G in scene_graphs]
    W = np.array(W_series)
    feats["F2_W_mean"] = float(np.mean(W))
    feats["F2_W_max"] = float(np.max(W))
    feats["F2_W_std"] = float(np.std(W))

    if len(W) > 1:
        p = W[W > 0]
        if len(p) > 1:
            p = p / p.sum()
            feats["F2_W_entropy"] = float(-np.sum(p * np.log2(p)))
        else:
            feats["F2_W_entropy"] = 0
    else:
        feats["F2_W_entropy"] = 0

    # Jaccard cast distance
    cast_sets = [set(sp) for sp in speakers_per_scene]
    D_series = []
    for i in range(1, len(cast_sets)):
        a, b = cast_sets[i - 1], cast_sets[i]
        if a or b:
            D_series.append(1 - len(a & b) / len(a | b) if (a | b) else 0)
    feats["F2_D_mean"] = float(np.mean(D_series)) if D_series else 0

    # ── F3: Protagonista ──
    total_strength = sum(strengths.values())
    feats["F3_protag_name"] = protag
    feats["F3_protag_strength_frac"] = strengths[protag] / total_strength if total_strength > 0 else 0

    protag_scenes = sum(1 for sp in speakers_per_scene if protag in sp)
    feats["F3_protag_presence"] = protag_scenes / n_scenes if n_scenes > 0 else 0

    feats["F3_strength_std"] = float(np.std(list(strengths.values())))

    if bw and protag in bw:
        sorted_bw = sorted(bw.values(), reverse=True)
        feats["F3_protag_betw_rank"] = (sorted_bw.index(bw[protag]) + 1) / n_nodes
    else:
        feats["F3_protag_betw_rank"] = 0

    # ── Emoção por personagem ──
    protag_sentiments = []
    others_sentiments = []
    n_scenes_total = len(scenes)

    for sc in scenes:
        protag_text = []
        others_text = []
        for turn in sc["turns"]:
            if turn["name"] not in valid_chars:
                continue
            if turn["name"] == protag:
                protag_text.append(turn["dialogue"])
            else:
                others_text.append(turn["dialogue"])

        if protag_text:
            v = vader_scene(" ".join(protag_text))["vader_compound"]
            protag_sentiments.append(v)
        if others_text:
            v = vader_scene(" ".join(others_text))["vader_compound"]
            others_sentiments.append(v)

    if protag_sentiments:
        p_arr = np.array(protag_sentiments)
        feats["protag_sentiment_mean"] = float(np.mean(p_arr))
        feats["protag_sentiment_std"] = float(np.std(p_arr))
        feats["protag_pct_positive"] = float(np.mean(p_arr > 0.05))
        feats["protag_pct_negative"] = float(np.mean(p_arr < -0.05))

        if len(p_arr) > 1:
            feats["protag_volatility"] = float(np.std(np.diff(p_arr)))
        else:
            feats["protag_volatility"] = 0

        # Arco: último terço vs primeiro
        lt = 2 * len(p_arr) // 3
        if lt > 0 and lt < len(p_arr):
            feats["protag_arc"] = float(np.mean(p_arr[lt:]) - np.mean(p_arr[:lt]))
        else:
            feats["protag_arc"] = 0
    else:
        feats["protag_sentiment_mean"] = 0
        feats["protag_sentiment_std"] = 0
        feats["protag_pct_positive"] = 0
        feats["protag_pct_negative"] = 0
        feats["protag_volatility"] = 0
        feats["protag_arc"] = 0

    if others_sentiments:
        feats["others_sentiment_mean"] = float(np.mean(others_sentiments))
        feats["protag_vs_others"] = feats["protag_sentiment_mean"] - feats["others_sentiment_mean"]
    else:
        feats["others_sentiment_mean"] = 0
        feats["protag_vs_others"] = 0

    return feats


def main():
    ensure_dirs()

    logging.info("Carregando MovieSum do HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("rohitsaxena/MovieSum")

    # Carregar mapa de filmes já no corpus
    imdb_map_path = METADATA / "moviesum_imdb_map.json"
    if imdb_map_path.exists():
        imdb_map = json.loads(imdb_map_path.read_text())
    else:
        imdb_map = {}

    # Filmes MovieSum que existem no nosso data/raw/
    from src.utils import RAW
    existing = {p.stem for p in RAW.glob("*.txt")}

    # Carregar features existentes (originais IMSDb)
    existing_char = FEATURES_CHARACTERS / "character_network_features.csv"
    if existing_char.exists():
        old_df = pd.read_csv(existing_char)
        old_ok = set(old_df[old_df["parse_ok"] == True]["title"])
    else:
        old_ok = set()

    logging.info("Features existentes: %d filmes OK", len(old_ok))

    # Processar MovieSum
    all_records = []
    for split in ds:
        for record in ds[split]:
            all_records.append(record)

    logging.info("MovieSum total: %d filmes", len(all_records))

    rows = []
    success = 0
    already_ok = 0

    for i, record in enumerate(all_records):
        movie_name = record["movie_name"]
        script_xml = record.get("script", "")
        imdb_id = record.get("imdb_id", "")

        # Normalizar nome
        filename = re.sub(r"_\d{4}$", "", movie_name).lower().strip()
        filename = re.sub(r"[^a-z0-9\s-]", "", filename)
        filename = re.sub(r"\s+", "-", filename)

        if not filename:
            continue

        # Já tem features boas do extrator original?
        if filename in old_ok:
            already_ok += 1
            continue

        # Processar XML
        try:
            scenes = parse_moviesum_xml(script_xml)
        except Exception:
            continue

        if len(scenes) < 3:
            continue

        total_turns = sum(len(sc["turns"]) for sc in scenes)
        if total_turns < 10:
            continue

        try:
            feats = build_character_features(scenes)
        except Exception:
            continue

        if feats is None:
            continue

        feats["title"] = filename
        feats["source"] = "moviesum"
        feats["imdb_id"] = imdb_id
        feats["parse_ok"] = True
        rows.append(feats)
        success += 1

        if (i + 1) % 200 == 0:
            logging.info("Progresso: %d/%d, sucesso=%d", i + 1, len(all_records), success)

    new_df = pd.DataFrame(rows)
    logging.info("MovieSum extraídos: %d filmes com features", len(new_df))

    # Merge com existentes
    if existing_char.exists():
        old_df = pd.read_csv(existing_char)
        # Manter originais OK + adicionar MovieSum novos
        old_ok_df = old_df[old_df["parse_ok"] == True].copy()
        old_ok_df["source"] = "imsdb"

        # Garantir mesmas colunas
        all_cols = list(set(old_ok_df.columns) | set(new_df.columns))
        for col in all_cols:
            if col not in old_ok_df.columns:
                old_ok_df[col] = None
            if col not in new_df.columns:
                new_df[col] = None

        # Deduplicar por título
        combined = pd.concat([old_ok_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="title", keep="first")
    else:
        combined = new_df

    # Salvar
    out_path = FEATURES_CHARACTERS / "character_network_features.csv"
    combined.to_csv(out_path, index=False)

    logging.info("=" * 60)
    logging.info("  RESULTADO FINAL")
    logging.info("=" * 60)
    logging.info("  Originais (IMSDb) mantidos: %d", already_ok)
    logging.info("  MovieSum novos: %d", success)
    logging.info("  Total combinado: %d", len(combined))
    logging.info("  Salvo: %s", out_path)


if __name__ == "__main__":
    main()
