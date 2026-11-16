#!/usr/bin/env python3
"""Etapa 32: Tom emocional do protagonista medido com RoBERTa.

Mesmo pipeline do script 24 (character_emotions) mas usando RoBERTa.
Compara VADER vs RoBERTa para emoção de personagem.

Para cada filme:
1. Identifica protagonista (maior força na rede)
2. Extrai falas do protagonista vs elenco
3. Calcula sentimento RoBERTa (em vez de VADER)
4. Gera features + compara com VADER
"""

import sys
import re
import json
import logging
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from statsmodels.stats.multitest import multipletests
from src.utils import (RAW, INTERMEDIATE, FEATURES, FEATURES_CHARACTERS,
                        METADATA, RESULTS, ensure_dirs)
from src.characters.sbsi_core import (
    ExtractionConfig, discover_and_validate_characters,
    split_script_into_scenes, extract_speaker_turns,
)
from src.characters.methods import graphs_turn_adjacency_per_scene
from src.sentiment_roberta import roberta_scenes_batch
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CONFIG = ExtractionConfig()
TARGETS = [("imdb_rating", "IMDb"), ("log_box_office", "log(BoxOffice)")]
GENRES = ["Thriller", "Horror", "Comedy", "Drama", "Action", "Crime"]

ROBERTA_CHAR_DIR = FEATURES / "roberta"
ROBERTA_CHAR_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = ROBERTA_CHAR_DIR / "roberta_char_cache.csv"


def extract_dialogue_by_character(scene_body, name_mapping, canonical_set):
    """Extrai texto de diálogo por personagem numa cena."""
    lines = scene_body.split("\n")
    dialogues = {}
    current_speaker = None
    current_speech = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r"\(.*?\)", "", stripped).strip().upper()
        if cleaned in name_mapping:
            if current_speaker and current_speech:
                canonical = name_mapping.get(current_speaker, current_speaker)
                if canonical in canonical_set:
                    dialogues.setdefault(canonical, []).append(" ".join(current_speech))
            current_speaker = cleaned
            current_speech = []
        elif current_speaker:
            if not (stripped.startswith("(") and stripped.endswith(")")):
                current_speech.append(stripped)

    if current_speaker and current_speech:
        canonical = name_mapping.get(current_speaker, current_speaker)
        if canonical in canonical_set:
            dialogues.setdefault(canonical, []).append(" ".join(current_speech))
    return dialogues


def extract_dialogue_from_moviesum_xml(script_xml, valid_chars):
    """Extrai diálogos por personagem direto do XML MovieSum."""
    scenes_data = []
    raw_scenes = re.findall(r"<scene>(.*?)</scene>", script_xml, re.DOTALL)

    for raw in raw_scenes:
        scene_dialogues = {}
        parts = re.split(r"(<character>.*?</character>)", raw)
        current_char = None

        for part in parts:
            char_match = re.match(r"<character>(.*?)</character>", part)
            if char_match:
                current_char = char_match.group(1).strip().upper()
                current_char = re.sub(r"\s*\(.*?\)\s*", "", current_char).strip()
                if current_char.endswith(":") or current_char not in valid_chars:
                    current_char = None
            elif current_char:
                dialogues = re.findall(r"<dialogue>(.*?)</dialogue>", part, re.DOTALL)
                for d in dialogues:
                    d_clean = re.sub(r"<.*?>", "", d).strip()
                    if d_clean:
                        scene_dialogues.setdefault(current_char, []).append(d_clean)

        scenes_data.append(scene_dialogues)
    return scenes_data


def process_film_imsdb(path):
    """Processa filme IMSDb (formato text com ALL-CAPS)."""
    title = path.stem
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    if len(text) < 500:
        return None

    try:
        name_mapping, canonical_set, _ = discover_and_validate_characters(text, CONFIG)
    except Exception:
        return None

    if len(canonical_set) < 2:
        return None

    try:
        scenes = split_script_into_scenes(text)
    except Exception:
        return None

    if len(scenes) < 3:
        return None

    # Identificar protagonista
    speakers_per_scene = []
    for sc in scenes:
        speakers, _ = extract_speaker_turns(sc.body, name_mapping, canonical_set, CONFIG)
        speakers_per_scene.append(speakers)

    scene_graphs = graphs_turn_adjacency_per_scene(speakers_per_scene)
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

    strengths = {n: sum(d["weight"] for _, _, d in film_G.edges(n, data=True))
                 for n in film_G.nodes()}
    protag = max(strengths, key=strengths.get)

    # Extrair diálogos por cena
    protag_texts = []
    others_texts = []

    for sc in scenes:
        dialogues = extract_dialogue_by_character(sc.body, name_mapping, canonical_set)
        if protag in dialogues:
            protag_texts.append(" ".join(dialogues[protag]))
        if any(c != protag for c in dialogues):
            others_text = " ".join(t for c, speeches in dialogues.items()
                                    if c != protag for t in speeches)
            if others_text.strip():
                others_texts.append(others_text)

    return compute_features(title, protag, protag_texts, others_texts, len(scenes))


def process_film_moviesum(record):
    """Processa filme MovieSum (formato XML)."""
    movie_name = record["movie_name"]
    script_xml = record.get("script", "")
    title = re.sub(r"_\d{4}$", "", movie_name).lower().strip()
    title = re.sub(r"[^a-z0-9\s-]", "", title)
    title = re.sub(r"\s+", "-", title)

    if not title or len(script_xml) < 500:
        return None

    # Identificar personagens
    all_chars = re.findall(r"<character>(.*?)</character>", script_xml)
    char_counts = Counter(c.strip().upper() for c in all_chars)
    valid_chars = {c for c, n in char_counts.items() if n >= 2 and len(c) > 0
                   and not c.endswith(":")}

    if len(valid_chars) < 2:
        return None

    protag = char_counts.most_common(1)[0][0]
    if protag not in valid_chars:
        return None

    # Extrair diálogos
    scenes_data = extract_dialogue_from_moviesum_xml(script_xml, valid_chars)
    if len(scenes_data) < 3:
        return None

    protag_texts = []
    others_texts = []

    for scene_dialogues in scenes_data:
        if protag in scene_dialogues:
            protag_texts.append(" ".join(scene_dialogues[protag]))
        others = " ".join(t for c, speeches in scene_dialogues.items()
                           if c != protag for t in speeches)
        if others.strip():
            others_texts.append(others)

    return compute_features(title, protag, protag_texts, others_texts, len(scenes_data))


def compute_features(title, protag, protag_texts, others_texts, n_scenes):
    """Calcula features emocionais do protagonista com RoBERTa."""
    if len(protag_texts) < 3:
        return None

    # Batch RoBERTa
    protag_results = roberta_scenes_batch(protag_texts)
    protag_sentiments = [r["roberta_compound"] for r in protag_results]

    others_sentiments = []
    if others_texts:
        others_results = roberta_scenes_batch(others_texts)
        others_sentiments = [r["roberta_compound"] for r in others_results]

    p = np.array(protag_sentiments)
    o = np.array(others_sentiments) if others_sentiments else np.array([0.0])

    feats = {"title": title, "protagonist": protag}
    feats["rob_protag_mean"] = float(np.mean(p))
    feats["rob_protag_std"] = float(np.std(p))
    feats["rob_protag_min"] = float(np.min(p))
    feats["rob_protag_max"] = float(np.max(p))
    feats["rob_protag_range"] = float(np.max(p) - np.min(p))
    feats["rob_protag_volatility"] = float(np.std(np.diff(p))) if len(p) > 1 else 0.0
    feats["rob_protag_pct_positive"] = float(np.mean(p > 0.05))
    feats["rob_protag_pct_negative"] = float(np.mean(p < -0.05))
    feats["rob_others_mean"] = float(np.mean(o))
    feats["rob_protag_vs_others"] = float(np.mean(p) - np.mean(o))

    # Arco emocional
    lt = 2 * len(p) // 3
    if 0 < lt < len(p):
        feats["rob_protag_arc"] = float(np.mean(p[lt:]) - np.mean(p[:lt]))
    else:
        feats["rob_protag_arc"] = 0.0

    feats["rob_protag_n_scenes"] = len(protag_texts)
    feats["rob_protag_coverage"] = len(protag_texts) / n_scenes if n_scenes > 0 else 0

    return feats


def main():
    ensure_dirs()

    # Retomada
    done_titles = set()
    rows = []
    if CACHE_PATH.exists():
        old = pd.read_csv(CACHE_PATH)
        done_titles = set(old["title"])
        rows = old.to_dict("records")
        logging.info("Retomando: %d filmes já processados", len(done_titles))

    # Forçar carregamento do modelo
    logging.info("Carregando RoBERTa...")
    _ = roberta_scenes_batch(["test"])
    logging.info("Modelo carregado.")

    t0 = time.time()
    success = len(done_titles)

    # 1. Processar IMSDb (formato original)
    raw_files = sorted(RAW.glob("*.txt"))
    imsdb_to_process = [p for p in raw_files if p.stem not in done_titles]
    logging.info("IMSDb a processar: %d", len(imsdb_to_process))

    for i, path in enumerate(imsdb_to_process):
        feats = process_film_imsdb(path)
        if feats:
            rows.append(feats)
            done_titles.add(feats["title"])
            success += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            logging.info("IMSDb: %d/%d (sucesso=%d, %.1f/min)", i + 1, len(imsdb_to_process), success, rate)

        if (i + 1) % 100 == 0:
            pd.DataFrame(rows).to_csv(CACHE_PATH, index=False)

    pd.DataFrame(rows).to_csv(CACHE_PATH, index=False)
    logging.info("IMSDb concluído: %d filmes", success)

    # 2. Processar MovieSum (formato XML)
    logging.info("Carregando MovieSum...")
    from datasets import load_dataset
    ds = load_dataset("rohitsaxena/MovieSum")

    all_records = []
    for split in ds:
        for record in ds[split]:
            all_records.append(record)

    moviesum_to_process = [r for r in all_records
                           if re.sub(r"\s+", "-", re.sub(r"[^a-z0-9\s-]", "",
                              re.sub(r"_\d{4}$", "", r["movie_name"]).lower().strip())) not in done_titles]
    logging.info("MovieSum a processar: %d", len(moviesum_to_process))

    for i, record in enumerate(moviesum_to_process):
        feats = process_film_moviesum(record)
        if feats:
            rows.append(feats)
            done_titles.add(feats["title"])
            success += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            logging.info("MovieSum: %d/%d (total_sucesso=%d, %.1f/min)",
                         i + 1, len(moviesum_to_process), success, rate)

        if (i + 1) % 100 == 0:
            pd.DataFrame(rows).to_csv(CACHE_PATH, index=False)

    # Salvar final
    feat_df = pd.DataFrame(rows)
    out_path = ROBERTA_CHAR_DIR / "roberta_character_emotions.csv"
    feat_df.to_csv(out_path, index=False)
    pd.DataFrame(rows).to_csv(CACHE_PATH, index=False)

    logging.info("=" * 60)
    logging.info("Total: %d filmes com emoção de protagonista (RoBERTa)", len(feat_df))
    logging.info("Salvo: %s", out_path)

    # ── Comparação rápida VADER vs RoBERTa para protagonista ──
    vader_char_path = FEATURES_CHARACTERS / "character_emotion_features.csv"
    if vader_char_path.exists() and len(feat_df) > 0:
        vader_char = pd.read_csv(vader_char_path)
        meta = pd.read_csv(METADATA / "film_list.csv")
        meta = meta[meta["matched"] == True]

        both = vader_char.merge(feat_df, on="title", suffixes=("_v", "_r"))
        both = both.merge(meta[["source_file", "imdb_rating"]],
                           left_on="title", right_on="source_file", how="inner")
        both = both[both["imdb_rating"].notna()]

        out = []
        out.append("=" * 70)
        out.append("  PROTAGONISTA: VADER vs RoBERTa")
        out.append("=" * 70)
        out.append(f"  Filmes em comum: {len(both)}")

        pairs = [
            ("protag_sentiment_mean", "rob_protag_mean", "Tom médio"),
            ("protag_pct_positive", "rob_protag_pct_positive", "% positivo"),
            ("protag_pct_negative", "rob_protag_pct_negative", "% negativo"),
            ("protag_volatility", "rob_protag_volatility", "Volatilidade"),
            ("protag_arc", "rob_protag_arc", "Arco emocional"),
            ("protag_vs_others", "rob_protag_vs_others", "Contraste vs elenco"),
        ]

        out.append(f"\n  {'Feature':<25s} {'VADER ρ':>10s} {'RoBERTa ρ':>10s} {'Melhor':>10s}")
        out.append(f"  {'-'*58}")

        for v_col, r_col, label in pairs:
            if v_col not in both.columns or r_col not in both.columns:
                continue
            v = both[[v_col, "imdb_rating"]].dropna()
            r = both[[r_col, "imdb_rating"]].dropna()
            rho_v, p_v = stats.spearmanr(v[v_col], v["imdb_rating"]) if len(v) > 30 else (0, 1)
            rho_r, p_r = stats.spearmanr(r[r_col], r["imdb_rating"]) if len(r) > 30 else (0, 1)
            melhor = "RoBERTa" if abs(rho_r) > abs(rho_v) else "VADER"
            sv = "*" if p_v < 0.05 else ""
            sr = "*" if p_r < 0.05 else ""
            out.append(f"  {label:<25s} {rho_v:>+9.3f}{sv} {rho_r:>+9.3f}{sr} {melhor:>10s}")

        report = "\n".join(out)
        print(report)
        (RESULTS / "roberta_character_comparison.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
