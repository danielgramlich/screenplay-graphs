#!/usr/bin/env python3
"""Etapa 24: Cruzar personagens com emoção — tom do protagonista vs sucesso.

Para cada filme:
1. Identifica protagonista (maior força na rede)
2. Extrai falas do protagonista vs resto do elenco
3. Calcula sentimento VADER das falas de cada um
4. Gera features: tom do protagonista, contraste com elenco, evolução

Pergunta central:
  O tom emocional do protagonista está relacionado com o sucesso do filme?
"""

import sys
import re
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from src.utils import RAW, FEATURES, FEATURES_CHARACTERS, METADATA, RESULTS, ensure_dirs
from src.characters.sbsi_core import (
    ExtractionConfig, discover_and_validate_characters,
    split_script_into_scenes, extract_speaker_turns,
)
from src.characters.methods import graphs_turn_adjacency_per_scene
from src.sentiment import vader_scene
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CONFIG = ExtractionConfig()

TARGETS = [
    ("imdb_rating", "IMDb"),
    ("log_box_office", "log(BoxOffice)"),
    ("log_revenue_multiple", "log(RevMultiple)"),
]
GENRES = ["Thriller", "Horror", "Comedy", "Drama", "Action", "Crime"]


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


def identify_protagonist(scenes, name_mapping, canonical_set):
    """Identifica protagonista pela força na rede de adjacência de turno."""
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

    if film_G.number_of_nodes() == 0:
        return None, {}

    strengths = {n: sum(d["weight"] for _, _, d in film_G.edges(n, data=True))
                 for n in film_G.nodes()}
    protag = max(strengths, key=strengths.get)
    return protag, strengths


def process_film(path):
    """Extrai features de emoção por personagem para um filme."""
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

    protag, strengths = identify_protagonist(scenes, name_mapping, canonical_set)
    if not protag:
        return None

    # Extrair diálogos por personagem por cena
    protag_sentiments = []
    others_sentiments = []
    protag_sentiments_by_scene = []
    n_scenes = len(scenes)
    protag_scene_count = 0

    for sc in scenes:
        dialogues = extract_dialogue_by_character(sc.body, name_mapping, canonical_set)

        # Protagonista
        if protag in dialogues:
            protag_text = " ".join(dialogues[protag])
            if len(protag_text.strip()) > 10:
                v = vader_scene(protag_text)["vader_compound"]
                protag_sentiments.append(v)
                protag_sentiments_by_scene.append(v)
                protag_scene_count += 1
            else:
                protag_sentiments_by_scene.append(None)
        else:
            protag_sentiments_by_scene.append(None)

        # Outros personagens
        for char, speeches in dialogues.items():
            if char != protag:
                char_text = " ".join(speeches)
                if len(char_text.strip()) > 10:
                    v = vader_scene(char_text)["vader_compound"]
                    others_sentiments.append(v)

    if len(protag_sentiments) < 3:
        return None

    p_arr = np.array(protag_sentiments)
    o_arr = np.array(others_sentiments) if others_sentiments else np.array([0.0])

    # Sentimento no último terço
    valid_scenes = [(i, v) for i, v in enumerate(protag_sentiments_by_scene) if v is not None]
    lt_start = 2 * n_scenes // 3
    protag_lt = [v for i, v in valid_scenes if i >= lt_start]
    protag_first = [v for i, v in valid_scenes if i < lt_start]

    feats = {"title": title, "protagonist": protag}

    # ── Tom do protagonista ──
    feats["protag_sentiment_mean"] = float(np.mean(p_arr))
    feats["protag_sentiment_std"] = float(np.std(p_arr))
    feats["protag_sentiment_min"] = float(np.min(p_arr))
    feats["protag_sentiment_max"] = float(np.max(p_arr))
    feats["protag_sentiment_range"] = float(np.max(p_arr) - np.min(p_arr))

    # Volatilidade emocional do protagonista
    if len(p_arr) > 1:
        feats["protag_volatility"] = float(np.std(np.diff(p_arr)))
    else:
        feats["protag_volatility"] = 0.0

    # ── Contraste protagonista vs elenco ──
    feats["others_sentiment_mean"] = float(np.mean(o_arr))
    feats["protag_vs_others"] = float(np.mean(p_arr) - np.mean(o_arr))
    feats["protag_vs_others_abs"] = float(abs(np.mean(p_arr) - np.mean(o_arr)))

    # ── Evolução emocional do protagonista ──
    if protag_lt and protag_first:
        feats["protag_ending_mean"] = float(np.mean(protag_lt))
        feats["protag_first_mean"] = float(np.mean(protag_first))
        feats["protag_arc"] = float(np.mean(protag_lt) - np.mean(protag_first))
    else:
        feats["protag_ending_mean"] = feats.get("protag_sentiment_mean", 0)
        feats["protag_first_mean"] = feats.get("protag_sentiment_mean", 0)
        feats["protag_arc"] = 0.0

    # ── Presença emocional ──
    feats["protag_n_scenes_with_dialogue"] = protag_scene_count
    feats["protag_dialogue_coverage"] = protag_scene_count / n_scenes if n_scenes > 0 else 0
    feats["protag_pct_positive"] = float(np.mean(p_arr > 0.05))
    feats["protag_pct_negative"] = float(np.mean(p_arr < -0.05))
    feats["protag_pct_neutral"] = float(np.mean((p_arr >= -0.05) & (p_arr <= 0.05)))

    return feats


def main():
    ensure_dirs()
    raw_files = sorted(RAW.glob("*.txt"))
    logging.info("Processando %d filmes...", len(raw_files))

    rows = []
    success = 0
    for i, path in enumerate(raw_files):
        result = process_film(path)
        if result:
            rows.append(result)
            success += 1
        if (i + 1) % 100 == 0:
            logging.info("Progresso: %d/%d (sucesso=%d)", i + 1, len(raw_files), success)

    feat_df = pd.DataFrame(rows)
    logging.info("Sucesso: %d/%d filmes", len(feat_df), len(raw_files))

    # Salvar features
    out_path = FEATURES_CHARACTERS / "character_emotion_features.csv"
    feat_df.to_csv(out_path, index=False)

    # ═══════════════════════════════════════════════════════
    # ANÁLISE
    # ═══════════════════════════════════════════════════════

    # Merge com metadata
    meta = pd.read_csv(METADATA / "film_list.csv")
    meta = meta[meta["matched"] == True]
    df = feat_df.merge(meta[["source_file", "imdb_rating", "box_office", "year", "genre"]],
                        left_on="title", right_on="source_file", how="inner")
    df["box_office_clean"] = df["box_office"].astype(str).str.replace(r"[\$,]", "", regex=True)
    df["box_office_clean"] = pd.to_numeric(df["box_office_clean"], errors="coerce")
    df["log_box_office"] = np.log1p(df["box_office_clean"])

    roi_path = FEATURES / "analysis_dataset_with_roi.csv"
    if roi_path.exists():
        roi = pd.read_csv(roi_path)
        if "log_revenue_multiple" in roi.columns:
            df["log_revenue_multiple"] = df["title"].map(roi.set_index("title")["log_revenue_multiple"])

    out = []
    out.append("=" * 70)
    out.append("  TOM EMOCIONAL DO PROTAGONISTA vs SUCESSO")
    out.append("=" * 70)
    out.append(f"\n  Filmes analisados: {len(df)}")

    # Features para testar
    test_feats = [
        ("protag_sentiment_mean", "Tom médio do protagonista"),
        ("protag_sentiment_std", "Variação emocional do protagonista"),
        ("protag_volatility", "Volatilidade do protagonista"),
        ("protag_vs_others", "Contraste protagonista vs elenco"),
        ("protag_vs_others_abs", "Distância emocional do protagonista"),
        ("protag_arc", "Arco emocional (final − início)"),
        ("protag_pct_positive", "% falas positivas do protagonista"),
        ("protag_pct_negative", "% falas negativas do protagonista"),
        ("protag_sentiment_range", "Amplitude emocional do protagonista"),
        ("protag_dialogue_coverage", "Cobertura de diálogo do protagonista"),
    ]

    # ── Correlações globais ──
    out.append(f"\n{'='*70}")
    out.append("  CORRELAÇÕES GLOBAIS")
    out.append("=" * 70)

    for tcol, tname in TARGETS:
        sub = df[df[tcol].notna()]
        if len(sub) < 30:
            continue

        out.append(f"\n  ── {tname} (n={len(sub)}) ──")
        corrs = []
        for feat, label in test_feats:
            v = sub[[feat, tcol]].dropna()
            if len(v) < 30:
                continue
            rho, p = stats.spearmanr(v[feat], v[tcol])
            if not np.isnan(rho):
                corrs.append({"feature": label, "feat_id": feat, "rho": rho, "p": p})

        if corrs:
            cdf = pd.DataFrame(corrs).sort_values("p")
            _, p_adj, _, _ = multipletests(cdf["p"], method="fdr_bh")
            cdf["p_adj"] = p_adj

            for _, r in cdf.iterrows():
                sig = "**" if r["p_adj"] < 0.05 else ("*" if r["p"] < 0.05 else "")
                out.append(f"    {r['feature']:<45s}  ρ={r['rho']:+.3f}  p={r['p']:.4f} {sig}")

    # ── Por gênero ──
    out.append(f"\n{'='*70}")
    out.append("  POR GÊNERO (só significativos)")
    out.append("=" * 70)

    for genre in GENRES:
        gsub = df[(df["genre"].str.contains(genre, na=False)) & (df["imdb_rating"].notna())]
        if len(gsub) < 20:
            continue

        hits = []
        for feat, label in test_feats:
            v = gsub[[feat, "imdb_rating"]].dropna()
            if len(v) < 15:
                continue
            rho, p = stats.spearmanr(v[feat], v["imdb_rating"])
            if p < 0.10:
                hits.append({"feature": label, "rho": rho, "p": p})

        if hits:
            out.append(f"\n  ── {genre} (n={len(gsub)}) ──")
            for h in sorted(hits, key=lambda x: x["p"]):
                sig = "**" if h["p"] < 0.01 else ("*" if h["p"] < 0.05 else "~")
                out.append(f"    {h['feature']:<45s}  ρ={h['rho']:+.3f}  p={h['p']:.4f} {sig}")

    # ── Exemplos: protagonistas dos top 10 vs bottom 10 IMDb ──
    out.append(f"\n{'='*70}")
    out.append("  EXEMPLOS: PROTAGONISTAS DE FILMES TOP vs BOTTOM")
    out.append("=" * 70)

    imdb_sorted = df[df["imdb_rating"].notna()].sort_values("imdb_rating", ascending=False)

    out.append(f"\n  Top 10 IMDb:")
    for _, r in imdb_sorted.head(10).iterrows():
        out.append(f"    {r['title']:<30s}  IMDb={r['imdb_rating']:.1f}  protag={r['protagonist']:<15s}  "
                   f"tom={r['protag_sentiment_mean']:+.3f}  arco={r['protag_arc']:+.3f}")

    out.append(f"\n  Bottom 10 IMDb:")
    for _, r in imdb_sorted.tail(10).iterrows():
        out.append(f"    {r['title']:<30s}  IMDb={r['imdb_rating']:.1f}  protag={r['protagonist']:<15s}  "
                   f"tom={r['protag_sentiment_mean']:+.3f}  arco={r['protag_arc']:+.3f}")

    report = "\n".join(out)
    print(report)
    (RESULTS / "character_emotions_report.txt").write_text(report, encoding="utf-8")
    logging.info("Salvo: character_emotion_features.csv + character_emotions_report.txt")


if __name__ == "__main__":
    main()
