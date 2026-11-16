#!/usr/bin/env python3
"""Etapa 28: Rebuild do dataset analítico com corpus expandido.

Junta:
  - Grafos emocionais (2259 filmes × 44 features)
  - Redes de personagens (2333 filmes × 29 features)
  - Emoção por personagem (do script 24 - originais + MovieSum via 26)
  - Metadata (IMDb, box office, gênero, ano)

Produz:
  - analysis_dataset_expanded.csv — dataset principal
  - Relatório de cobertura
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from collections import Counter
from src.utils import (FEATURES, FEATURES_EMOTIONAL, FEATURES_CHARACTERS,
                        METADATA, RESULTS, ensure_dirs)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    ensure_dirs()

    # ── Carregar fontes ──
    logging.info("Carregando fontes...")

    # Grafos emocionais
    emo = pd.read_csv(FEATURES_EMOTIONAL / "refined_graph_features.csv")
    emo_cols = [c for c in emo.columns if c.startswith("gv")]
    logging.info("  Emocionais: %d filmes, %d features", len(emo), len(emo_cols))

    # Redes de personagens
    char = pd.read_csv(FEATURES_CHARACTERS / "character_network_features.csv")
    char = char[char["parse_ok"] == True].copy()
    char_meta_cols = ["title", "source", "F3_protag_name", "n_canonical_chars"]
    char_feat_cols = [c for c in char.columns if c.startswith("F1_") or c.startswith("F2_") or
                      (c.startswith("F3_") and c != "F3_protag_name")]
    # Adicionar features de emoção por personagem se existirem
    protag_emo_cols = [c for c in char.columns if c.startswith("protag_") or c == "others_sentiment_mean"]
    char_keep = ["title"] + char_feat_cols + protag_emo_cols
    char_keep = [c for c in char_keep if c in char.columns]
    logging.info("  Personagens: %d filmes, %d features", len(char), len(char_feat_cols) + len(protag_emo_cols))

    # Metadata
    meta = pd.read_csv(METADATA / "film_list.csv")
    meta = meta[meta["matched"] == True].copy()
    logging.info("  Metadata: %d filmes matched", len(meta))

    # ── Merge ──
    logging.info("Fazendo merge...")

    # Emoção + Personagens
    df = emo[["title"] + emo_cols].merge(
        char[char_keep], on="title", how="outer", suffixes=("", "_char")
    )

    # + Metadata
    df = df.merge(
        meta[["source_file", "imdb_rating", "box_office", "year", "genre"]],
        left_on="title", right_on="source_file", how="inner"
    ).drop(columns=["source_file"])

    # Limpar box office
    df["box_office_clean"] = (
        df["box_office"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df["log_box_office"] = np.log1p(df["box_office_clean"])

    logging.info("Dataset merged: %d filmes", len(df))

    # ── Flags de cobertura ──
    df["has_emotion"] = df[emo_cols[0]].notna()
    df["has_characters"] = df[char_feat_cols[0]].notna() if char_feat_cols[0] in df.columns else False
    df["has_imdb"] = df["imdb_rating"].notna()
    df["has_box_office"] = df["box_office_clean"].notna()
    df["has_all"] = df["has_emotion"] & df["has_characters"] & df["has_imdb"]

    # ── Salvar ──
    out_path = FEATURES / "analysis_dataset_expanded.csv"
    df.to_csv(out_path, index=False)

    # ── Relatório ──
    out = []
    out.append("=" * 70)
    out.append("  DATASET ANALÍTICO EXPANDIDO")
    out.append("=" * 70)

    out.append(f"\n  Total de filmes: {len(df)}")
    out.append(f"  Com grafos emocionais: {df['has_emotion'].sum()}")
    out.append(f"  Com rede de personagens: {df['has_characters'].sum()}")
    out.append(f"  Com IMDb rating: {df['has_imdb'].sum()}")
    out.append(f"  Com box office: {df['has_box_office'].sum()}")
    out.append(f"  Com TUDO (emoção + personagens + IMDb): {df['has_all'].sum()}")

    out.append(f"\n  Features emocionais: {len(emo_cols)}")
    out.append(f"  Features de personagens: {len(char_feat_cols)}")
    out.append(f"  Features de emoção do protagonista: {len(protag_emo_cols)}")
    out.append(f"  Total de colunas: {len(df.columns)}")

    # IMDb stats
    has_imdb = df[df["has_imdb"]]
    out.append(f"\n  IMDb Rating (n={len(has_imdb)}):")
    out.append(f"    mean={has_imdb['imdb_rating'].mean():.2f}  "
               f"median={has_imdb['imdb_rating'].median():.1f}  "
               f"std={has_imdb['imdb_rating'].std():.2f}  "
               f"range=[{has_imdb['imdb_rating'].min():.1f}, {has_imdb['imdb_rating'].max():.1f}]")

    # Gêneros
    all_g = []
    for g in df["genre"].dropna():
        for genre in g.split(", "):
            all_g.append(genre.strip())
    counts = Counter(all_g)

    out.append(f"\n  Gêneros (top 12):")
    for g, n in counts.most_common(12):
        sub = df[df["genre"].str.contains(g, na=False)]
        n_all = sub["has_all"].sum()
        med = sub.loc[sub["has_imdb"], "imdb_rating"].median()
        out.append(f"    {g:<15s}  total={n:>5d}  com_tudo={n_all:>5d}  mediana_imdb={med:.1f}")

    # Comparação antes vs depois
    out.append(f"\n  Comparação com dataset anterior:")
    old_path = FEATURES / "analysis_dataset_with_roi.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        out.append(f"    Antes: {len(old)} filmes, {old['imdb_rating'].notna().sum()} com IMDb")
        out.append(f"    Agora: {len(df)} filmes, {df['has_imdb'].sum()} com IMDb")
        out.append(f"    Crescimento: {len(df)/len(old):.1f}×")

    report = "\n".join(out)
    print(report)

    (RESULTS / "expanded_dataset_report.txt").write_text(report, encoding="utf-8")
    logging.info("Salvo: %s", out_path)
    logging.info("Relatório: %s", RESULTS / "expanded_dataset_report.txt")


if __name__ == "__main__":
    main()
