#!/usr/bin/env python3
"""Etapa 8: Montar dataset analitico final (1 linha por filme)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.utils import FEATURES, METADATA

def main():
    feats = pd.read_csv(FEATURES / "emotional_graphs" / "refined_graph_features.csv")
    meta = pd.read_csv(METADATA / "film_list.csv")
    meta = meta[meta["matched"] == True]

    # Limpar box_office
    meta = meta.copy()
    meta["box_office_raw"] = meta["box_office"]
    meta["box_office"] = (
        meta["box_office"]
        .str.replace(r"[\$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )
    meta["log_box_office"] = np.log1p(meta["box_office"])

    # Merge
    df = feats.merge(
        meta[["source_file", "imdb_rating", "box_office", "log_box_office", "year", "genre"]],
        left_on="title", right_on="source_file", how="inner",
    ).drop(columns=["source_file"])

    # Reportar
    n = len(df)
    n_imdb = df["imdb_rating"].notna().sum()
    n_bo = df["box_office"].notna().sum()
    n_both = df[["imdb_rating", "box_office"]].dropna().shape[0]
    metric_cols = [c for c in df.columns if c.startswith("gv")]

    print(f"Dataset final: {n} filmes")
    print(f"  Com IMDb rating: {n_imdb}")
    print(f"  Com box office:  {n_bo}")
    print(f"  Com ambos:       {n_both}")
    print(f"  Metricas:        {len(metric_cols)}")
    print(f"  Colunas total:   {len(df.columns)}")
    print(f"\n  IMDb: {df['imdb_rating'].describe().to_string()}")
    print(f"\n  Box Office: {df['box_office'].describe().to_string()}")

    # Salvar
    out = FEATURES / "analysis_dataset.csv"
    df.to_csv(out, index=False)
    print(f"\nSalvo: {out}")

if __name__ == "__main__":
    main()
