#!/usr/bin/env python3
"""Etapa 9: Analise descritiva do dataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from src.utils import FEATURES, RESULTS, ensure_dirs

def main():
    ensure_dirs()
    df = pd.read_csv(FEATURES / "analysis_dataset.csv")
    metric_cols = [c for c in df.columns if c.startswith("gv")]
    out = []

    out.append("=" * 60)
    out.append("  ANALISE DESCRITIVA DO DATASET")
    out.append("=" * 60)
    out.append(f"\nFilmes: {len(df)}")
    out.append(f"Metricas: {len(metric_cols)}")

    # --- Targets ---
    out.append("\n--- TARGETS ---")
    for col in ["imdb_rating", "box_office", "log_box_office"]:
        s = df[col].dropna()
        out.append(f"\n{col} (n={len(s)}):")
        out.append(f"  mean={s.mean():.2f}  std={s.std():.2f}  "
                   f"min={s.min():.2f}  Q1={s.quantile(.25):.2f}  "
                   f"median={s.median():.2f}  Q3={s.quantile(.75):.2f}  max={s.max():.2f}")

    # --- Missing ---
    out.append("\n--- MISSING VALUES ---")
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        out.append(f"  {col}: {missing[col]} missing ({missing[col]/len(df)*100:.1f}%)")

    # --- Generos ---
    out.append("\n--- DISTRIBUICAO POR GENERO ---")
    all_g = []
    for g in df["genre"].dropna():
        for genre in g.split(", "):
            all_g.append(genre.strip())
    counts = Counter(all_g)
    for g, n in counts.most_common(15):
        sub = df[df["genre"].str.contains(g, na=False)]
        med = sub["imdb_rating"].median()
        out.append(f"  {g:<15s}  n={n:>4d}  mediana_imdb={med:.1f}")

    # --- Outliers nas metricas ---
    out.append("\n--- OUTLIERS NAS METRICAS (>3 std) ---")
    for col in metric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        z = (s - s.mean()) / s.std()
        n_out = (z.abs() > 3).sum()
        if n_out > 0:
            out.append(f"  {col}: {n_out} outliers")

    report = "\n".join(out)
    print(report)

    report_path = RESULTS / "descriptive_report.txt"
    report_path.write_text(report, encoding="utf-8")

    # --- Graficos ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#21262d")

    # IMDb
    df["imdb_rating"].dropna().hist(bins=30, ax=axes[0], color="#1f6feb", edgecolor="#0d1117")
    axes[0].set_title("IMDb Rating", color="#c9d1d9")
    axes[0].set_xlabel("Rating", color="#8b949e")

    # Box Office (log)
    df["log_box_office"].dropna().hist(bins=30, ax=axes[1], color="#3fb950", edgecolor="#0d1117")
    axes[1].set_title("log(Box Office)", color="#c9d1d9")
    axes[1].set_xlabel("log($)", color="#8b949e")

    # Year
    df["year"].dropna().hist(bins=30, ax=axes[2], color="#d29922", edgecolor="#0d1117")
    axes[2].set_title("Ano", color="#c9d1d9")
    axes[2].set_xlabel("Ano", color="#8b949e")

    plt.tight_layout()
    plt.savefig(RESULTS / "distributions.png", dpi=150, facecolor="#0d1117")
    plt.close()

    # Heatmap de correlacao entre features
    corr = df[metric_cols].corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, fmt=".1f",
                xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.8})
    ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_title("Correlacao entre metricas dos grafos", color="#c9d1d9", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS / "feature_correlation_heatmap.png", dpi=150, facecolor="#0d1117")
    plt.close()

    print(f"\nGraficos salvos em {RESULTS}/")

if __name__ == "__main__":
    main()
