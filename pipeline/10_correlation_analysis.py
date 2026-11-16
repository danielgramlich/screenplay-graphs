#!/usr/bin/env python3
"""Etapa 10: Correlacoes metricas dos grafos x targets (IMDb, box office)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from src.utils import FEATURES, RESULTS, ensure_dirs

GENRES = ["Drama", "Comedy", "Crime", "Thriller", "Action", "Horror"]
MIN_N = 30

def main():
    ensure_dirs()
    df = pd.read_csv(FEATURES / "analysis_dataset.csv")
    metric_cols = [c for c in df.columns if c.startswith("gv")]
    targets = [("imdb_rating", "IMDb"), ("log_box_office", "log(BoxOffice)")]

    rows = []

    # --- GLOBAL ---
    for target_col, target_name in targets:
        sub = df[df[target_col].notna()]
        for m in metric_cols:
            valid = sub[[m, target_col]].dropna()
            if len(valid) < MIN_N:
                continue
            rho, p = stats.spearmanr(valid[m], valid[target_col])
            rows.append({
                "metric": m, "target": target_name, "slice": "Global",
                "n": len(valid), "rho": round(rho, 4), "p_value": round(p, 6),
            })

    # --- POR GENERO ---
    for genre in GENRES:
        genre_df = df[df["genre"].str.contains(genre, na=False)]
        for target_col, target_name in targets:
            sub = genre_df[genre_df[target_col].notna()]
            if len(sub) < MIN_N:
                continue
            for m in metric_cols:
                valid = sub[[m, target_col]].dropna()
                if len(valid) < MIN_N:
                    continue
                rho, p = stats.spearmanr(valid[m], valid[target_col])
                rows.append({
                    "metric": m, "target": target_name, "slice": genre,
                    "n": len(valid), "rho": round(rho, 4), "p_value": round(p, 6),
                })

    result = pd.DataFrame(rows)

    # Correcao BH por slice+target
    result["p_adjusted"] = np.nan
    for (sl, tgt), group in result.groupby(["slice", "target"]):
        if len(group) > 1:
            _, p_adj, _, _ = multipletests(group["p_value"], method="fdr_bh")
            result.loc[group.index, "p_adjusted"] = p_adj

    result = result.sort_values(["target", "slice", "p_value"])
    result.to_csv(RESULTS / "correlation_table.csv", index=False)

    # --- Resumo ---
    out = []
    out.append("=" * 80)
    out.append("  CORRELACOES: METRICAS DOS GRAFOS x TARGETS")
    out.append("=" * 80)

    for tgt in ["IMDb", "log(BoxOffice)"]:
        out.append(f"\n{'='*40}")
        out.append(f"  TARGET: {tgt}")
        out.append(f"{'='*40}")

        for sl in ["Global"] + GENRES:
            sub = result[(result["target"] == tgt) & (result["slice"] == sl)]
            if sub.empty:
                continue
            sig = sub[sub["p_value"] < 0.05]
            sig_adj = sub[sub["p_adjusted"] < 0.05]

            out.append(f"\n  --- {sl} (n={sub.iloc[0]['n']}) ---")
            out.append(f"  Significativos (p<0.05): {len(sig)}")
            out.append(f"  Significativos (p_adj<0.05): {len(sig_adj)}")

            top = sub.head(5)
            for _, r in top.iterrows():
                flag = "**" if r["p_adjusted"] < 0.05 else ("*" if r["p_value"] < 0.05 else "")
                out.append(f"    {r['metric']:<30s}  rho={r['rho']:+.3f}  "
                           f"p={r['p_value']:.4f}  p_adj={r['p_adjusted']:.4f} {flag}")

    summary = "\n".join(out)
    print(summary)
    (RESULTS / "correlation_summary.txt").write_text(summary, encoding="utf-8")
    print(f"\nSalvo: correlation_table.csv + correlation_summary.txt")

if __name__ == "__main__":
    main()
