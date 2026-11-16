#!/usr/bin/env python3
"""Etapa 11: Modelos baseline (year + genre) para comparacao futura."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from src.utils import FEATURES, RESULTS, ensure_dirs

GENRES_DUMMIES = ["Drama", "Comedy", "Action", "Crime", "Thriller",
                  "Adventure", "Romance", "Horror"]

def build_genre_dummies(df):
    """One-hot encode dos generos principais."""
    for g in GENRES_DUMMIES:
        df[f"genre_{g}"] = df["genre"].str.contains(g, na=False).astype(int)
    return df

def loo_ridge(X, y, alpha=1.0):
    """Ridge com LOO-CV, retorna previsoes."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        model = Ridge(alpha=alpha)
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    return preds

def main():
    ensure_dirs()
    df = pd.read_csv(FEATURES / "analysis_dataset.csv")
    df = build_genre_dummies(df)

    feature_cols = ["year"] + [f"genre_{g}" for g in GENRES_DUMMIES]
    results = []

    for target, target_name in [("imdb_rating", "IMDb"), ("log_box_office", "log(BoxOffice)")]:
        sub = df[df[target].notna()].copy()
        sub = sub.dropna(subset=feature_cols + [target])

        X = sub[feature_cols].values
        y = sub[target].values

        preds = loo_ridge(X, y)

        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        rho, p = stats.spearmanr(y, preds)

        results.append({
            "target": target_name,
            "model": "year + genre (Ridge LOO)",
            "n": len(sub),
            "R2": round(r2, 4),
            "MAE": round(mae, 4),
            "Spearman": round(rho, 4),
            "p_value": round(p, 6),
        })

        print(f"\n--- Baseline: {target_name} ---")
        print(f"  n = {len(sub)}")
        print(f"  R2 = {r2:.4f}")
        print(f"  MAE = {mae:.4f}")
        print(f"  Spearman = {rho:+.4f} (p = {p:.4e})")

        # Salvar previsoes
        sub[f"pred_{target}"] = preds
        sub[["title", target, f"pred_{target}"]].to_csv(
            RESULTS / f"baseline_predictions_{target}.csv", index=False
        )

    pd.DataFrame(results).to_csv(RESULTS / "baseline_results.csv", index=False)
    print(f"\nSalvo: baseline_results.csv")

if __name__ == "__main__":
    main()
