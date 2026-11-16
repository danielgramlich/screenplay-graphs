#!/usr/bin/env python3
"""Etapa 13: Análise completa — exploratória, modelagem e interpretação."""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from src.utils import FEATURES, RESULTS, ensure_dirs

SEED = 42
GENRES_DUMMIES = ["Drama", "Comedy", "Action", "Crime", "Thriller",
                  "Adventure", "Romance", "Horror"]
TARGETS = [
    ("imdb_rating", "IMDb Rating"),
    ("log_box_office", "log(Box Office)"),
    ("log_revenue_multiple", "log(Revenue Multiple)"),
]


def load_data():
    df = pd.read_csv(FEATURES / "analysis_dataset_with_roi.csv")
    metric_cols = [c for c in df.columns if c.startswith("gv")]

    # Genre dummies
    for g in GENRES_DUMMIES:
        df[f"g_{g}"] = df["genre"].str.contains(g, na=False).astype(int)
    genre_cols = [f"g_{g}" for g in GENRES_DUMMIES]

    return df, metric_cols, genre_cols


def loo_ridge(X, y, alpha=1.0):
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = Ridge(alpha=alpha)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


def run_correlations(df, metric_cols, target_col, label, out):
    sub = df[df[target_col].notna()].copy()
    n = len(sub)
    out.append(f"\n{'='*70}")
    out.append(f"  CORRELAÇÕES: {label} (n={n})")
    out.append(f"{'='*70}")

    rows = []
    for m in metric_cols:
        valid = sub[[m, target_col]].dropna()
        if len(valid) < 30:
            continue
        rho, p = stats.spearmanr(valid[m], valid[target_col])
        if np.isnan(rho):
            continue
        rows.append({"metric": m, "rho": rho, "p": p, "n": len(valid)})

    if not rows:
        out.append("  Sem correlações válidas.")
        return pd.DataFrame()

    res = pd.DataFrame(rows)
    _, p_adj, _, _ = multipletests(res["p"], method="fdr_bh")
    res["p_adj"] = p_adj
    res = res.sort_values("p")

    # Top positivas
    pos = res[res["rho"] > 0].head(5)
    neg = res[res["rho"] < 0].head(5)

    out.append(f"\n  Top associações POSITIVAS:")
    for _, r in pos.iterrows():
        flag = "**" if r["p_adj"] < 0.05 else ("*" if r["p"] < 0.05 else "")
        out.append(f"    {r['metric']:<30s}  ρ={r['rho']:+.3f}  p={r['p']:.4f}  p_adj={r['p_adj']:.4f} {flag}")

    out.append(f"\n  Top associações NEGATIVAS:")
    for _, r in neg.iterrows():
        flag = "**" if r["p_adj"] < 0.05 else ("*" if r["p"] < 0.05 else "")
        out.append(f"    {r['metric']:<30s}  ρ={r['rho']:+.3f}  p={r['p']:.4f}  p_adj={r['p_adj']:.4f} {flag}")

    sig = res[res["p"] < 0.05]
    sig_adj = res[res["p_adj"] < 0.05]
    out.append(f"\n  Significativos: {len(sig)} (p<0.05), {len(sig_adj)} (p_adj<0.05)")
    out.append(f"  Max |ρ|: {res['rho'].abs().max():.3f}")

    return res


def run_models(df, metric_cols, genre_cols, target_col, label, out):
    sub = df[df[target_col].notna()].copy()

    # Dropar linhas com NaN em features
    all_features = metric_cols + genre_cols + ["year", "log_budget"]
    available = [c for c in all_features if c in sub.columns]

    # Dataset para modelo simples (só grafos)
    simple_sub = sub.dropna(subset=metric_cols + [target_col])
    # Dataset para modelo controlado (grafos + controles)
    control_features = [c for c in ["year", "log_budget"] + genre_cols if c in sub.columns]
    full_features = metric_cols + control_features
    controlled_sub = sub.dropna(subset=full_features + [target_col])

    out.append(f"\n{'='*70}")
    out.append(f"  MODELOS: {label}")
    out.append(f"{'='*70}")

    results = []

    # --- Modelo 0: Baseline (só controles) ---
    if len(controlled_sub) >= 30:
        X_base = StandardScaler().fit_transform(controlled_sub[control_features].values)
        y_base = controlled_sub[target_col].values
        preds_base = loo_ridge(X_base, y_base)
        r2_base = r2_score(y_base, preds_base)
        mae_base = mean_absolute_error(y_base, preds_base)
        sp_base, _ = stats.spearmanr(y_base, preds_base)

        out.append(f"\n  Modelo 0 — Baseline (year + genre + log_budget)")
        out.append(f"    n={len(controlled_sub)}, R²={r2_base:.4f}, MAE={mae_base:.4f}, Spearman={sp_base:+.4f}")
        results.append({"model": "Baseline", "target": label,
                        "n": len(controlled_sub), "R2": r2_base, "MAE": mae_base, "Spearman": sp_base})

    # --- Modelo 1: Só grafos ---
    if len(simple_sub) >= 30:
        X1 = StandardScaler().fit_transform(simple_sub[metric_cols].values)
        y1 = simple_sub[target_col].values
        preds1 = loo_ridge(X1, y1)
        r2_1 = r2_score(y1, preds1)
        mae_1 = mean_absolute_error(y1, preds1)
        sp_1, _ = stats.spearmanr(y1, preds1)

        out.append(f"\n  Modelo 1 — Só features de grafo")
        out.append(f"    n={len(simple_sub)}, R²={r2_1:.4f}, MAE={mae_1:.4f}, Spearman={sp_1:+.4f}")
        results.append({"model": "Grafos", "target": label,
                        "n": len(simple_sub), "R2": r2_1, "MAE": mae_1, "Spearman": sp_1})

        # Coeficientes do modelo full fit
        model = Ridge(alpha=1.0)
        model.fit(X1, y1)
        coefs = sorted(zip(metric_cols, model.coef_), key=lambda x: abs(x[1]), reverse=True)
        out.append(f"    Top coeficientes:")
        for name, coef in coefs[:8]:
            out.append(f"      {name:<30s}  β={coef:+.4f}")

    # --- Modelo 2: Grafos + controles ---
    if len(controlled_sub) >= 30:
        X2 = StandardScaler().fit_transform(controlled_sub[full_features].values)
        y2 = controlled_sub[target_col].values
        preds2 = loo_ridge(X2, y2)
        r2_2 = r2_score(y2, preds2)
        mae_2 = mean_absolute_error(y2, preds2)
        sp_2, _ = stats.spearmanr(y2, preds2)

        out.append(f"\n  Modelo 2 — Grafos + controles (year + genre + log_budget)")
        out.append(f"    n={len(controlled_sub)}, R²={r2_2:.4f}, MAE={mae_2:.4f}, Spearman={sp_2:+.4f}")
        results.append({"model": "Grafos+Ctrl", "target": label,
                        "n": len(controlled_sub), "R2": r2_2, "MAE": mae_2, "Spearman": sp_2})

        # Delta vs baseline
        if results[0]["model"] == "Baseline":
            delta_r2 = r2_2 - r2_base
            delta_sp = sp_2 - sp_base
            out.append(f"\n  DELTA vs Baseline:")
            out.append(f"    ΔR² = {delta_r2:+.4f}")
            out.append(f"    ΔSpearman = {delta_sp:+.4f}")
            if delta_r2 > 0.01:
                out.append(f"    → Grafos adicionam valor incremental")
            elif delta_r2 > -0.01:
                out.append(f"    → Grafos não adicionam nem prejudicam")
            else:
                out.append(f"    → Grafos prejudicam (overfitting ou ruído)")

    return results


def main():
    ensure_dirs()
    df_full, metric_cols, genre_cols = load_data()

    # Duas versões
    df_filtered = df_full[
        (df_full["budget"].isna()) | (df_full["budget"] > 1_000_000)
    ].copy()

    out = []
    all_results = []

    out.append("=" * 70)
    out.append("  ANÁLISE COMPLETA — GRAFOS EMOCIONAIS vs SUCESSO")
    out.append("=" * 70)

    for version_name, df in [("DATASET COMPLETO", df_full),
                              ("FILTRADO (budget > $1M)", df_filtered)]:
        out.append(f"\n\n{'#'*70}")
        out.append(f"  {version_name}")
        out.append(f"  n={len(df)}")
        out.append(f"{'#'*70}")

        # Distribuição dos targets
        out.append(f"\n--- Distribuição dos targets ---")
        for col, label in TARGETS:
            s = df[col].dropna()
            if len(s) > 0:
                out.append(f"  {label}: n={len(s)}, mean={s.mean():.2f}, "
                           f"std={s.std():.2f}, median={s.median():.2f}, "
                           f"skew={s.skew():.2f}")

        # Correlações
        for col, label in TARGETS:
            run_correlations(df, metric_cols, col, f"{label} [{version_name}]", out)

        # Modelos
        for col, label in TARGETS:
            results = run_models(df, metric_cols, genre_cols, col,
                                 f"{label} [{version_name}]", out)
            for r in results:
                r["version"] = version_name
            all_results.extend(results)

    # --- COMPARAÇÃO ENTRE TARGETS ---
    out.append(f"\n\n{'#'*70}")
    out.append(f"  COMPARAÇÃO ENTRE TARGETS")
    out.append(f"{'#'*70}")

    res_df = pd.DataFrame(all_results)
    for version in ["DATASET COMPLETO", "FILTRADO (budget > $1M)"]:
        out.append(f"\n  --- {version} ---")
        v = res_df[res_df["version"] == version]
        out.append(f"  {'Target':<25s} {'Modelo':<15s} {'R²':>8s} {'Spearman':>10s} {'MAE':>8s} {'n':>5s}")
        out.append(f"  {'-'*70}")
        for _, r in v.iterrows():
            out.append(f"  {r['target']:<25s} {r['model']:<15s} "
                       f"{r['R2']:>8.4f} {r['Spearman']:>+10.4f} {r['MAE']:>8.4f} {r['n']:>5.0f}")

    # --- INTERPRETAÇÃO ---
    out.append(f"\n\n{'#'*70}")
    out.append(f"  INTERPRETAÇÃO")
    out.append(f"{'#'*70}")

    # Análise automática dos resultados
    for version in ["DATASET COMPLETO", "FILTRADO (budget > $1M)"]:
        out.append(f"\n  --- {version} ---")
        v = res_df[res_df["version"] == version]

        for target_label in ["IMDb Rating", "log(Box Office)", "log(Revenue Multiple)"]:
            t = v[v["target"] == target_label]
            if t.empty:
                continue

            baseline = t[t["model"] == "Baseline"]
            grafos = t[t["model"] == "Grafos"]
            combined = t[t["model"] == "Grafos+Ctrl"]

            out.append(f"\n  {target_label}:")

            if not grafos.empty:
                r2_g = grafos.iloc[0]["R2"]
                sp_g = grafos.iloc[0]["Spearman"]
                if r2_g > 0.05:
                    strength = "moderado"
                elif r2_g > 0.01:
                    strength = "fraco"
                elif r2_g > -0.01:
                    strength = "nulo"
                else:
                    strength = "negativo (overfitting)"
                out.append(f"    Sinal dos grafos sozinhos: {strength} (R²={r2_g:.4f}, ρ={sp_g:+.3f})")

            if not baseline.empty and not combined.empty:
                delta = combined.iloc[0]["R2"] - baseline.iloc[0]["R2"]
                if delta > 0.02:
                    out.append(f"    Valor incremental dos grafos: SIM (ΔR²={delta:+.4f})")
                elif delta > 0:
                    out.append(f"    Valor incremental dos grafos: marginal (ΔR²={delta:+.4f})")
                else:
                    out.append(f"    Valor incremental dos grafos: NÃO (ΔR²={delta:+.4f})")

    report = "\n".join(out)
    print(report)

    (RESULTS / "full_analysis_report.txt").write_text(report, encoding="utf-8")
    res_df.to_csv(RESULTS / "model_comparison.csv", index=False)

    print(f"\nSalvo: full_analysis_report.txt + model_comparison.csv")


if __name__ == "__main__":
    main()
