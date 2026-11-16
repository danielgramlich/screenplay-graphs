#!/usr/bin/env python3
"""Etapa 35: Análise controlada por orçamento (budget-controlled).

Responde à crítica da orientadora sobre baseline não validado e ao confundidor
de marketing. Implementa duas estratégias complementares:

(1) Baseline forte literatura-validada: Ridge(ano + gênero + log_budget) seguindo
    Sharda & Delen (2006) e Hunter et al. (2016). O orçamento de produção é proxy
    consolidado de gastos de marketing (Eliashberg et al., 2007).

(2) Análise de resíduos: regride alvo em ano+gênero (+log_budget), extrai resíduos,
    e correlaciona resíduos com features narrativas. Isola o componente que pode
    ser atribuído ao roteiro após remover tendências não-narrativas.

Reporta:
 - M0: ano + gênero (baseline atual)
 - M1: ano + gênero + log_budget (baseline forte, literatura)
 - M2: M1 + features narrativas (modelo completo)
 - ΔR² de M2 sobre M1 (contribuição incremental do roteiro isolada de marketing)
 - Correlação de features com resíduos
"""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from src.utils import FEATURES, RESULTS, ensure_dirs

TARGETS = [("imdb_rating", "IMDb"), ("log_box_office", "log(BoxOffice)")]
GENRE_DUMMIES = ["Drama", "Comedy", "Action", "Crime", "Thriller",
                 "Adventure", "Romance", "Horror"]
SEED = 42

ROB_GRAPH = ["rob_gv5_autoloop", "rob_gv3_pr_neg", "rob_gv5_pr_v5_pos",
             "rob_gv3_entropy"]
ROB_PROTAG = ["rob_protag_mean", "rob_protag_pct_positive",
              "rob_protag_pct_negative", "rob_protag_volatility",
              "rob_protag_arc", "rob_protag_coverage"]
CHAR_NET = ["F1_density", "F1_strength_centralization", "F1_max_betweenness",
            "F1_avg_clustering", "F1_modularity", "F2_W_entropy",
            "F3_protag_strength_frac", "F3_protag_presence"]


def kfold_ridge(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    preds = np.zeros(len(y))
    for tr, te in kf.split(X):
        m = Ridge(alpha=1.0, random_state=SEED)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


def fit_ridge_oos(df, features, target):
    """Ajusta Ridge 5-fold CV e retorna (r2, mae, rho, preds, n)."""
    sub = df.dropna(subset=features + [target]).copy()
    if len(sub) < 50:
        return None
    X = StandardScaler().fit_transform(sub[features].values)
    y = sub[target].values
    preds = kfold_ridge(X, y)
    return {
        "r2": r2_score(y, preds),
        "mae": mean_absolute_error(y, preds),
        "rho": stats.spearmanr(y, preds).statistic,
        "n": len(sub),
        "sub": sub,
        "preds": preds,
    }


def residualize(df, target, controls):
    """Regride target em controls e retorna df com coluna 'resid'."""
    sub = df.dropna(subset=controls + [target]).copy()
    X = StandardScaler().fit_transform(sub[controls].values)
    y = sub[target].values
    m = Ridge(alpha=1.0, random_state=SEED).fit(X, y)
    sub["resid"] = y - m.predict(X)
    return sub


def main():
    ensure_dirs()
    out = []

    # ── Carregar final_combined_dataset (1655 filmes com todas features) ──
    final = pd.read_csv(FEATURES / "final_combined_dataset.csv")
    roi = pd.read_csv(FEATURES / "analysis_dataset_with_roi.csv")

    # Mergear budget por title
    df = final.merge(roi[["title", "budget", "log_budget"]],
                     on="title", how="left")

    # Garantir dummies de gênero
    genre_cols = [f"g_{g}" for g in GENRE_DUMMIES if f"g_{g}" in df.columns]
    if len(genre_cols) < len(GENRE_DUMMIES):
        for g in GENRE_DUMMIES:
            if f"g_{g}" not in df.columns:
                df[f"g_{g}"] = df["genre"].str.contains(g, na=False).astype(int)
        genre_cols = [f"g_{g}" for g in GENRE_DUMMIES]

    rg_ok = [f for f in ROB_GRAPH if f in df.columns]
    rp_ok = [f for f in ROB_PROTAG if f in df.columns]
    cn_ok = [f for f in CHAR_NET if f in df.columns]
    narr_all = rg_ok + rp_ok + cn_ok

    n_total = len(df)
    n_budget = df["log_budget"].notna().sum()

    out.append("=" * 72)
    out.append("  ANÁLISE CONTROLADA POR ORÇAMENTO (pipeline/35)")
    out.append("  Baseline literatura-validado + isolamento do efeito roteiro")
    out.append("=" * 72)
    out.append(f"\n  Corpus total:             {n_total} filmes")
    out.append(f"  Com orçamento (TMDb):     {n_budget} ({n_budget/n_total*100:.1f}%)")
    out.append(f"  Features narrativas:      {len(rg_ok)} grafos + "
               f"{len(rp_ok)} protag + {len(cn_ok)} rede  (Σ={len(narr_all)})")
    out.append("\n  Referências do baseline forte:")
    out.append("    Sharda & Delen (2006) — Expert Syst. Appl. 30(2)")
    out.append("    Hunter, Smith & Singh (2016) — J. Screenwriting 7(2)")
    out.append("    Eliashberg et al. (2007) — Marketing Science")

    # ═══════════════════════════════════════════════════════════════════
    # PARTE 1 — COMPARAÇÃO DE MODELOS: M0 vs M1 vs M2
    # ═══════════════════════════════════════════════════════════════════
    out.append(f"\n{'='*72}")
    out.append("  PARTE 1 — MODELOS ANINHADOS (Ridge, 5-fold CV)")
    out.append("=" * 72)
    out.append("\n  M0: ano + gênero            (baseline atual)")
    out.append("  M1: ano + gênero + log_budget  (baseline literatura)")
    out.append("  M2: M1 + features narrativas   (modelo completo)")
    out.append("  ΔR² = M2 − M1  →  contribuição do roteiro isolada de marketing")

    results_rows = []

    for tcol, tname in TARGETS:
        out.append(f"\n  ── {tname} ──")

        # Sub-amostra: apenas filmes com budget E todas as features narrativas
        required = narr_all + genre_cols + ["year", "log_budget", tcol]
        sub = df.dropna(subset=required).copy()
        n = len(sub)

        if n < 100:
            out.append(f"    Amostra insuficiente (n={n}), pulando.")
            continue

        base_ctrl = ["year"] + genre_cols
        budget_ctrl = base_ctrl + ["log_budget"]

        configs = [
            ("M0_year+genre",      base_ctrl),
            ("M1_+log_budget",     budget_ctrl),
            ("M2_+narrativa",      budget_ctrl + narr_all),
            ("M2a_+grafo",         budget_ctrl + rg_ok),
            ("M2b_+protag",        budget_ctrl + rp_ok),
            ("M2c_+rede",          budget_ctrl + cn_ok),
        ]

        out.append(f"    {'Modelo':<20s} {'R²':>8s} {'ρ':>8s} "
                   f"{'MAE':>7s} {'n':>5s} {'feat':>5s}")
        out.append(f"    {'-'*60}")

        r2_by_model = {}
        for name, feats in configs:
            X = StandardScaler().fit_transform(sub[feats].values)
            y = sub[tcol].values
            preds = kfold_ridge(X, y)
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            rho = stats.spearmanr(y, preds).statistic
            r2_by_model[name] = r2
            out.append(f"    {name:<20s} {r2:>8.4f} {rho:>+8.3f} "
                       f"{mae:>7.3f} {n:>5d} {len(feats):>5d}")
            results_rows.append({"target": tname, "model": name, "n": n,
                                 "R2": round(r2,4), "rho": round(rho,4),
                                 "MAE": round(mae,4), "n_features": len(feats)})

        # Deltas
        d01 = r2_by_model["M1_+log_budget"] - r2_by_model["M0_year+genre"]
        d12 = r2_by_model["M2_+narrativa"] - r2_by_model["M1_+log_budget"]
        out.append(f"\n    ΔR² M1−M0 (budget):        {d01:+.4f}   "
                   "(ganho ao controlar por marketing)")
        out.append(f"    ΔR² M2−M1 (narrativa):     {d12:+.4f}   "
                   "(contribuição do roteiro isolada)")
        for block, feats in [("grafo", rg_ok), ("protag", rp_ok), ("rede", cn_ok)]:
            key = {"grafo":"M2a_+grafo","protag":"M2b_+protag","rede":"M2c_+rede"}[block]
            d = r2_by_model[key] - r2_by_model["M1_+log_budget"]
            out.append(f"      └─ ΔR² {block:<7s}: {d:+.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # PARTE 2 — ANÁLISE DE RESÍDUOS (isolamento do efeito roteiro)
    # ═══════════════════════════════════════════════════════════════════
    out.append(f"\n{'='*72}")
    out.append("  PARTE 2 — RESÍDUOS (após controlar ano + gênero + log_budget)")
    out.append("=" * 72)
    out.append("\n  Método: regride target em (ano + gênero + log_budget) → resíduo ε.")
    out.append("  Correlaciona ε com cada feature narrativa (Spearman).")
    out.append("  Interpretação: correlação ε × feature = efeito do roteiro após")
    out.append("  remover tendências temporais, de gênero e de gastos de produção.")

    resid_rows = []

    for tcol, tname in TARGETS:
        controls = ["year"] + genre_cols + ["log_budget"]
        required = controls + [tcol]
        sub = df.dropna(subset=required).copy()
        if len(sub) < 100:
            continue

        res = residualize(sub, tcol, controls)

        out.append(f"\n  ── {tname} (n={len(res)}) ──")
        out.append(f"    {'Feature':<35s} {'ρ(ε,feat)':>10s} {'p':>8s} {'p_adj':>8s}")
        out.append(f"    {'-'*63}")

        rows = []
        for f in narr_all:
            v = res[[f, "resid"]].dropna()
            if len(v) < 50:
                continue
            r = stats.spearmanr(v[f], v["resid"])
            rows.append({"target": tname, "feature": f,
                         "rho": r.statistic, "p": r.pvalue, "n": len(v)})

        if rows:
            rdf = pd.DataFrame(rows).sort_values("p")
            _, p_adj, _, _ = multipletests(rdf["p"], method="fdr_bh")
            rdf["p_adj"] = p_adj

            for _, r in rdf.iterrows():
                sig = "**" if r["p_adj"] < 0.05 else ("*" if r["p"] < 0.05 else "")
                out.append(f"    {r['feature']:<35s} {r['rho']:>+10.3f} "
                           f"{r['p']:>8.4f} {r['p_adj']:>8.4f} {sig}")

            n_sig = (rdf["p_adj"] < 0.05).sum()
            out.append(f"\n    Significativos (BH q<0.05): {n_sig}/{len(rdf)}")
            resid_rows.extend(rdf.to_dict("records"))

    # ═══════════════════════════════════════════════════════════════════
    # PARTE 3 — SANITY CHECK: budget sozinho vs narrativa sozinha
    # ═══════════════════════════════════════════════════════════════════
    out.append(f"\n{'='*72}")
    out.append("  PARTE 3 — CORRELAÇÃO BRUTA (contextualização)")
    out.append("=" * 72)

    for tcol, tname in TARGETS:
        sub = df.dropna(subset=["log_budget", tcol])
        if len(sub) < 50:
            continue
        r = stats.spearmanr(sub["log_budget"], sub[tcol])
        out.append(f"\n  {tname} vs log_budget (n={len(sub)}):  "
                   f"ρ={r.statistic:+.3f}  p={r.pvalue:.4f}")

    # ── Salvar ──
    txt = "\n".join(out)
    print(txt)
    (RESULTS / "budget_controlled_analysis.txt").write_text(txt, encoding="utf-8")
    pd.DataFrame(results_rows).to_csv(
        RESULTS / "budget_controlled_model_comparison.csv", index=False)
    if resid_rows:
        pd.DataFrame(resid_rows).to_csv(
            RESULTS / "budget_controlled_residual_correlations.csv", index=False)

    print(f"\n✓ Relatório: {RESULTS / 'budget_controlled_analysis.txt'}")
    print(f"✓ Tabela modelos: {RESULTS / 'budget_controlled_model_comparison.csv'}")
    print(f"✓ Tabela resíduos: {RESULTS / 'budget_controlled_residual_correlations.csv'}")


if __name__ == "__main__":
    main()
