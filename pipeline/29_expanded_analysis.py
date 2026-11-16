#!/usr/bin/env python3
"""Etapa 29: Análise completa com corpus expandido (1537 filmes).

Roda as mesmas análises dos scripts 10, 14, 23 mas no corpus 3× maior.
Compara resultados antes vs depois para ver se os padrões se confirmam.
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
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from src.utils import FEATURES, RESULTS, ensure_dirs

TARGETS = [("imdb_rating", "IMDb"), ("log_box_office", "log(BoxOffice)")]
GENRES = ["Drama", "Comedy", "Action", "Crime", "Thriller", "Horror", "Romance"]
GENRE_DUMMIES = ["Drama", "Comedy", "Action", "Crime", "Thriller",
                  "Adventure", "Romance", "Horror"]

# Features selecionadas
EMO_FEATS = ["gv3_pr_neg", "gv5_pr_v5_pos", "gv5_autoloop", "gv3_entropy", "gv3_neg_to_pos"]
CHAR_FEATS = ["F1_density", "F1_strength_centralization", "F1_max_betweenness",
              "F1_avg_clustering", "F1_modularity", "F2_W_entropy",
              "F2_D_mean", "F3_protag_strength_frac", "F3_protag_presence"]
PROTAG_FEATS = ["protag_sentiment_mean", "protag_volatility", "protag_pct_positive",
                "protag_pct_negative", "protag_arc", "protag_vs_others"]
SEED = 42


def loo_ridge(X, y):
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = Ridge(alpha=1.0, random_state=SEED)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


def main():
    ensure_dirs()
    df = pd.read_csv(FEATURES / "analysis_dataset_expanded.csv")

    # Filtrar: só filmes com tudo
    df = df[df["has_all"] == True].copy()

    for g in GENRE_DUMMIES:
        df[f"g_{g}"] = df["genre"].str.contains(g, na=False).astype(int)
    genre_cols = [f"g_{g}" for g in GENRE_DUMMIES]

    # Verificar features disponíveis
    emo_ok = [f for f in EMO_FEATS if f in df.columns]
    char_ok = [f for f in CHAR_FEATS if f in df.columns]
    protag_ok = [f for f in PROTAG_FEATS if f in df.columns]

    out = []
    out.append("=" * 70)
    out.append("  ANÁLISE EXPANDIDA — 3× MAIS FILMES")
    out.append("=" * 70)
    out.append(f"\n  Dataset: {len(df)} filmes com emoção + personagens + IMDb")
    out.append(f"  Features: {len(emo_ok)} emocionais, {len(char_ok)} personagens, {len(protag_ok)} protagonista")

    # ═══════════════════════════════════════════════════════
    # PARTE 1 — CORRELAÇÕES GLOBAIS
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  PARTE 1 — CORRELAÇÕES GLOBAIS")
    out.append("=" * 70)

    all_feats = [("EMO", emo_ok), ("CHAR", char_ok), ("PROTAG", protag_ok)]

    for tcol, tname in TARGETS:
        sub = df[df[tcol].notna()]
        out.append(f"\n  ── {tname} (n={len(sub)}) ──")

        all_corrs = []
        for group_name, feats in all_feats:
            for feat in feats:
                v = sub[[feat, tcol]].dropna()
                if len(v) < 30:
                    continue
                rho, p = stats.spearmanr(v[feat], v[tcol])
                if not np.isnan(rho):
                    all_corrs.append({"group": group_name, "feature": feat,
                                      "rho": rho, "p": p})

        if all_corrs:
            cdf = pd.DataFrame(all_corrs).sort_values("p")
            _, p_adj, _, _ = multipletests(cdf["p"], method="fdr_bh")
            cdf["p_adj"] = p_adj

            sig_adj = (cdf["p_adj"] < 0.05).sum()
            out.append(f"    Significativos (p_adj<0.05): {sig_adj}/{len(cdf)}")
            out.append(f"    {'Grupo':<8s} {'Feature':<35s} {'ρ':>7s} {'p':>8s} {'p_adj':>8s}")
            out.append(f"    {'-'*70}")

            for _, r in cdf.head(15).iterrows():
                sig = "**" if r["p_adj"] < 0.05 else ("*" if r["p"] < 0.05 else "")
                out.append(f"    {r['group']:<8s} {r['feature']:<35s} {r['rho']:>+7.3f} "
                           f"{r['p']:>8.4f} {r['p_adj']:>8.4f} {sig}")

    # ═══════════════════════════════════════════════════════
    # PARTE 2 — MODELOS COMPARATIVOS (LOO-CV Ridge)
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  PARTE 2 — MODELOS COMPARATIVOS (Ridge LOO-CV)")
    out.append("=" * 70)

    model_results = []

    for tcol, tname in TARGETS:
        ctrl = ["year"] + genre_cols
        sub_full = df.dropna(subset=[tcol])

        configs = [
            ("A_baseline", ctrl),
            ("B_emocao", emo_ok + ctrl),
            ("C_personagens", char_ok + ctrl),
            ("D_protagonista", protag_ok + ctrl),
            ("E_emo+char", emo_ok + char_ok + ctrl),
            ("F_tudo", emo_ok + char_ok + protag_ok + ctrl),
        ]

        out.append(f"\n  ── {tname} ──")

        for model_name, feats in configs:
            available = [f for f in feats if f in sub_full.columns]
            sub = sub_full.dropna(subset=available)
            if len(sub) < 50:
                continue

            X = StandardScaler().fit_transform(sub[available].values)
            y = sub[tcol].values

            # LOO-CV é lento com 1500+ filmes — usar apenas para modelos pequenos
            # Para modelos com muitas features, usar 5-fold
            if len(available) <= 15 and len(sub) <= 600:
                preds = loo_ridge(X, y)
                cv_type = "LOO"
            else:
                from sklearn.model_selection import cross_val_predict, KFold
                kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
                m = Ridge(alpha=1.0, random_state=SEED)
                preds = cross_val_predict(m, X, y, cv=kf)
                cv_type = "5-fold"

            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            rho, _ = stats.spearmanr(y, preds)

            model_results.append({
                "target": tname, "model": model_name, "n": len(sub),
                "R2": round(r2, 4), "MAE": round(mae, 4),
                "Spearman": round(rho, 4), "n_features": len(available), "cv": cv_type,
            })

            out.append(f"    [{model_name:<15s}]  n={len(sub):>5d}  R²={r2:>7.4f}  "
                       f"MAE={mae:>6.3f}  ρ={rho:>+7.3f}  ({len(available)} feats, {cv_type})")

        # Deltas
        models = {r["model"]: r for r in model_results if r["target"] == tname}
        if "A_baseline" in models:
            baseline_r2 = models["A_baseline"]["R2"]
            for m in ["B_emocao", "C_personagens", "D_protagonista", "E_emo+char", "F_tudo"]:
                if m in models:
                    delta = models[m]["R2"] - baseline_r2
                    tag = "MELHORA" if delta > 0.005 else ("PIORA" if delta < -0.005 else "IGUAL")
                    out.append(f"    Δ({m} vs baseline) = {delta:+.4f} → {tag}")

    # ═══════════════════════════════════════════════════════
    # PARTE 3 — POR GÊNERO (correlações top)
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  PARTE 3 — TOP CORRELAÇÕES POR GÊNERO (IMDb)")
    out.append("=" * 70)

    all_genre_hits = []
    all_test_feats = emo_ok + char_ok + protag_ok

    for genre in GENRES:
        gsub = df[(df["genre"].str.contains(genre, na=False)) & (df["imdb_rating"].notna())]
        if len(gsub) < 30:
            continue

        hits = []
        for feat in all_test_feats:
            v = gsub[[feat, "imdb_rating"]].dropna()
            if len(v) < 25:
                continue
            rho, p = stats.spearmanr(v[feat], v["imdb_rating"])
            if p < 0.05:
                hits.append({"genre": genre, "feature": feat, "rho": rho, "p": p, "n": len(v)})
                all_genre_hits.append({"genre": genre, "feature": feat, "rho": rho, "p": p, "n": len(v)})

        if hits:
            out.append(f"\n  ── {genre} (n={len(gsub)}) — {len(hits)} efeitos sig. ──")
            for h in sorted(hits, key=lambda x: x["p"])[:8]:
                # Identificar grupo
                if h["feature"] in emo_ok:
                    grp = "EMO"
                elif h["feature"] in char_ok:
                    grp = "CHAR"
                else:
                    grp = "PROT"
                sig = "**" if h["p"] < 0.01 else "*"
                out.append(f"    [{grp}] {h['feature']:<35s}  ρ={h['rho']:+.3f}  p={h['p']:.4f} {sig}")

    # ═══════════════════════════════════════════════════════
    # PARTE 4 — COMPARAÇÃO ANTES vs DEPOIS
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  PARTE 4 — CONFIRMAÇÃO: PADRÕES SE MANTÊM?")
    out.append("=" * 70)

    # Achados anteriores que precisam ser confirmados
    previous = [
        ("Thriller", "gv5_autoloop", +0.332, "Estabilidade emocional"),
        ("Horror", "gv5_autoloop", +0.347, "Estabilidade emocional"),
        ("Thriller", "gv3_pr_neg", -0.293, "Centralidade negativa"),
        ("Comedy", "gv3_pr_neg", -0.212, "Centralidade negativa"),
        ("Horror", "F1_strength_centralization", +0.290, "Centralização da rede"),
        ("Horror", "F3_protag_presence", +0.275, "Presença do protagonista"),
        ("Crime", "protag_pct_negative", -0.274, "% falas negativas protagonista"),
    ]

    out.append(f"\n  {'Gênero':<12s} {'Feature':<35s} {'ρ antes':>8s} {'ρ agora':>8s} {'Confirma?':>10s}")
    out.append(f"  {'-'*80}")

    for genre, feat, old_rho, label in previous:
        if feat not in df.columns:
            out.append(f"  {genre:<12s} {label:<35s} {old_rho:>+8.3f}     —      N/D")
            continue

        gsub = df[(df["genre"].str.contains(genre, na=False)) & (df["imdb_rating"].notna())]
        v = gsub[[feat, "imdb_rating"]].dropna()
        if len(v) < 20:
            out.append(f"  {genre:<12s} {label:<35s} {old_rho:>+8.3f}     —      n<20")
            continue

        rho, p = stats.spearmanr(v[feat], v["imdb_rating"])
        same_direction = (rho > 0) == (old_rho > 0)
        sig = p < 0.05
        if same_direction and sig:
            status = "CONFIRMADO"
        elif same_direction:
            status = "DIREÇÃO OK"
        else:
            status = "INVERTEU"

        out.append(f"  {genre:<12s} {label:<35s} {old_rho:>+8.3f} {rho:>+8.3f} {status:>10s}  (p={p:.4f}, n={len(v)})")

    # ═══════════════════════════════════════════════════════
    # RESUMO
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  RESUMO EXECUTIVO")
    out.append("=" * 70)

    mr = pd.DataFrame(model_results)
    out.append(f"\n  Modelos (IMDb):")
    imdb_models = mr[mr["target"] == "IMDb"].sort_values("R2", ascending=False)
    for _, r in imdb_models.iterrows():
        out.append(f"    {r['model']:<15s}  R²={r['R2']:>7.4f}  ρ={r['Spearman']:>+7.3f}")

    out.append(f"\n  Efeitos por gênero (p<0.05): {len(all_genre_hits)}")
    genre_counts = pd.DataFrame(all_genre_hits)
    if not genre_counts.empty:
        for genre in GENRES:
            g_hits = genre_counts[genre_counts["genre"] == genre]
            if len(g_hits) > 0:
                out.append(f"    {genre}: {len(g_hits)} efeitos")

    report = "\n".join(out)
    print(report)

    (RESULTS / "expanded_analysis_report.txt").write_text(report, encoding="utf-8")
    mr.to_csv(RESULTS / "expanded_model_comparison.csv", index=False)
    if all_genre_hits:
        pd.DataFrame(all_genre_hits).to_csv(RESULTS / "expanded_genre_effects.csv", index=False)

    print(f"\nSalvo: expanded_analysis_report.txt + expanded_model_comparison.csv")


if __name__ == "__main__":
    main()
