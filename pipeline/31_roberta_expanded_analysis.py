#!/usr/bin/env python3
"""Etapa 31: Análise expandida com RoBERTa — correlações, modelos, gêneros.

Mesma estrutura do script 29 mas usando features RoBERTa.
Compara diretamente VADER vs RoBERTa em cada teste.
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
from src.utils import FEATURES, FEATURES_EMOTIONAL, FEATURES_CHARACTERS, METADATA, RESULTS, ensure_dirs

TARGETS = [("imdb_rating", "IMDb"), ("log_box_office", "log(BoxOffice)")]
GENRES = ["Drama", "Comedy", "Action", "Crime", "Thriller", "Horror", "Romance"]
GENRE_DUMMIES = ["Drama", "Comedy", "Action", "Crime", "Thriller",
                  "Adventure", "Romance", "Horror"]
SEED = 42

# Features selecionadas (mesmas do script 29 para comparação justa)
EMO_SELECTED = ["gv3_pr_neg", "gv5_pr_v5_pos", "gv5_autoloop", "gv3_entropy", "gv3_neg_to_pos"]
CHAR_SELECTED = ["F1_density", "F1_strength_centralization", "F1_max_betweenness",
                  "F1_avg_clustering", "F1_modularity", "F2_W_entropy",
                  "F2_D_mean", "F3_protag_strength_frac", "F3_protag_presence"]


def kfold_ridge(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    preds = np.zeros(len(y))
    for tr, te in kf.split(X):
        m = Ridge(alpha=1.0, random_state=SEED)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


def main():
    ensure_dirs()
    out = []

    # ── Carregar dados ──
    roberta = pd.read_csv(FEATURES / "roberta" / "roberta_graph_features.csv")
    vader = pd.read_csv(FEATURES_EMOTIONAL / "refined_graph_features.csv")
    char = pd.read_csv(FEATURES_CHARACTERS / "character_network_features.csv")
    char = char[char["parse_ok"] == True].copy()
    meta = pd.read_csv(METADATA / "film_list.csv")
    meta = meta[meta["matched"] == True]

    # Renomear colunas RoBERTa para distinguir
    rob_feat_cols = [c for c in roberta.columns if c.startswith("gv")]
    rob_renamed = {c: f"rob_{c}" for c in rob_feat_cols}
    roberta = roberta.rename(columns=rob_renamed)
    rob_selected = [f"rob_{c}" for c in EMO_SELECTED]

    # Merge tudo
    df = vader[["title"] + [c for c in vader.columns if c.startswith("gv")]].merge(
        roberta[["title"] + list(rob_renamed.values()) + ["roberta_mean", "roberta_std", "roberta_min"]],
        on="title", how="inner"
    )
    char_cols_keep = ["title"] + [c for c in char.columns if c.startswith("F1_") or c.startswith("F2_") or
                                   (c.startswith("F3_") and c != "F3_protag_name")]
    char_cols_keep = [c for c in char_cols_keep if c in char.columns]
    df = df.merge(char[char_cols_keep], on="title", how="inner")
    df = df.merge(meta[["source_file", "imdb_rating", "box_office", "year", "genre"]],
                   left_on="title", right_on="source_file", how="inner").drop(columns=["source_file"])

    # Limpar box office
    df["box_office_clean"] = df["box_office"].astype(str).str.replace(r"[\$,]", "", regex=True)
    df["box_office_clean"] = pd.to_numeric(df["box_office_clean"], errors="coerce")
    df["log_box_office"] = np.log1p(df["box_office_clean"])

    # Genre dummies
    for g in GENRE_DUMMIES:
        df[f"g_{g}"] = df["genre"].str.contains(g, na=False).astype(int)
    genre_cols = [f"g_{g}" for g in GENRE_DUMMIES]

    vader_ok = [c for c in EMO_SELECTED if c in df.columns]
    rob_ok = [c for c in rob_selected if c in df.columns]
    char_ok = [c for c in CHAR_SELECTED if c in df.columns]

    out.append("=" * 70)
    out.append("  ANÁLISE ROBERTA vs VADER — CORPUS EXPANDIDO")
    out.append("=" * 70)
    out.append(f"\n  Dataset: {len(df)} filmes (com VADER + RoBERTa + personagens + metadata)")
    out.append(f"  Com IMDb: {df['imdb_rating'].notna().sum()}")

    # ═══════════════════════════════════════════════════════
    # PARTE 1 — CORRELAÇÕES: VADER vs RoBERTa (head-to-head)
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  PARTE 1 — HEAD-TO-HEAD: VADER vs RoBERTa vs IMDb")
    out.append("=" * 70)

    for tcol, tname in TARGETS:
        sub = df[df[tcol].notna()]
        if len(sub) < 30:
            continue

        out.append(f"\n  ── {tname} (n={len(sub)}) ──")
        out.append(f"  {'Feature':<25s} {'VADER ρ':>10s} {'RoBERTa ρ':>10s} {'Melhor':>10s}")
        out.append(f"  {'-'*60}")

        for v_feat, r_feat in zip(vader_ok, rob_ok):
            v_valid = sub[[v_feat, tcol]].dropna()
            r_valid = sub[[r_feat, tcol]].dropna()

            rho_v, p_v = stats.spearmanr(v_valid[v_feat], v_valid[tcol]) if len(v_valid) > 30 else (0, 1)
            rho_r, p_r = stats.spearmanr(r_valid[r_feat], r_valid[tcol]) if len(r_valid) > 30 else (0, 1)

            melhor = "RoBERTa" if abs(rho_r) > abs(rho_v) else "VADER"
            sig_v = "*" if p_v < 0.05 else ""
            sig_r = "*" if p_r < 0.05 else ""
            feat_name = v_feat.replace("gv3_", "").replace("gv5_", "").replace("pr_v5_", "pr_")

            out.append(f"  {feat_name:<25s} {rho_v:>+9.3f}{sig_v} {rho_r:>+9.3f}{sig_r} {melhor:>10s}")

    # ═══════════════════════════════════════════════════════
    # PARTE 2 — MODELOS COMPARATIVOS
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  PARTE 2 — MODELOS COMPARATIVOS (Ridge 5-fold CV)")
    out.append("=" * 70)

    model_results = []

    for tcol, tname in TARGETS:
        ctrl = ["year"] + genre_cols

        configs = [
            ("A_baseline", ctrl),
            ("B_vader", vader_ok + ctrl),
            ("C_roberta", rob_ok + ctrl),
            ("D_personagens", char_ok + ctrl),
            ("E_vader+char", vader_ok + char_ok + ctrl),
            ("F_roberta+char", rob_ok + char_ok + ctrl),
            ("G_tudo", vader_ok + rob_ok + char_ok + ctrl),
        ]

        out.append(f"\n  ── {tname} ──")

        for model_name, feats in configs:
            available = [f for f in feats if f in df.columns]
            sub = df.dropna(subset=available + [tcol])
            if len(sub) < 50:
                continue

            X = StandardScaler().fit_transform(sub[available].values)
            y = sub[tcol].values
            preds = kfold_ridge(X, y)

            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            rho, _ = stats.spearmanr(y, preds)

            model_results.append({
                "target": tname, "model": model_name, "n": len(sub),
                "R2": round(r2, 4), "MAE": round(mae, 4),
                "Spearman": round(rho, 4), "n_features": len(available),
            })

            out.append(f"    [{model_name:<15s}]  n={len(sub):>5d}  R²={r2:>7.4f}  "
                       f"MAE={mae:>6.3f}  ρ={rho:>+7.3f}  ({len(available)} feats)")

        # Deltas vs baseline
        models = {r["model"]: r for r in model_results if r["target"] == tname}
        if "A_baseline" in models:
            base = models["A_baseline"]["R2"]
            for m in ["B_vader", "C_roberta", "F_roberta+char"]:
                if m in models:
                    delta = models[m]["R2"] - base
                    out.append(f"    Δ({m} vs baseline) = {delta:+.4f}")

    # ═══════════════════════════════════════════════════════
    # PARTE 3 — POR GÊNERO (head-to-head VADER vs RoBERTa)
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  PARTE 3 — POR GÊNERO: VADER vs RoBERTa (IMDb)")
    out.append("=" * 70)

    genre_results = []

    for genre in GENRES:
        gsub = df[(df["genre"].str.contains(genre, na=False)) & (df["imdb_rating"].notna())]
        if len(gsub) < 30:
            continue

        out.append(f"\n  ── {genre} (n={len(gsub)}) ──")
        vader_wins = 0
        roberta_wins = 0

        for v_feat, r_feat in zip(vader_ok, rob_ok):
            v_valid = gsub[[v_feat, "imdb_rating"]].dropna()
            r_valid = gsub[[r_feat, "imdb_rating"]].dropna()
            if len(v_valid) < 20 or len(r_valid) < 20:
                continue

            rho_v, p_v = stats.spearmanr(v_valid[v_feat], v_valid["imdb_rating"])
            rho_r, p_r = stats.spearmanr(r_valid[r_feat], r_valid["imdb_rating"])

            if abs(rho_r) > abs(rho_v):
                roberta_wins += 1
            else:
                vader_wins += 1

            # Só mostrar se pelo menos um é significativo
            if p_v < 0.05 or p_r < 0.05:
                feat_name = v_feat.replace("gv3_", "").replace("gv5_", "").replace("pr_v5_", "pr_")
                melhor = "ROB" if abs(rho_r) > abs(rho_v) else "VAD"
                out.append(f"    {feat_name:<20s}  V={rho_v:+.3f}{'*' if p_v<0.05 else ' '}  "
                           f"R={rho_r:+.3f}{'*' if p_r<0.05 else ' '}  → {melhor}")

                genre_results.append({
                    "genre": genre, "feature": feat_name,
                    "vader_rho": rho_v, "vader_p": p_v,
                    "roberta_rho": rho_r, "roberta_p": p_r,
                })

        out.append(f"    Score: VADER {vader_wins} vs RoBERTa {roberta_wins}")

    # ═══════════════════════════════════════════════════════
    # PARTE 4 — CONFIRMAÇÃO DOS ACHADOS ANTERIORES
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  PARTE 4 — ACHADOS ANTERIORES: CONFIRMAM COM RoBERTa?")
    out.append("=" * 70)

    checks = [
        ("Thriller", "gv5_autoloop", "rob_gv5_autoloop", "Estabilidade emocional"),
        ("Thriller", "gv3_pr_neg", "rob_gv3_pr_neg", "Centralidade negativa"),
        ("Horror", "gv5_autoloop", "rob_gv5_autoloop", "Estabilidade emocional"),
        ("Comedy", "gv3_pr_neg", "rob_gv3_pr_neg", "Centralidade negativa"),
    ]

    out.append(f"\n  {'Gênero':<12s} {'Feature':<25s} {'VADER ρ':>10s} {'RoBERTa ρ':>10s} {'Status':>12s}")
    out.append(f"  {'-'*75}")

    for genre, v_feat, r_feat, label in checks:
        gsub = df[(df["genre"].str.contains(genre, na=False)) & (df["imdb_rating"].notna())]

        rho_v, p_v = (0, 1)
        rho_r, p_r = (0, 1)

        v_valid = gsub[[v_feat, "imdb_rating"]].dropna()
        if len(v_valid) > 20:
            rho_v, p_v = stats.spearmanr(v_valid[v_feat], v_valid["imdb_rating"])

        r_valid = gsub[[r_feat, "imdb_rating"]].dropna()
        if len(r_valid) > 20:
            rho_r, p_r = stats.spearmanr(r_valid[r_feat], r_valid["imdb_rating"])

        # Confirma se mesma direção e pelo menos um significativo
        same_dir = (rho_v > 0) == (rho_r > 0) if rho_v != 0 and rho_r != 0 else False
        either_sig = p_v < 0.05 or p_r < 0.05
        stronger = "ROB" if abs(rho_r) > abs(rho_v) else "VAD"

        if same_dir and either_sig:
            status = f"CONFIRMA ({stronger})"
        elif same_dir:
            status = "mesma dir."
        else:
            status = "DIVERGE"

        out.append(f"  {genre:<12s} {label:<25s} {rho_v:>+10.3f} {rho_r:>+10.3f} {status:>12s}")

    # ═══════════════════════════════════════════════════════
    # RESUMO
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  RESUMO EXECUTIVO")
    out.append("=" * 70)

    mr = pd.DataFrame(model_results)
    out.append(f"\n  Tabela de modelos:")
    out.append(f"  {'Target':<16s} {'Modelo':<16s} {'R²':>8s} {'ρ':>8s}")
    out.append(f"  {'-'*50}")
    for _, r in mr.sort_values(["target", "R2"], ascending=[True, False]).iterrows():
        out.append(f"  {r['target']:<16s} {r['model']:<16s} {r['R2']:>8.4f} {r['Spearman']:>+8.3f}")

    report = "\n".join(out)
    print(report)

    (RESULTS / "roberta_expanded_analysis.txt").write_text(report, encoding="utf-8")
    mr.to_csv(RESULTS / "roberta_model_comparison.csv", index=False)
    if genre_results:
        pd.DataFrame(genre_results).to_csv(RESULTS / "roberta_genre_effects.csv", index=False)

    print(f"\nSalvo: roberta_expanded_analysis.txt + roberta_model_comparison.csv")


if __name__ == "__main__":
    main()
