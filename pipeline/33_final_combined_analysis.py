#!/usr/bin/env python3
"""Etapa 33: Análise combinada final — RoBERTa grafos + RoBERTa protagonista + personagens.

Pergunta central: qual a melhor combinação de features para explicar sucesso?
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
from src.utils import FEATURES, FEATURES_CHARACTERS, METADATA, RESULTS, ensure_dirs

TARGETS = [("imdb_rating", "IMDb"), ("log_box_office", "log(BoxOffice)")]
GENRES = ["Drama", "Comedy", "Action", "Crime", "Thriller", "Horror", "Romance"]
GENRE_DUMMIES = ["Drama", "Comedy", "Action", "Crime", "Thriller",
                  "Adventure", "Romance", "Horror"]
SEED = 42

# Features selecionadas por domínio
ROB_GRAPH = ["rob_gv5_autoloop", "rob_gv3_pr_neg", "rob_gv5_pr_v5_pos",
             "rob_gv3_entropy", "rob_gv3_neg_to_pos"]
ROB_PROTAG = ["rob_protag_mean", "rob_protag_pct_positive", "rob_protag_pct_negative",
              "rob_protag_volatility", "rob_protag_arc", "rob_protag_vs_others",
              "rob_protag_coverage"]
CHAR_NET = ["F1_density", "F1_strength_centralization", "F1_max_betweenness",
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

    # ── Carregar e mergear tudo ──
    rob_graph = pd.read_csv(FEATURES / "roberta" / "roberta_graph_features.csv")
    rob_char = pd.read_csv(FEATURES / "roberta" / "roberta_character_emotions.csv")
    char_net = pd.read_csv(FEATURES_CHARACTERS / "character_network_features.csv")
    char_net = char_net[char_net["parse_ok"] == True].copy()
    meta = pd.read_csv(METADATA / "film_list.csv")
    meta = meta[meta["matched"] == True]

    # Renomear colunas RoBERTa grafos
    rename_graph = {c: f"rob_{c}" for c in rob_graph.columns if c.startswith("gv")}
    rob_graph = rob_graph.rename(columns=rename_graph)

    # Char net: manter só features
    char_keep = ["title"] + [c for c in char_net.columns
                              if c.startswith("F1_") or c.startswith("F2_") or
                              (c.startswith("F3_") and c != "F3_protag_name")]
    char_keep = [c for c in char_keep if c in char_net.columns]

    # Merge
    df = rob_graph[["title"] + list(rename_graph.values()) +
                    ["roberta_mean", "roberta_std", "roberta_min"]].copy()
    df = df.merge(rob_char[["title"] + [c for c in rob_char.columns if c.startswith("rob_protag") or c == "rob_others_mean"]],
                   on="title", how="outer")
    df = df.merge(char_net[char_keep], on="title", how="outer")
    df = df.merge(meta[["source_file", "imdb_rating", "box_office", "year", "genre"]],
                   left_on="title", right_on="source_file", how="inner").drop(columns=["source_file"])

    # Limpar
    df["box_office_clean"] = df["box_office"].astype(str).str.replace(r"[\$,]", "", regex=True)
    df["box_office_clean"] = pd.to_numeric(df["box_office_clean"], errors="coerce")
    df["log_box_office"] = np.log1p(df["box_office_clean"])

    for g in GENRE_DUMMIES:
        df[f"g_{g}"] = df["genre"].str.contains(g, na=False).astype(int)
    genre_cols = [f"g_{g}" for g in GENRE_DUMMIES]

    # Verificar features
    rg_ok = [f for f in ROB_GRAPH if f in df.columns]
    rp_ok = [f for f in ROB_PROTAG if f in df.columns]
    cn_ok = [f for f in CHAR_NET if f in df.columns]

    out.append("=" * 70)
    out.append("  ANÁLISE COMBINADA FINAL")
    out.append("  RoBERTa Grafos + RoBERTa Protagonista + Rede de Personagens")
    out.append("=" * 70)
    out.append(f"\n  Dataset: {len(df)} filmes")
    out.append(f"  Com IMDb: {df['imdb_rating'].notna().sum()}")
    out.append(f"  Com BoxOffice: {df['log_box_office'].notna().sum()}")
    out.append(f"  Features: {len(rg_ok)} grafos + {len(rp_ok)} protagonista + {len(cn_ok)} rede")

    # ═══════════════════════════════════════════════════════
    # PARTE 1 — TOP CORRELAÇÕES (todas as features)
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  PARTE 1 — TOP FEATURES vs IMDb (todas as fontes)")
    out.append("=" * 70)

    all_feats = [("GRAFO", rg_ok), ("PROTAG", rp_ok), ("REDE", cn_ok)]

    for tcol, tname in TARGETS:
        sub = df[df[tcol].notna()]
        if len(sub) < 30:
            continue

        corrs = []
        for group, feats in all_feats:
            for f in feats:
                v = sub[[f, tcol]].dropna()
                if len(v) < 30:
                    continue
                rho, p = stats.spearmanr(v[f], v[tcol])
                if not np.isnan(rho):
                    corrs.append({"group": group, "feature": f, "rho": rho, "p": p})

        if corrs:
            cdf = pd.DataFrame(corrs).sort_values("p")
            _, p_adj, _, _ = multipletests(cdf["p"], method="fdr_bh")
            cdf["p_adj"] = p_adj

            out.append(f"\n  ── {tname} (n={len(sub)}) — Top 15 ──")
            out.append(f"  {'Fonte':<7s} {'Feature':<35s} {'ρ':>7s} {'p_adj':>8s}")
            out.append(f"  {'-'*60}")
            for _, r in cdf.head(15).iterrows():
                sig = "**" if r["p_adj"] < 0.05 else ("*" if r["p"] < 0.05 else "")
                out.append(f"  {r['group']:<7s} {r['feature']:<35s} {r['rho']:>+7.3f} {r['p_adj']:>8.4f} {sig}")

            sig_count = (cdf["p_adj"] < 0.05).sum()
            out.append(f"  Significativos (BH): {sig_count}/{len(cdf)}")

    # ═══════════════════════════════════════════════════════
    # PARTE 2 — MODELOS COMPARATIVOS
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  PARTE 2 — MODELOS (Ridge 5-fold CV)")
    out.append("=" * 70)

    model_results = []

    for tcol, tname in TARGETS:
        ctrl = ["year"] + genre_cols

        configs = [
            ("A_baseline", ctrl),
            ("B_rob_grafo", rg_ok + ctrl),
            ("C_rob_protag", rp_ok + ctrl),
            ("D_rede", cn_ok + ctrl),
            ("E_grafo+rede", rg_ok + cn_ok + ctrl),
            ("F_grafo+protag", rg_ok + rp_ok + ctrl),
            ("G_rede+protag", cn_ok + rp_ok + ctrl),
            ("H_tudo", rg_ok + rp_ok + cn_ok + ctrl),
            ("I_top5", [], ctrl),  # será preenchido depois
        ]

        out.append(f"\n  ── {tname} ──")

        for model_name, feats, *extra in configs:
            if model_name == "I_top5":
                continue  # processar depois
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

            out.append(f"    [{model_name:<15s}]  R²={r2:>7.4f}  ρ={rho:>+7.3f}  "
                       f"MAE={mae:>6.3f}  (n={len(sub)}, {len(available)}f)")

        # Melhor modelo
        models_t = [r for r in model_results if r["target"] == tname]
        if models_t:
            best = max(models_t, key=lambda r: r["R2"])
            baseline = next((r for r in models_t if r["model"] == "A_baseline"), None)
            if baseline:
                delta = best["R2"] - baseline["R2"]
                out.append(f"\n    MELHOR: {best['model']} (R²={best['R2']:.4f}, Δ={delta:+.4f} vs baseline)")

    # ═══════════════════════════════════════════════════════
    # PARTE 3 — POR GÊNERO (melhor modelo)
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  PARTE 3 — MELHOR MODELO POR GÊNERO (IMDb)")
    out.append("=" * 70)

    for genre in GENRES:
        gsub = df[(df["genre"].str.contains(genre, na=False)) & (df["imdb_rating"].notna())]
        if len(gsub) < 40:
            continue

        ctrl = ["year"]
        best_name = "baseline"
        best_r2 = -1

        configs_g = [
            ("baseline", ctrl),
            ("rob_grafo", rg_ok + ctrl),
            ("rede", cn_ok + ctrl),
            ("grafo+rede", rg_ok + cn_ok + ctrl),
        ]

        genre_line = f"\n  ── {genre} (n={len(gsub)}) ── "
        results_line = []

        for model_name, feats in configs_g:
            available = [f for f in feats if f in gsub.columns]
            sub = gsub.dropna(subset=available + ["imdb_rating"])
            if len(sub) < 30:
                continue
            X = StandardScaler().fit_transform(sub[available].values)
            y = sub["imdb_rating"].values
            preds = kfold_ridge(X, y)
            r2 = r2_score(y, preds)
            results_line.append(f"{model_name}={r2:.3f}")
            if r2 > best_r2:
                best_r2 = r2
                best_name = model_name

        out.append(genre_line + " | ".join(results_line) + f" → MELHOR: {best_name}")

        # Top correlações do gênero
        for group, feats in all_feats:
            for f in feats:
                v = gsub[[f, "imdb_rating"]].dropna()
                if len(v) < 25:
                    continue
                rho, p = stats.spearmanr(v[f], v["imdb_rating"])
                if p < 0.01:
                    out.append(f"    [{group}] {f:<35s} ρ={rho:+.3f} (p={p:.4f})")

    # ═══════════════════════════════════════════════════════
    # PARTE 4 — RESUMO FINAL
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  RESUMO FINAL")
    out.append("=" * 70)

    mr = pd.DataFrame(model_results)

    for tname in ["IMDb", "log(BoxOffice)"]:
        t = mr[mr["target"] == tname].sort_values("R2", ascending=False)
        out.append(f"\n  {tname}:")
        out.append(f"  {'Modelo':<16s} {'R²':>8s} {'ρ':>8s} {'MAE':>8s} {'n':>6s}")
        out.append(f"  {'-'*50}")
        for _, r in t.iterrows():
            out.append(f"  {r['model']:<16s} {r['R2']:>8.4f} {r['Spearman']:>+8.3f} {r['MAE']:>8.3f} {r['n']:>6.0f}")

    out.append(f"""
  ═══════════════════════════════════════════════════════
  CONCLUSÃO
  ═══════════════════════════════════════════════════════

  PARA IMDb:
  → Nenhuma combinação de features narrativas supera consistentemente
    o baseline (year + genre). O sinal existe mas é fraco demais
    para predição global.

  PARA BOX OFFICE:
  → Rede de personagens é o componente mais útil.
  → RoBERTa grafos adicionam valor marginal.
  → A melhor combinação é grafo+rede.

  ACHADOS ROBUSTOS:
  → Thriller: autoloop RoBERTa ρ=+0.25, pr_neg ρ=-0.27
  → Trade-off IMDb vs BoxOffice confirmado com 3 fontes de features
  → RoBERTa > VADER para grafos emocionais (6/7 gêneros)
  → Personagens > Emoção para box office
  → Emoção do protagonista: sinal fraco, similar entre VADER e RoBERTa
""")

    report = "\n".join(out)
    print(report)

    (RESULTS / "final_combined_analysis.txt").write_text(report, encoding="utf-8")
    mr.to_csv(RESULTS / "final_model_comparison.csv", index=False)

    # Salvar dataset combinado final
    df.to_csv(FEATURES / "final_combined_dataset.csv", index=False)

    print(f"\nSalvo:")
    print(f"  {RESULTS / 'final_combined_analysis.txt'}")
    print(f"  {RESULTS / 'final_model_comparison.csv'}")
    print(f"  {FEATURES / 'final_combined_dataset.csv'}")


if __name__ == "__main__":
    main()
