#!/usr/bin/env python3
"""Etapa 23: Merge personagens + emoção e testar poder combinado.

Pergunta central:
  Features de rede de personagens + grafos emocionais juntos
  explicam mais que cada um sozinho?
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
from src.utils import FEATURES, FEATURES_EMOTIONAL, FEATURES_CHARACTERS, METADATA, RESULTS, ensure_dirs

TARGETS = [
    ("imdb_rating", "IMDb"),
    ("log_box_office", "log(BoxOffice)"),
    ("log_revenue_multiple", "log(RevMultiple)"),
]
GENRES = ["Thriller", "Horror", "Comedy", "Drama", "Action", "Crime"]
GENRE_DUMMIES = ["Drama", "Comedy", "Action", "Crime", "Thriller",
                  "Adventure", "Romance", "Horror"]
SEED = 42

# Features selecionadas de cada domínio (não usar tudo — evitar overfitting)
EMOTION_FEATS = ["gv3_pr_neg", "gv5_pr_v5_pos", "gv5_autoloop", "gv3_entropy", "gv3_neg_to_pos"]
CHAR_FEATS = ["F1_n_chars", "F1_density", "F1_strength_centralization", "F1_max_betweenness",
              "F1_avg_clustering", "F1_modularity", "F2_W_entropy", "F2_C_trend",
              "F2_D_mean", "F3_protag_strength_frac", "F3_protag_presence"]


def loo_ridge(X, y, alpha=1.0):
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = Ridge(alpha=alpha, random_state=SEED)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


def main():
    ensure_dirs()
    out = []

    # ═══════════════════════════════════════════════════════
    # CARREGAR E MERGEAR DATASETS
    # ═══════════════════════════════════════════════════════
    emo = pd.read_csv(FEATURES_EMOTIONAL / "refined_graph_features.csv")
    char = pd.read_csv(FEATURES_CHARACTERS / "character_network_features.csv")
    char = char[char["parse_ok"] == True].drop(columns=["parse_ok", "error", "F3_protag_name"], errors="ignore")
    meta = pd.read_csv(METADATA / "film_list.csv")
    meta = meta[meta["matched"] == True]

    # Merge
    df = emo.merge(char, on="title", how="inner")
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

    # ROI se disponível
    roi_path = FEATURES / "analysis_dataset_with_roi.csv"
    if roi_path.exists():
        roi_df = pd.read_csv(roi_path)
        if "log_revenue_multiple" in roi_df.columns:
            roi_map = roi_df.set_index("title")["log_revenue_multiple"]
            df["log_revenue_multiple"] = df["title"].map(roi_map)

    # Genre dummies
    for g in GENRE_DUMMIES:
        df[f"g_{g}"] = df["genre"].str.contains(g, na=False).astype(int)
    genre_cols = [f"g_{g}" for g in GENRE_DUMMIES]

    out.append("=" * 70)
    out.append("  ANÁLISE COMBINADA: PERSONAGENS + EMOÇÃO")
    out.append("=" * 70)
    out.append(f"\n  Dataset combinado: {len(df)} filmes")
    out.append(f"  Features emocionais: {len(EMOTION_FEATS)}")
    out.append(f"  Features de personagens: {len(CHAR_FEATS)}")
    out.append(f"  Com IMDb: {df['imdb_rating'].notna().sum()}")
    out.append(f"  Com BoxOffice: {df['log_box_office'].notna().sum()}")

    # Verificar quais features existem
    emo_ok = [f for f in EMOTION_FEATS if f in df.columns]
    char_ok = [f for f in CHAR_FEATS if f in df.columns]
    out.append(f"\n  Features emocionais disponíveis: {len(emo_ok)}/{len(EMOTION_FEATS)}")
    out.append(f"  Features de personagens disponíveis: {len(char_ok)}/{len(CHAR_FEATS)}")

    if not emo_ok or not char_ok:
        out.append("  ERRO: features insuficientes")
        print("\n".join(out))
        return

    # ═══════════════════════════════════════════════════════
    # CORRELAÇÕES DAS FEATURES DE PERSONAGENS
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  CORRELAÇÕES: FEATURES DE PERSONAGENS vs TARGETS")
    out.append("=" * 70)

    for tcol, tname in TARGETS:
        sub = df[df[tcol].notna()]
        if len(sub) < 30:
            continue
        out.append(f"\n  ── {tname} (n={len(sub)}) ──")

        corrs = []
        for feat in char_ok:
            v = sub[[feat, tcol]].dropna()
            if len(v) < 30:
                continue
            rho, p = stats.spearmanr(v[feat], v[tcol])
            if not np.isnan(rho):
                corrs.append({"feature": feat, "rho": rho, "p": p})

        if corrs:
            cdf = pd.DataFrame(corrs).sort_values("p")
            _, p_adj, _, _ = multipletests(cdf["p"], method="fdr_bh")
            cdf["p_adj"] = p_adj

            for _, r in cdf.head(8).iterrows():
                sig = "**" if r["p_adj"] < 0.05 else ("*" if r["p"] < 0.05 else "")
                out.append(f"    {r['feature']:<35s}  ρ={r['rho']:+.3f}  p={r['p']:.4f}  p_adj={r['p_adj']:.4f} {sig}")

            sig_count = (cdf["p"] < 0.05).sum()
            sig_adj = (cdf["p_adj"] < 0.05).sum()
            out.append(f"    Significativos: {sig_count} (p<0.05), {sig_adj} (p_adj<0.05)")

    # ═══════════════════════════════════════════════════════
    # MODELOS COMPARATIVOS (LOO-CV Ridge)
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  MODELOS COMPARATIVOS (Ridge LOO-CV)")
    out.append("=" * 70)
    out.append("  4 modelos por target:")
    out.append("    [A] Baseline: year + genre")
    out.append("    [B] Só emoção: 5 features emocionais")
    out.append("    [C] Só personagens: 11 features de rede")
    out.append("    [D] Combinado: emoção + personagens")

    model_results = []

    for tcol, tname in TARGETS:
        ctrl = ["year"] + genre_cols
        if tcol != "imdb_rating":
            ctrl_budget = ctrl + ["log_budget"] if "log_budget" in df.columns else ctrl
        else:
            ctrl_budget = ctrl

        out.append(f"\n  ── {tname} ──")

        configs = [
            ("A_baseline", ctrl_budget),
            ("B_emocao", emo_ok + ctrl_budget),
            ("C_personagens", char_ok + ctrl_budget),
            ("D_combinado", emo_ok + char_ok + ctrl_budget),
        ]

        for model_name, feats in configs:
            sub = df[feats + [tcol]].dropna()
            if len(sub) < 30:
                continue

            X = StandardScaler().fit_transform(sub[feats].values)
            y = sub[tcol].values
            preds = loo_ridge(X, y)
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            rho, _ = stats.spearmanr(y, preds)

            model_results.append({
                "target": tname, "model": model_name, "n": len(sub),
                "R2": round(r2, 4), "MAE": round(mae, 4), "Spearman": round(rho, 4),
                "n_features": len(feats),
            })

            out.append(f"    [{model_name:<15s}]  n={len(sub):>4d}  R²={r2:>7.4f}  "
                       f"MAE={mae:>6.4f}  ρ={rho:>+7.4f}  ({len(feats)} feats)")

        # Delta
        models = {r["model"]: r for r in model_results if r["target"] == tname}
        if "A_baseline" in models and "D_combinado" in models:
            delta = models["D_combinado"]["R2"] - models["A_baseline"]["R2"]
            out.append(f"    ΔR² (combinado vs baseline) = {delta:+.4f}")
            if delta > 0.01:
                out.append(f"    → Combinação MELHORA o baseline")
            elif delta > -0.01:
                out.append(f"    → Sem diferença significativa")
            else:
                out.append(f"    → Combinação NÃO melhora (possível overfitting)")

        if "B_emocao" in models and "C_personagens" in models:
            diff = models["C_personagens"]["R2"] - models["B_emocao"]["R2"]
            melhor = "personagens" if diff > 0 else "emoção"
            out.append(f"    Personagens vs Emoção: ΔR² = {diff:+.4f} → {melhor} é melhor sozinho")

    # ═══════════════════════════════════════════════════════
    # ANÁLISE POR GÊNERO
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  ANÁLISE POR GÊNERO (Spearman, features de personagens vs IMDb)")
    out.append("=" * 70)

    genre_hits = []
    for genre in GENRES:
        gsub = df[(df["genre"].str.contains(genre, na=False)) & (df["imdb_rating"].notna())]
        if len(gsub) < 25:
            continue

        out.append(f"\n  ── {genre} (n={len(gsub)}) ──")
        for feat in char_ok:
            v = gsub[[feat, "imdb_rating"]].dropna()
            if len(v) < 20:
                continue
            rho, p = stats.spearmanr(v[feat], v["imdb_rating"])
            if p < 0.10:
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "~")
                out.append(f"    {feat:<35s}  ρ={rho:+.3f}  p={p:.4f} {sig}")
                genre_hits.append({"genre": genre, "feature": feat, "rho": rho, "p": p})

    if genre_hits:
        out.append(f"\n  Resumo: {len([g for g in genre_hits if g['p'] < 0.05])} efeitos significativos")

    # ═══════════════════════════════════════════════════════
    # TOP FEATURES DE PERSONAGENS (global)
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  TOP FEATURES DE PERSONAGENS — O QUE SIGNIFICAM")
    out.append("=" * 70)

    imdb_sub = df[df["imdb_rating"].notna()]
    top_char = []
    for feat in char_ok:
        v = imdb_sub[[feat, "imdb_rating"]].dropna()
        if len(v) > 30:
            rho, p = stats.spearmanr(v[feat], v["imdb_rating"])
            top_char.append({"feature": feat, "rho": rho, "p": p})

    top_char.sort(key=lambda x: abs(x["rho"]), reverse=True)
    labels = {
        "F1_n_chars": "Número de personagens",
        "F1_density": "Densidade da rede (todos se conhecem?)",
        "F1_strength_centralization": "Centralização (um personagem domina?)",
        "F1_max_betweenness": "Intermediação máxima (ponte entre grupos?)",
        "F1_avg_clustering": "Clustering (personagens formam grupos?)",
        "F1_modularity": "Modularidade (grupos bem separados?)",
        "F2_W_entropy": "Entropia de interações (distribuição uniforme?)",
        "F2_C_trend": "Tendência da centralização (rede centraliza no final?)",
        "F2_D_mean": "Mudança média de elenco entre cenas",
        "F3_protag_strength_frac": "Dominância do protagonista (fração da força)",
        "F3_protag_presence": "Presença do protagonista (% de cenas)",
    }

    for tc in top_char[:8]:
        feat = tc["feature"]
        label = labels.get(feat, feat)
        sig = "*" if tc["p"] < 0.05 else ""
        direction = "↑" if tc["rho"] > 0 else "↓"
        out.append(f"  {direction} {label}")
        out.append(f"    {feat}: ρ={tc['rho']:+.3f} (p={tc['p']:.4f}) {sig}")
        # Interpretação
        if tc["rho"] > 0.05 and tc["p"] < 0.05:
            out.append(f"    → Mais {label.lower()} = melhor IMDb")
        elif tc["rho"] < -0.05 and tc["p"] < 0.05:
            out.append(f"    → Mais {label.lower()} = pior IMDb")
        out.append("")

    # ═══════════════════════════════════════════════════════
    # INTERPRETAÇÃO FINAL
    # ═══════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  INTERPRETAÇÃO")
    out.append("=" * 70)

    mr = pd.DataFrame(model_results)
    out.append(f"""
  COMPARAÇÃO DE MODELOS (resumo):
  {'Target':<18s} {'Baseline':>8s} {'Emoção':>8s} {'Person.':>8s} {'Combin.':>8s}
  {'-'*55}""")

    for tname in ["IMDb", "log(BoxOffice)", "log(RevMultiple)"]:
        t = mr[mr["target"] == tname]
        vals = []
        for m in ["A_baseline", "B_emocao", "C_personagens", "D_combinado"]:
            row = t[t["model"] == m]
            vals.append(f"{row.iloc[0]['R2']:>8.4f}" if len(row) > 0 else f"{'—':>8s}")
        out.append(f"  {tname:<18s} {vals[0]} {vals[1]} {vals[2]} {vals[3]}")

    out.append(f"""
  CONCLUSÃO:
  - Se D > A: a combinação emoção + personagens melhora o baseline
  - Se C > B: personagens sozinhos explicam mais que emoção sozinha
  - Se D ≈ A: nenhuma feature narrativa adiciona valor preditivo
  - Se D < A: overfitting — muitas features, pouco sinal""")

    report = "\n".join(out)
    print(report)

    (RESULTS / "combined_analysis_report.txt").write_text(report, encoding="utf-8")
    mr.to_csv(RESULTS / "combined_model_comparison.csv", index=False)
    if genre_hits:
        pd.DataFrame(genre_hits).to_csv(RESULTS / "character_genre_effects.csv", index=False)

    # Salvar dataset combinado
    combined_out = FEATURES / "combined_dataset.csv"
    df.to_csv(combined_out, index=False)

    print(f"\nSalvo:")
    print(f"  {RESULTS / 'combined_analysis_report.txt'}")
    print(f"  {RESULTS / 'combined_model_comparison.csv'}")
    print(f"  {FEATURES / 'combined_dataset.csv'}")


if __name__ == "__main__":
    main()
