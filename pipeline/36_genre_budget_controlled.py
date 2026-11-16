#!/usr/bin/env python3
"""Etapa 36: Análise por gênero controlada por orçamento.

Pergunta central: os efeitos narrativos específicos por gênero sobrevivem
quando controlamos por ano + log_budget (proxy de marketing)?

Estratégia:
  Para cada gênero-alvo (Thriller, Horror, Crime, Comedy, Drama, Action):
    1. Filtrar filmes do gênero COM budget disponível.
    2. Correlação BRUTA (Spearman) de cada feature com IMDb / log_box_office.
    3. Correlação PARCIAL controlando por ano + log_budget (resíduos).
    4. Reportar direção, magnitude e p-value com correção FDR-BH por gênero.

Interpretação:
  * Se uma feature mantém |ρ_parcial| > 0.15 com p<0.05 dentro do gênero, o
    efeito narrativo é robusto a marketing — contribuição publicável.
  * Se o efeito desaparece ao controlar budget, era marketing disfarçado.
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
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from src.utils import FEATURES, RESULTS, ensure_dirs

TARGETS = [("imdb_rating", "IMDb"), ("log_box_office", "log(BoxOffice)")]
GENRES_FOCUS = ["Thriller", "Horror", "Crime", "Comedy", "Drama", "Action"]
SEED = 42

ROB_GRAPH = ["rob_gv5_autoloop", "rob_gv3_pr_neg", "rob_gv5_pr_v5_pos",
             "rob_gv3_entropy"]
ROB_PROTAG = ["rob_protag_mean", "rob_protag_pct_positive",
              "rob_protag_pct_negative", "rob_protag_volatility",
              "rob_protag_arc", "rob_protag_coverage"]
CHAR_NET = ["F1_density", "F1_strength_centralization", "F1_max_betweenness",
            "F1_avg_clustering", "F1_modularity", "F2_W_entropy",
            "F3_protag_strength_frac", "F3_protag_presence"]


def partial_spearman(df, feat, target, controls):
    """Correlação parcial (Spearman) via resíduos Ridge.

    Regride target ~ controls  → resíduo_y.
    Regride feat   ~ controls  → resíduo_x.
    Correlação de Spearman(resíduo_x, resíduo_y).
    """
    sub = df.dropna(subset=[feat, target] + controls).copy()
    if len(sub) < 30:
        return None
    Xc = StandardScaler().fit_transform(sub[controls].values)
    y = sub[target].values
    x = sub[feat].values

    resy = y - Ridge(alpha=1.0, random_state=SEED).fit(Xc, y).predict(Xc)
    resx = x - Ridge(alpha=1.0, random_state=SEED).fit(Xc, x).predict(Xc)

    r = stats.spearmanr(resx, resy)
    return {"feature": feat, "rho_partial": r.statistic, "p": r.pvalue,
            "n": len(sub)}


def raw_spearman(df, feat, target):
    sub = df.dropna(subset=[feat, target])
    if len(sub) < 30:
        return None
    r = stats.spearmanr(sub[feat], sub[target])
    return {"feature": feat, "rho_raw": r.statistic, "p_raw": r.pvalue,
            "n": len(sub)}


def main():
    ensure_dirs()
    out = []

    # ── Dados ──
    final = pd.read_csv(FEATURES / "final_combined_dataset.csv")
    roi = pd.read_csv(FEATURES / "analysis_dataset_with_roi.csv")
    df = final.merge(roi[["title", "budget", "log_budget"]],
                     on="title", how="left")

    # Filtrar apenas filmes com budget
    df_bud = df[df["log_budget"].notna()].copy()

    narr_all = [f for f in ROB_GRAPH + ROB_PROTAG + CHAR_NET if f in df_bud.columns]

    out.append("=" * 74)
    out.append("  ANÁLISE POR GÊNERO — CONTROLADA POR ORÇAMENTO (pipeline/36)")
    out.append("=" * 74)
    out.append(f"\n  Subamostra com budget: {len(df_bud)} filmes")
    out.append(f"  Features narrativas testadas: {len(narr_all)}")
    out.append(f"  Controles: year + log_budget")

    # Tamanho de amostra por gênero
    out.append("\n  ── Tamanhos amostrais por gênero (com budget) ──")
    sizes = []
    for g in GENRES_FOCUS:
        col = f"g_{g}"
        if col not in df_bud.columns:
            continue
        n = int(df_bud[col].sum())
        n_imdb = int((df_bud[col] == 1).sum() and df_bud[df_bud[col]==1]["imdb_rating"].notna().sum())
        n_bo = int((df_bud[col] == 1).sum() and df_bud[df_bud[col]==1]["log_box_office"].notna().sum())
        sizes.append((g, n, n_imdb, n_bo))
        out.append(f"    {g:<12s}  n={n:>3d}  (IMDb={n_imdb}, BoxOffice={n_bo})")

    # ═══════════════════════════════════════════════════════════════
    # Análise por gênero × target
    # ═══════════════════════════════════════════════════════════════
    all_rows = []

    for genre in GENRES_FOCUS:
        col = f"g_{genre}"
        if col not in df_bud.columns:
            continue
        gdf = df_bud[df_bud[col] == 1].copy()
        if len(gdf) < 40:
            out.append(f"\n  ⚠ {genre}: n={len(gdf)} insuficiente, pulando.")
            continue

        out.append(f"\n{'='*74}")
        out.append(f"  GÊNERO: {genre.upper()}   (n={len(gdf)} com budget)")
        out.append("=" * 74)

        for tcol, tname in TARGETS:
            n_t = gdf[tcol].notna().sum()
            if n_t < 40:
                out.append(f"\n  {tname}: n={n_t} insuficiente, pulando.")
                continue

            rows = []
            for feat in narr_all:
                raw = raw_spearman(gdf, feat, tcol)
                par = partial_spearman(gdf, feat, tcol,
                                       controls=["year", "log_budget"])
                if raw and par:
                    rows.append({
                        "genre": genre, "target": tname, "feature": feat,
                        "n": raw["n"],
                        "rho_raw": raw["rho_raw"], "p_raw": raw["p_raw"],
                        "rho_partial": par["rho_partial"], "p_partial": par["p"],
                    })

            if not rows:
                continue
            rdf = pd.DataFrame(rows).sort_values("p_partial")
            _, p_adj_raw, _, _ = multipletests(rdf["p_raw"], method="fdr_bh")
            _, p_adj_par, _, _ = multipletests(rdf["p_partial"], method="fdr_bh")
            rdf["p_adj_raw"] = p_adj_raw
            rdf["p_adj_partial"] = p_adj_par

            out.append(f"\n  ── {tname} (n={n_t}) ── "
                       "[ρ_bruta → ρ_parcial após ano+log_budget]")
            out.append(f"    {'Feature':<34s} "
                       f"{'ρ_bru':>7s} {'p_adj':>7s}   "
                       f"{'ρ_par':>7s} {'p_adj':>7s}")
            out.append(f"    {'-'*72}")

            for _, r in rdf.iterrows():
                sig_raw = "**" if r["p_adj_raw"] < 0.05 else (
                          "*" if r["p_raw"] < 0.05 else "")
                sig_par = "**" if r["p_adj_partial"] < 0.05 else (
                          "*" if r["p_partial"] < 0.05 else "")
                # Indicador de robustez: efeito sobrevive a controle?
                survive = ""
                if r["p_raw"] < 0.05 and r["p_partial"] < 0.05:
                    if abs(r["rho_partial"]) >= 0.8 * abs(r["rho_raw"]):
                        survive = "  ✓ robusto"
                    else:
                        survive = "  ~ atenuado"
                elif r["p_raw"] < 0.05 and r["p_partial"] >= 0.05:
                    survive = "  ✗ marketing"

                out.append(f"    {r['feature']:<34s} "
                           f"{r['rho_raw']:>+7.3f} {r['p_adj_raw']:>7.3f}{sig_raw:<2s} "
                           f"{r['rho_partial']:>+7.3f} {r['p_adj_partial']:>7.3f}{sig_par:<2s}"
                           f"{survive}")

            # Resumo do gênero × target
            n_sig_raw = (rdf["p_raw"] < 0.05).sum()
            n_sig_par = (rdf["p_partial"] < 0.05).sum()
            n_robust = ((rdf["p_raw"] < 0.05) &
                        (rdf["p_partial"] < 0.05) &
                        (rdf["rho_partial"].abs() >= 0.8 *
                         rdf["rho_raw"].abs())).sum()
            out.append(f"\n    Resumo: {n_sig_raw}/{len(rdf)} sig. brutas  "
                       f"→  {n_sig_par}/{len(rdf)} sig. parciais  "
                       f"→  {n_robust} robustas a marketing")

            all_rows.extend(rdf.to_dict("records"))

    # ═══════════════════════════════════════════════════════════════
    # Síntese final
    # ═══════════════════════════════════════════════════════════════
    out.append(f"\n{'='*74}")
    out.append("  SÍNTESE: FEATURES ROBUSTAS A MARKETING (por gênero × target)")
    out.append("=" * 74)

    if all_rows:
        full = pd.DataFrame(all_rows)
        # Critério de robustez publicável
        robust = full[
            (full["p_partial"] < 0.05) &
            (full["p_raw"] < 0.05) &
            (full["rho_partial"].abs() >= 0.15) &
            (full["rho_partial"].abs() >= 0.8 * full["rho_raw"].abs())
        ].sort_values("rho_partial", key=abs, ascending=False)

        if len(robust) == 0:
            out.append("\n  ⚠ Nenhuma feature atinge critério estrito")
            out.append("    (|ρ_parcial|≥0.15, p<0.05, retenção ≥80% da ρ bruta).")
            out.append("    Relaxando para |ρ_parcial|≥0.10 e p<0.10:\n")
            robust = full[
                (full["p_partial"] < 0.10) &
                (full["rho_partial"].abs() >= 0.10)
            ].sort_values("rho_partial", key=abs, ascending=False)

        if len(robust) > 0:
            out.append(f"\n  {'Gênero':<10s} {'Target':<15s} "
                       f"{'Feature':<32s} "
                       f"{'ρ_bru':>7s} {'ρ_par':>7s} {'p_par':>7s}")
            out.append(f"  {'-'*84}")
            for _, r in robust.head(30).iterrows():
                out.append(f"  {r['genre']:<10s} {r['target']:<15s} "
                           f"{r['feature']:<32s} "
                           f"{r['rho_raw']:>+7.3f} {r['rho_partial']:>+7.3f} "
                           f"{r['p_partial']:>7.4f}")
        else:
            out.append("\n  Nenhum efeito sobrevive com critério relaxado.")

        # Salvar tabela
        pd.DataFrame(all_rows).to_csv(
            RESULTS / "genre_budget_controlled.csv", index=False)

    # Salvar relatório
    txt = "\n".join(out)
    print(txt)
    (RESULTS / "genre_budget_controlled.txt").write_text(txt, encoding="utf-8")

    print(f"\n✓ Relatório: {RESULTS / 'genre_budget_controlled.txt'}")
    print(f"✓ Tabela:    {RESULTS / 'genre_budget_controlled.csv'}")


if __name__ == "__main__":
    main()
