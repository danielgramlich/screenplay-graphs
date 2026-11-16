#!/usr/bin/env python3
"""Etapa 37: Validação em hold-out temporal.

Responde à crítica de que H1-H4 foram formuladas após análise exploratória sobre
todos os filmes. Divide a subamostra com orçamento em:

  * Train   : filmes com year <= 2005  (n=316)
  * Hold-out: filmes com year >  2005  (n=146)

Re-roda o procedimento de correlação parcial da Etapa 36 separadamente nos dois
conjuntos. Compara magnitudes e direções, e emite "veredicto de replicação" por
feature robusta identificada na análise principal.

Critérios de replicação (pre-especificados):
  * Mesma direção do coeficiente (sinal).
  * |rho| no hold-out >= 50% do |rho| no train (tolerância a perda por amostra menor).
  * Feature robusta no train permanece com p < 0.10 no hold-out (limite relaxado
    dada a menor potência estatística do hold-out).
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

from src.utils import FEATURES, RESULTS, ensure_dirs

TARGETS = [("imdb_rating", "IMDb"), ("log_box_office", "log(BoxOffice)")]
GENRES_FOCUS = ["Thriller", "Horror", "Comedy", "Action", "Drama", "Crime"]
YEAR_CUT = 2005
SEED = 42

ROB_GRAPH = ["rob_gv5_autoloop", "rob_gv3_pr_neg", "rob_gv5_pr_v5_pos",
             "rob_gv3_entropy"]
ROB_PROTAG = ["rob_protag_mean", "rob_protag_pct_positive",
              "rob_protag_pct_negative", "rob_protag_volatility",
              "rob_protag_arc", "rob_protag_coverage"]
CHAR_NET = ["F1_density", "F1_strength_centralization", "F1_max_betweenness",
            "F1_avg_clustering", "F1_modularity", "F2_W_entropy",
            "F3_protag_strength_frac", "F3_protag_presence"]

# Features robustas identificadas na análise principal (Etapa 36)
# que deverão ser testadas formalmente no hold-out.
FEATURES_TO_VALIDATE = [
    ("Horror",   "IMDb",           "F1_strength_centralization"),
    ("Horror",   "IMDb",           "F1_avg_clustering"),
    ("Horror",   "IMDb",           "F3_protag_strength_frac"),
    ("Thriller", "IMDb",           "rob_gv3_pr_neg"),
    ("Thriller", "IMDb",           "rob_gv5_autoloop"),
    ("Thriller", "IMDb",           "rob_gv3_entropy"),
    ("Thriller", "IMDb",           "F1_density"),
    ("Comedy",   "log(BoxOffice)", "rob_protag_mean"),
    ("Comedy",   "log(BoxOffice)", "rob_protag_pct_positive"),
    ("Comedy",   "log(BoxOffice)", "rob_protag_pct_negative"),
    ("Action",   "IMDb",           "F3_protag_strength_frac"),
    ("Action",   "IMDb",           "rob_gv3_pr_neg"),
    ("Action",   "IMDb",           "rob_protag_mean"),
]


def partial_spearman(df, feat, target, controls):
    """Correlação parcial de Spearman via resíduos Ridge."""
    sub = df.dropna(subset=[feat, target] + controls).copy()
    if len(sub) < 10:
        return None
    Xc = StandardScaler().fit_transform(sub[controls].values)
    y = sub[target].values
    x = sub[feat].values
    resy = y - Ridge(alpha=1.0, random_state=SEED).fit(Xc, y).predict(Xc)
    resx = x - Ridge(alpha=1.0, random_state=SEED).fit(Xc, x).predict(Xc)
    r = stats.spearmanr(resx, resy)
    return {"rho": r.statistic, "p": r.pvalue, "n": len(sub)}


def verdict(rho_train, rho_ho, p_ho):
    """Emite veredicto de replicação segundo critérios pré-especificados."""
    if rho_ho is None or np.isnan(rho_ho):
        return "n insuf"
    # Mesma direção
    if np.sign(rho_train) != np.sign(rho_ho):
        return "✗ inverte"
    # Retenção de magnitude
    if abs(rho_ho) < 0.5 * abs(rho_train):
        return "~ atenua"
    # Significância relaxada no hold-out
    if p_ho < 0.05:
        return "✓ replica"
    if p_ho < 0.10:
        return "✓ replica (p<0,10)"
    return "? nao-sig"


def main():
    ensure_dirs()
    out = []

    # ── Dados ──
    final = pd.read_csv(FEATURES / "final_combined_dataset.csv")
    roi = pd.read_csv(FEATURES / "analysis_dataset_with_roi.csv")
    df = final.merge(roi[["title", "budget", "log_budget"]],
                     on="title", how="left")
    df = df[df["log_budget"].notna() & df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    train = df[df["year"] <= YEAR_CUT].copy()
    ho = df[df["year"] > YEAR_CUT].copy()

    out.append("=" * 74)
    out.append("  VALIDAÇÃO EM HOLD-OUT TEMPORAL (pipeline/37)")
    out.append("=" * 74)
    out.append(f"\n  Critério de divisão: ano de lançamento <= {YEAR_CUT}")
    out.append(f"  Train   (ano <= {YEAR_CUT}): n = {len(train)} filmes")
    out.append(f"  Hold-out (ano >  {YEAR_CUT}): n = {len(ho)} filmes")
    out.append("\n  Controles em cada conjunto: year + log_budget")
    out.append("  Método: correlação parcial de Spearman dentro de cada gênero.")

    # Tamanhos por gênero
    out.append("\n  Tamanhos amostrais por gênero:")
    out.append(f"    {'Gênero':<12s} {'Train':>7s} {'Hold-out':>10s}")
    out.append(f"    {'-'*31}")
    for g in GENRES_FOCUS:
        col = f"g_{g}"
        if col in df.columns:
            nt = int(train[col].sum())
            nh = int(ho[col].sum())
            out.append(f"    {g:<12s} {nt:>7d} {nh:>10d}")

    # ═══════════════════════════════════════════════════════════════
    # Teste formal de replicação para cada feature robusta
    # ═══════════════════════════════════════════════════════════════
    out.append(f"\n{'='*74}")
    out.append("  TESTE DE REPLICAÇÃO: FEATURES ROBUSTAS DA ANÁLISE PRINCIPAL")
    out.append("=" * 74)
    out.append("\n  Critérios de replicação (pre-especificados):")
    out.append("    1. Mesmo sinal de rho.")
    out.append("    2. |rho_hold-out| >= 50% de |rho_train|.")
    out.append("    3. p < 0,10 no hold-out (relaxamento por menor potência).")

    rows = []
    for genre, target_name, feat in FEATURES_TO_VALIDATE:
        tcol = "imdb_rating" if target_name == "IMDb" else "log_box_office"
        col = f"g_{genre}"
        if col not in df.columns:
            continue

        gdf_tr = train[train[col] == 1]
        gdf_ho = ho[ho[col] == 1]

        r_tr = partial_spearman(gdf_tr, feat, tcol,
                                 controls=["year", "log_budget"])
        r_ho = partial_spearman(gdf_ho, feat, tcol,
                                 controls=["year", "log_budget"])

        if r_tr is None:
            continue

        verdict_str = verdict(
            r_tr["rho"],
            r_ho["rho"] if r_ho else None,
            r_ho["p"] if r_ho else 1.0,
        )

        rows.append({
            "genre": genre, "target": target_name, "feature": feat,
            "n_train": r_tr["n"],
            "rho_train": round(r_tr["rho"], 3),
            "p_train": round(r_tr["p"], 4),
            "n_ho": r_ho["n"] if r_ho else 0,
            "rho_ho": round(r_ho["rho"], 3) if r_ho else None,
            "p_ho": round(r_ho["p"], 4) if r_ho else None,
            "verdict": verdict_str,
        })

    out.append(f"\n  {'Gênero':<10s} {'Alvo':<14s} {'Feature':<32s} "
               f"{'ρ_train':>8s} {'ρ_HO':>8s} {'p_HO':>7s} {'Veredicto':<20s}")
    out.append("  " + "-" * 88)
    for r in rows:
        rho_ho_str = f"{r['rho_ho']:+.3f}" if r['rho_ho'] is not None else "  n.d."
        p_ho_str = f"{r['p_ho']:.3f}" if r['p_ho'] is not None else "  n.d."
        out.append(
            f"  {r['genre']:<10s} {r['target']:<14s} {r['feature']:<32s} "
            f"{r['rho_train']:>+8.3f} {rho_ho_str:>8s} {p_ho_str:>7s} {r['verdict']:<20s}"
        )

    # ─── Síntese ───
    n_total = len(rows)
    n_rep = sum(1 for r in rows if "replica" in r["verdict"])
    n_att = sum(1 for r in rows if "atenua" in r["verdict"])
    n_inv = sum(1 for r in rows if "inverte" in r["verdict"])
    n_ns  = sum(1 for r in rows if "nao-sig" in r["verdict"])
    n_in  = sum(1 for r in rows if "insuf"  in r["verdict"])

    out.append(f"\n  Síntese: {n_rep}/{n_total} replicam, "
               f"{n_att} atenuam, {n_inv} invertem, {n_ns} perdem significância, "
               f"{n_in} amostra insuficiente.")

    # ═══════════════════════════════════════════════════════════════
    # Tabela final para o artigo
    # ═══════════════════════════════════════════════════════════════
    pd.DataFrame(rows).to_csv(RESULTS / "temporal_holdout.csv", index=False)

    txt = "\n".join(out)
    print(txt)
    (RESULTS / "temporal_holdout.txt").write_text(txt, encoding="utf-8")
    print(f"\n✓ Relatório: {RESULTS / 'temporal_holdout.txt'}")
    print(f"✓ Tabela:    {RESULTS / 'temporal_holdout.csv'}")


if __name__ == "__main__":
    main()
