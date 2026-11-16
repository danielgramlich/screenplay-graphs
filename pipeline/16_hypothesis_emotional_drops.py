#!/usr/bin/env python3
"""Etapa 16: Teste de hipótese — quedas emocionais bruscas e sucesso."""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from src.utils import FEATURES, RESULTS, ensure_dirs

FEATURE = "gv3_pos_to_neg"
FEATURE_LABEL = "Queda Emocional (Pos→Neg)"
FEATURE_RECOVERY = "gv3_neg_to_pos"
RECOVERY_LABEL = "Recuperação (Neg→Pos)"

TARGETS = [
    ("imdb_rating", "IMDb"),
    ("log_box_office", "log(BoxOffice)"),
    ("log_revenue_multiple", "log(RevMultiple)"),
]
GENRES = ["Thriller", "Horror", "Comedy", "Drama"]
GENRE_DUMMIES = ["Drama", "Comedy", "Action", "Crime", "Thriller",
                  "Adventure", "Romance", "Horror"]


def main():
    ensure_dirs()
    df = pd.read_csv(FEATURES / "analysis_dataset_with_roi.csv")
    for g in GENRE_DUMMIES:
        df[f"g_{g}"] = df["genre"].str.contains(g, na=False).astype(int)

    out = []
    out.append("=" * 70)
    out.append("  HIPÓTESE: QUEDAS EMOCIONAIS BRUSCAS E SUCESSO")
    out.append("=" * 70)
    out.append(f"\n  Feature: {FEATURE} = proporção de transições Positivo → Negativo")
    out.append(f"  Complemento: {FEATURE_RECOVERY} = proporção de transições Negativo → Positivo")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 1 — EXPLORAÇÃO
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  ETAPA 1 — DISTRIBUIÇÃO E CORRELAÇÕES")
    out.append("=" * 70)

    s = df[FEATURE].dropna()
    out.append(f"\n  {FEATURE_LABEL} (n={len(s)}):")
    out.append(f"    mean={s.mean():.4f}  std={s.std():.4f}  "
               f"min={s.min():.4f}  median={s.median():.4f}  max={s.max():.4f}")
    out.append(f"    skew={s.skew():.2f}  — {'simétrico' if abs(s.skew()) < 0.5 else 'assimétrico'}")

    sr = df[FEATURE_RECOVERY].dropna()
    out.append(f"\n  {RECOVERY_LABEL} (n={len(sr)}):")
    out.append(f"    mean={sr.mean():.4f}  std={sr.std():.4f}  median={sr.median():.4f}")

    # Correlação queda vs recuperação
    both = df[[FEATURE, FEATURE_RECOVERY]].dropna()
    rho_qr, p_qr = stats.spearmanr(both[FEATURE], both[FEATURE_RECOVERY])
    out.append(f"\n  Correlação queda ↔ recuperação: ρ={rho_qr:+.3f} (p={p_qr:.4f})")
    out.append(f"    {'Fortemente correlacionadas — filmes que caem também sobem' if rho_qr > 0.5 else 'Moderadamente correlacionadas' if rho_qr > 0.3 else 'Fracamente correlacionadas'}")

    out.append(f"\n  Correlações com targets:")
    out.append(f"    {'Target':<18s} {'Queda ρ':>10s} {'p':>8s}  {'Recup ρ':>10s} {'p':>8s}")
    out.append(f"    {'-'*58}")
    for tcol, tname in TARGETS:
        v = df[[FEATURE, FEATURE_RECOVERY, tcol]].dropna()
        if len(v) < 30:
            continue
        rho1, p1 = stats.spearmanr(v[FEATURE], v[tcol])
        rho2, p2 = stats.spearmanr(v[FEATURE_RECOVERY], v[tcol])
        sig1 = "*" if p1 < 0.05 else ""
        sig2 = "*" if p2 < 0.05 else ""
        out.append(f"    {tname:<18s} {rho1:>+10.3f} {p1:>8.4f}{sig1:>2s}  {rho2:>+10.3f} {p2:>8.4f}{sig2:>2s}")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 2 — GRUPOS
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  ETAPA 2 — ANÁLISE POR GRUPOS")
    out.append("=" * 70)

    valid = df[df[FEATURE].notna()].copy()
    q33, q66 = valid[FEATURE].quantile([0.33, 0.66])
    valid["drop_group"] = pd.cut(valid[FEATURE], bins=[-1, q33, q66, 1],
                                  labels=["Poucas quedas", "Moderado", "Muitas quedas"])

    out.append(f"\n  Limites: Poucas < {q33:.4f} | Moderado < {q66:.4f} | Muitas ≥ {q66:.4f}")
    out.append(f"\n  {'Grupo':<18s} {'n':>5s} {'IMDb':>8s} {'logBO':>8s} {'logRM':>8s}")
    out.append(f"  {'-'*50}")

    for group in ["Poucas quedas", "Moderado", "Muitas quedas"]:
        g = valid[valid["drop_group"] == group]
        imdb = g["imdb_rating"].mean()
        bo = g["log_box_office"].mean()
        rm = g["log_revenue_multiple"].mean()
        out.append(f"  {group:<18s} {len(g):>5d} {imdb:>8.2f} {bo:>8.2f} {rm:>8.2f}")

    # Teste entre extremos
    low = valid[valid["drop_group"] == "Poucas quedas"]
    high = valid[valid["drop_group"] == "Muitas quedas"]
    for tcol, tname in TARGETS:
        l = low[tcol].dropna()
        h = high[tcol].dropna()
        if len(l) > 10 and len(h) > 10:
            u, p = stats.mannwhitneyu(l, h, alternative="two-sided")
            diff = h.mean() - l.mean()
            out.append(f"\n    {tname}: Muitas vs Poucas → Δ={diff:+.3f}, p={p:.4f} "
                       f"{'*' if p < 0.05 else ''}")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 3 — MODELO CONTROLADO
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  ETAPA 3 — MODELO CONTROLADO (OLS)")
    out.append("=" * 70)

    genre_cols = [f"g_{g}" for g in GENRE_DUMMIES]

    for tcol, tname in TARGETS:
        if tcol in ("log_box_office", "log_revenue_multiple"):
            controls = ["year", "log_budget"] + genre_cols
        else:
            controls = ["year"] + genre_cols

        features = [FEATURE, FEATURE_RECOVERY] + controls
        sub = df[features + [tcol]].dropna()
        if len(sub) < 30:
            continue

        X = sm.add_constant(sub[features])
        y = sub[tcol]
        model = sm.OLS(y, X).fit()

        out.append(f"\n  {tname} (n={len(sub)}, R²={model.rsquared:.4f}):")
        for feat in [FEATURE, FEATURE_RECOVERY]:
            coef = model.params[feat]
            p = model.pvalues[feat]
            label = FEATURE_LABEL if feat == FEATURE else RECOVERY_LABEL
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            out.append(f"    {label:<30s}  β={coef:>+8.4f}  p={p:.4f} {sig}")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 4 — POR GÊNERO
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  ETAPA 4 — CORRELAÇÕES POR GÊNERO")
    out.append("=" * 70)

    out.append(f"\n  {'Gênero':<12s} {'Target':<18s} {'Queda ρ':>10s} {'p':>8s}  {'Recup ρ':>10s} {'p':>8s}  {'n':>5s}")
    out.append(f"  {'-'*70}")

    for genre in GENRES:
        gsub = df[df["genre"].str.contains(genre, na=False)]
        for tcol, tname in TARGETS:
            v = gsub[[FEATURE, FEATURE_RECOVERY, tcol]].dropna()
            if len(v) < 20:
                continue
            rho1, p1 = stats.spearmanr(v[FEATURE], v[tcol])
            rho2, p2 = stats.spearmanr(v[FEATURE_RECOVERY], v[tcol])
            sig1 = "*" if p1 < 0.05 else ""
            sig2 = "*" if p2 < 0.05 else ""
            if p1 < 0.10 or p2 < 0.10:
                out.append(f"  {genre:<12s} {tname:<18s} {rho1:>+10.3f} {p1:>8.4f}{sig1:>2s}  "
                           f"{rho2:>+10.3f} {p2:>8.4f}{sig2:>2s}  {len(v):>5d}")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 5 — NÃO LINEARIDADE
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  ETAPA 5 — TESTE DE NÃO LINEARIDADE (QUARTIS)")
    out.append("=" * 70)

    valid2 = df[df[FEATURE].notna()].copy()
    valid2["quartil"] = pd.qcut(valid2[FEATURE], 4, labels=["Q1 (menos)", "Q2", "Q3", "Q4 (mais)"],
                                 duplicates="drop")

    for tcol, tname in TARGETS:
        sub = valid2[valid2[tcol].notna()]
        out.append(f"\n  {tname}:")
        out.append(f"    {'Quartil':<15s} {'n':>5s} {'Média':>8s} {'Mediana':>8s}")
        out.append(f"    {'-'*40}")

        quartil_means = []
        for q in ["Q1 (menos)", "Q2", "Q3", "Q4 (mais)"]:
            qs = sub[sub["quartil"] == q][tcol].dropna()
            if len(qs) > 5:
                out.append(f"    {q:<15s} {len(qs):>5d} {qs.mean():>8.3f} {qs.median():>8.3f}")
                quartil_means.append(qs.mean())

        if len(quartil_means) == 4:
            # Padrão: monótono ou U/invertido?
            diffs = np.diff(quartil_means)
            if all(d > 0 for d in diffs):
                pattern = "monótono crescente (mais queda → melhor)"
            elif all(d < 0 for d in diffs):
                pattern = "monótono decrescente (mais queda → pior)"
            elif quartil_means[1] > quartil_means[0] and quartil_means[1] > quartil_means[3]:
                pattern = "U invertido (nível moderado é melhor)"
            elif quartil_means[1] < quartil_means[0] and quartil_means[1] < quartil_means[3]:
                pattern = "U (extremos são melhores)"
            else:
                pattern = "sem padrão claro"
            out.append(f"    → Padrão: {pattern}")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 6 — INTERPRETAÇÃO
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  ETAPA 6 — INTERPRETAÇÃO")
    out.append("=" * 70)

    # Calcular dados para interpretação
    global_corrs = {}
    for tcol, tname in TARGETS:
        v = df[[FEATURE, tcol]].dropna()
        if len(v) > 30:
            rho, p = stats.spearmanr(v[FEATURE], v[tcol])
            global_corrs[tname] = (rho, p)

    out.append(f"""
  1. QUEDAS BRUSCAS AJUDAM OU PREJUDICAM?

     Globalmente:
     - IMDb: ρ = {global_corrs.get('IMDb', (0,1))[0]:+.3f} → {'SEM efeito significativo' if global_corrs.get('IMDb', (0,1))[1] > 0.05 else 'efeito significativo'}
     - Box Office: ρ = {global_corrs.get('log(BoxOffice)', (0,1))[0]:+.3f} → {'positivo: quedas vendem mais' if global_corrs.get('log(BoxOffice)', (0,1))[0] > 0 else 'negativo'}
     - Revenue: ρ = {global_corrs.get('log(RevMultiple)', (0,1))[0]:+.3f}

  2. DEPENDE DO TIPO DE SUCESSO?
     Se confirmado pelos dados: IMDb e box office podem reagir
     diferentemente a quedas bruscas. O público geral pode gostar
     de montanhas-russas emocionais (mais bilheteira), mas os
     avaliadores do IMDb podem preferir controle narrativo.

  3. DEPENDE DO GÊNERO?
     Ver resultados da Etapa 4 acima. Thriller e Action tendem
     a ter padrões mais claros porque a queda emocional é central
     à estrutura narrativa desses gêneros.

  4. EXISTE NÍVEL ÓTIMO?
     Ver análise de quartis (Etapa 5). Se houver U invertido,
     sugere que um nível moderado de tensão é ideal.""")

    report = "\n".join(out)
    print(report)
    (RESULTS / "hypothesis_emotional_drops.txt").write_text(report, encoding="utf-8")
    print(f"\nSalvo: hypothesis_emotional_drops.txt")


if __name__ == "__main__":
    main()
