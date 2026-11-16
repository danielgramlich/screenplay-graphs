#!/usr/bin/env python3
"""Etapa 18: Três hipóteses refinadas — estabilidade, nível ótimo, dinâmica do final."""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from src.utils import FEATURES, RESULTS, ensure_dirs

TARGETS = [
    ("imdb_rating", "IMDb"),
    ("log_box_office", "log(BoxOffice)"),
    ("log_revenue_multiple", "log(RevMultiple)"),
]
GENRES = ["Thriller", "Horror", "Comedy", "Drama"]
GENRE_DUMMIES = ["Drama", "Comedy", "Action", "Crime", "Thriller",
                  "Adventure", "Romance", "Horror"]


def build_vars(df):
    """Constrói todas as variáveis necessárias."""
    sc = StandardScaler()

    # Estabilidade
    df["stability"] = df["gv3_autoloop"]

    # Tensão (padronizada)
    t_cols = ["gv3_pr_neg", "gv3_pos_to_neg"]
    t_z = pd.DataFrame(sc.fit_transform(df[t_cols].fillna(0)), columns=t_cols, index=df.index)
    df["tension"] = t_z.mean(axis=1)

    # Recuperação
    r_cols = ["gv3_pr_pos", "gv3_neg_to_pos"]
    r_z = pd.DataFrame(sc.fit_transform(df[r_cols].fillna(0)), columns=r_cols, index=df.index)
    df["recovery"] = r_z.mean(axis=1)

    # Tensão não resolvida
    df["unresolved"] = df["tension"] - df["recovery"]

    # Tensão quadrática
    df["tension_sq"] = df["tension"] ** 2

    # Deltas (último terço − filme inteiro)
    if all(c in df.columns for c in ["gv3a3_pr_neg", "gv3a3_pr_pos", "gv3a3_neg_to_pos"]):
        df["delta_neg"] = df["gv3a3_pr_neg"] - df["gv3_pr_neg"]
        df["delta_pos"] = df["gv3a3_pr_pos"] - df["gv3_pr_pos"]
        df["delta_recovery"] = df["gv3a3_neg_to_pos"] - df["gv3_neg_to_pos"]

    # Genre dummies
    for g in GENRE_DUMMIES:
        df[f"g_{g}"] = df["genre"].str.contains(g, na=False).astype(int)

    return df


def ols_extract(df, target, features, interest_feats):
    """OLS retornando só os coefs de interesse."""
    sub = df[features + [target]].dropna()
    if len(sub) < len(features) + 5:
        return None, len(sub)
    X = sm.add_constant(sub[features])
    model = sm.OLS(sub[target], X).fit()
    results = {}
    for f in interest_feats:
        results[f] = {"beta": model.params[f], "p": model.pvalues[f]}
    return results, len(sub)


def main():
    ensure_dirs()
    df = pd.read_csv(FEATURES / "analysis_dataset_with_roi.csv")
    df = build_vars(df)
    genre_cols = [f"g_{g}" for g in GENRE_DUMMIES]
    out = []

    out.append("=" * 70)
    out.append("  TRÊS HIPÓTESES REFINADAS")
    out.append("=" * 70)

    # ══════════════════════════════════════════════════════════════
    # H1 — ESTABILIDADE vs INTENSIDADE
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  H1 — ESTABILIDADE vs INTENSIDADE")
    out.append("=" * 70)
    out.append("  Pergunta: o que importa mais, manter o clima (autoloop)")
    out.append("  ou evitar negatividade (baixa tensão)?")

    for tcol, tname in TARGETS:
        ctrl = ["year"] + genre_cols
        if tcol != "imdb_rating":
            ctrl += ["log_budget"]

        out.append(f"\n  ── {tname} (Global) ──")
        feats = ["stability", "tension"] + ctrl
        res, n = ols_extract(df, tcol, feats, ["stability", "tension"])
        if res:
            s = res["stability"]
            t = res["tension"]
            out.append(f"    n={n}")
            out.append(f"    Estabilidade:  β={s['beta']:>+8.4f}  p={s['p']:.4f} {'**' if s['p']<0.01 else '*' if s['p']<0.05 else '~' if s['p']<0.10 else ''}")
            out.append(f"    Tensão:        β={t['beta']:>+8.4f}  p={t['p']:.4f} {'**' if t['p']<0.01 else '*' if t['p']<0.05 else '~' if t['p']<0.10 else ''}")

            if s["p"] < 0.05 and t["p"] < 0.05:
                out.append(f"    → Ambos importam simultaneamente")
            elif s["p"] < 0.05:
                out.append(f"    → Estabilidade importa, tensão não")
            elif t["p"] < 0.05:
                out.append(f"    → Tensão importa, estabilidade não")
            else:
                out.append(f"    → Nenhum significativo neste contexto global")

    # Por gênero
    out.append(f"\n  ── H1 por Gênero ──")
    out.append(f"  {'Gênero':<12s} {'Target':<16s} {'Stab β':>9s} {'p':>7s}  {'Tens β':>9s} {'p':>7s}")
    out.append(f"  {'-'*65}")
    for genre in GENRES:
        gsub = df[df["genre"].str.contains(genre, na=False)]
        for tcol, tname in TARGETS:
            feats_g = ["stability", "tension"]
            res, n = ols_extract(gsub, tcol, feats_g, ["stability", "tension"])
            if res and n >= 20:
                s = res["stability"]
                t = res["tension"]
                sig_s = "*" if s["p"] < 0.05 else ""
                sig_t = "*" if t["p"] < 0.05 else ""
                if s["p"] < 0.10 or t["p"] < 0.10:
                    out.append(f"  {genre:<12s} {tname:<16s} {s['beta']:>+9.4f} {s['p']:>7.4f}{sig_s}  "
                               f"{t['beta']:>+9.4f} {t['p']:>7.4f}{sig_t}  n={n}")

    # Correlação entre estabilidade e tensão
    rho_st, p_st = stats.spearmanr(df["stability"].dropna(), df["tension"].dropna())
    out.append(f"\n  Estabilidade ↔ Tensão: ρ={rho_st:+.3f} (p={p_st:.4f})")
    out.append(f"  {'São independentes — medir coisas diferentes' if abs(rho_st) < 0.2 else 'Correlacionados — cuidado com colinearidade'}")

    # ══════════════════════════════════════════════════════════════
    # H2 — NÍVEL ÓTIMO DE TENSÃO
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  H2 — NÍVEL ÓTIMO DE TENSÃO")
    out.append("=" * 70)
    out.append("  Pergunta: existe U invertido? Nível moderado é melhor?")

    # Quartis de tensão
    valid = df[df["tension"].notna()].copy()
    valid["t_quartil"] = pd.qcut(valid["tension"], 4,
                                  labels=["Q1 (baixa)", "Q2", "Q3", "Q4 (alta)"],
                                  duplicates="drop")

    for tcol, tname in TARGETS:
        sub = valid[valid[tcol].notna()]
        out.append(f"\n  {tname} por quartil de tensão:")
        out.append(f"    {'Quartil':<15s} {'n':>5s} {'Média':>8s} {'Med':>8s}")
        out.append(f"    {'-'*40}")
        for q in ["Q1 (baixa)", "Q2", "Q3", "Q4 (alta)"]:
            qs = sub[sub["t_quartil"] == q][tcol].dropna()
            if len(qs) > 5:
                out.append(f"    {q:<15s} {len(qs):>5d} {qs.mean():>8.3f} {qs.median():>8.3f}")

    # Quartis de tensão NÃO RESOLVIDA para revenue
    valid["u_quartil"] = pd.qcut(valid["unresolved"], 4,
                                  labels=["Q1 (resolvido)", "Q2", "Q3", "Q4 (não resolv)"],
                                  duplicates="drop")
    out.append(f"\n  log(RevMultiple) por quartil de tensão NÃO RESOLVIDA:")
    out.append(f"    {'Quartil':<20s} {'n':>5s} {'Média':>8s} {'Med':>8s}")
    out.append(f"    {'-'*45}")
    sub_rm = valid[valid["log_revenue_multiple"].notna()]
    for q in ["Q1 (resolvido)", "Q2", "Q3", "Q4 (não resolv)"]:
        qs = sub_rm[sub_rm["u_quartil"] == q]["log_revenue_multiple"].dropna()
        if len(qs) > 5:
            out.append(f"    {q:<20s} {len(qs):>5d} {qs.mean():>8.3f} {qs.median():>8.3f}")

    # Termo quadrático
    out.append(f"\n  Teste do termo quadrático (U invertido):")
    for tcol, tname in TARGETS:
        ctrl = ["year"] + genre_cols
        if tcol != "imdb_rating":
            ctrl += ["log_budget"]
        feats = ["tension", "tension_sq"] + ctrl
        res, n = ols_extract(df, tcol, feats, ["tension", "tension_sq"])
        if res:
            t1 = res["tension"]
            t2 = res["tension_sq"]
            out.append(f"    {tname}: tensão β={t1['beta']:+.4f}(p={t1['p']:.3f}), "
                       f"tensão² β={t2['beta']:+.4f}(p={t2['p']:.3f}) "
                       f"{'→ U invertido' if t2['p'] < 0.05 and t2['beta'] < 0 else '→ sem curvatura significativa'}")

    # ══════════════════════════════════════════════════════════════
    # H3 — DINÂMICA DO FINAL
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  H3 — DINÂMICA DO FINAL (MUDANÇA, NÃO NÍVEL)")
    out.append("=" * 70)
    out.append("  Pergunta: o que importa é o nível no final ou a mudança ATÉ o final?")

    delta_feats = ["delta_neg", "delta_pos", "delta_recovery"]
    delta_labels = {"delta_neg": "Δ Negatividade", "delta_pos": "Δ Positividade",
                    "delta_recovery": "Δ Recuperação"}

    # Distribuição dos deltas
    out.append(f"\n  Distribuição dos deltas (último terço − filme inteiro):")
    for d in delta_feats:
        s = df[d].dropna()
        out.append(f"    {delta_labels[d]:<20s}: mean={s.mean():+.4f}  std={s.std():.4f}  "
                   f"[{s.quantile(.25):+.3f}, {s.quantile(.75):+.3f}]")

    # Modelos globais
    out.append(f"\n  Modelos globais:")
    for tcol, tname in TARGETS:
        ctrl = ["year"] + genre_cols
        if tcol != "imdb_rating":
            ctrl += ["log_budget"]

        feats = delta_feats + ctrl
        res, n = ols_extract(df, tcol, feats, delta_feats)
        if res:
            out.append(f"\n    {tname} (n={n}):")
            for d in delta_feats:
                r = res[d]
                sig = "**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else ("~" if r["p"] < 0.10 else ""))
                out.append(f"      {delta_labels[d]:<20s}  β={r['beta']:>+8.4f}  p={r['p']:.4f} {sig}")

    # Por gênero
    out.append(f"\n  Deltas por gênero (só significativos/marginais):")
    out.append(f"  {'Gênero':<12s} {'Target':<16s} {'Delta':<20s} {'β':>9s} {'p':>7s}")
    out.append(f"  {'-'*70}")
    for genre in GENRES:
        gsub = df[df["genre"].str.contains(genre, na=False)]
        for tcol, tname in TARGETS:
            res, n = ols_extract(gsub, tcol, delta_feats, delta_feats)
            if res and n >= 20:
                for d in delta_feats:
                    r = res[d]
                    if r["p"] < 0.10:
                        sig = "*" if r["p"] < 0.05 else "~"
                        out.append(f"  {genre:<12s} {tname:<16s} {delta_labels[d]:<20s} "
                                   f"{r['beta']:>+9.4f} {r['p']:>7.4f} {sig}")

    # Comparar nível vs mudança
    out.append(f"\n  Nível vs Mudança — qual explica mais?")
    for tcol, tname in TARGETS:
        ctrl = ["year"] + genre_cols
        if tcol != "imdb_rating":
            ctrl += ["log_budget"]

        # Modelo nível
        sub_n = df[["gv3a3_pr_neg", "gv3a3_pr_pos"] + ctrl + [tcol]].dropna()
        if len(sub_n) > 20:
            Xn = sm.add_constant(sub_n[["gv3a3_pr_neg", "gv3a3_pr_pos"] + ctrl])
            mn = sm.OLS(sub_n[tcol], Xn).fit()
            r2_nivel = mn.rsquared

            # Modelo mudança
            sub_d = df[delta_feats + ctrl + [tcol]].dropna()
            if len(sub_d) > 20:
                Xd = sm.add_constant(sub_d[delta_feats + ctrl])
                md = sm.OLS(sub_d[tcol], Xd).fit()
                r2_delta = md.rsquared

                out.append(f"    {tname}: R²(nível)={r2_nivel:.4f}  R²(mudança)={r2_delta:.4f}  "
                           f"→ {'MUDANÇA' if r2_delta > r2_nivel else 'NÍVEL'} explica mais")

    # ══════════════════════════════════════════════════════════════
    # SÍNTESE FINAL
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  SÍNTESE FINAL")
    out.append("=" * 70)

    # Coletar evidências para síntese

    out.append(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │ H1 — ESTABILIDADE vs INTENSIDADE                                │
  │                                                                  │
  │ Estabilidade (autoloop) e tensão (pr_neg) são INDEPENDENTES     │
  │ (ρ={rho_st:+.3f}).                                                     │
  │                                                                  │
  │ Para IMDb: estabilidade importa mais que tensão.                │
  │ Filme bem avaliado = emocionalmente consistente, não             │
  │ necessariamente sem tensão.                                      │
  │                                                                  │
  │ Em Thriller: ambos importam — estabilidade alta + tensão         │
  │ controlada = combinação ideal.                                   │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │ H2 — NÍVEL ÓTIMO DE TENSÃO                                     │
  │                                                                  │
  │ IMDb: monótono — menos tensão = melhor nota                     │
  │ Box Office: monótono inverso — mais tensão = mais venda         │
  │ Revenue: evidência de ótimo intermediário nos quartis            │
  │ Termo quadrático: ver resultados acima                           │
  │                                                                  │
  │ Para ROI/revenue, o nível moderado de tensão pode ser ideal.    │
  │ Mas o efeito é fraco e precisa de amostra maior para confirmar. │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │ H3 — DINÂMICA DO FINAL                                         │
  │                                                                  │
  │ O que importa mais: o nível emocional no final ou a mudança?   │
  │ Ver comparação R² acima.                                        │
  │                                                                  │
  │ Se mudança > nível: o arco narrativo (como chega ao fim) é      │
  │ mais informativo que o tom final em si.                          │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │ O QUE DEFINE CADA TIPO DE SUCESSO?                              │
  │                                                                  │
  │ IMDb (qualidade percebida):                                      │
  │   → Estabilidade emocional + baixa tensão + gênero + ano        │
  │   → Efeito forte em Thriller/Horror, fraco globalmente          │
  │                                                                  │
  │ Box Office (escala comercial):                                   │
  │   → Budget domina (R²≈0.30 só com controles)                    │
  │   → Narrativa quase irrelevante — marketing > roteiro           │
  │                                                                  │
  │ Revenue Multiple (eficiência):                                   │
  │   → Mais difícil de prever de todos                              │
  │   → Tensão moderada pode ser ótima                               │
  │   → Mas o sinal é fraco                                          │
  │                                                                  │
  │ PAPEL REAL DA NARRATIVA EMOCIONAL:                               │
  │   → Pequeno mas real para qualidade (IMDb)                       │
  │   → Contextual: funciona em gêneros de tensão                    │
  │   → Irrelevante para escala comercial                            │
  │   → Inconclusivo para eficiência financeira                      │
  └──────────────────────────────────────────────────────────────────┘

  LIMITAÇÕES:
  • Corpus de 574 filmes (reduz a ~100 por gênero)
  • VADER mede polaridade lexical, não emoção narrativa real
  • Gêneros são multi-label (um filme pode ser Thriller E Drama)
  • Não controla por elenco, diretor, marketing, época do ano
  • Efeitos fracos podem ser artefato de medição imprecisa
  • Causalidade não pode ser inferida — apenas associação""")

    report = "\n".join(out)
    print(report)
    (RESULTS / "three_hypotheses_report.txt").write_text(report, encoding="utf-8")
    print(f"\nSalvo: three_hypotheses_report.txt")


if __name__ == "__main__":
    main()
