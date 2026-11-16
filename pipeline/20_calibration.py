#!/usr/bin/env python3
"""Etapa 20: Calibração diagnóstica + recalibração + validação."""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from src.utils import FEATURES, RESULTS, ensure_dirs

out = []


def section(title):
    out.append(f"\n{'='*70}")
    out.append(f"  {title}")
    out.append("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# RODADA 1 — DIAGNÓSTICO
# ═══════════════════════════════════════════════════════════════════

def diagnose(df):
    section("RODADA 1 — DIAGNÓSTICO DOS PERFIS E ÍNDICES")

    # Distribuição dos índices
    out.append(f"\n  Percentis dos índices:")
    out.append(f"  {'Índice':<18s} {'P10':>7s} {'P25':>7s} {'P50':>7s} {'P75':>7s} {'P90':>7s}")
    out.append(f"  {'-'*55}")
    for idx in ["stability", "tension", "recovery", "unresolved", "ending_shift"]:
        s = df[idx].dropna()
        ps = [s.quantile(p) for p in [0.10, 0.25, 0.50, 0.75, 0.90]]
        out.append(f"  {idx:<18s} {ps[0]:>7.3f} {ps[1]:>7.3f} {ps[2]:>7.3f} {ps[3]:>7.3f} {ps[4]:>7.3f}")

    # Perfis vs IMDb real
    out.append(f"\n  Perfis vs IMDb real:")
    out.append(f"  {'Perfil':<22s} {'n':>5s} {'IMDb':>7s} {'IMDb_est':>8s} {'MAE':>6s} {'Cenas':>7s}")
    out.append(f"  {'-'*60}")

    for profile in df["profile"].value_counts().index:
        p = df[df["profile"] == profile]
        imdb = p["imdb_actual"].dropna()
        est = p.loc[imdb.index, "imdb_estimate"]
        mae = (est - imdb).abs().mean() if len(imdb) > 0 else np.nan
        out.append(f"  {profile:<22s} {len(p):>5d} {imdb.mean():>7.2f} {est.mean():>8.2f} "
                   f"{mae:>6.2f} {p.get('num_scenes', pd.Series()).mean():>7.0f}"
                   if len(imdb) > 0 else f"  {profile:<22s} {len(p):>5d}   —")

    # ROI diagnóstico
    out.append(f"\n  ROI tendency vs revenue_multiple real:")
    if "revenue_multiple_actual" in df.columns:
        rm = df[["roi_tendency", "revenue_multiple_actual"]].dropna(subset=["revenue_multiple_actual"])
        for t in ["desfavorável", "neutro", "favorável"]:
            sub = rm[rm["roi_tendency"] == t]["revenue_multiple_actual"]
            if len(sub) > 0:
                out.append(f"    {t:<15s}  n={len(sub):>4d}  median={sub.median():>7.2f}x  "
                           f"mean={sub.mean():>7.2f}x")
    else:
        out.append("    (coluna revenue_multiple_actual não encontrada)")

    # Onde o IMDb falha mais
    has = df[df["imdb_actual"].notna()].copy()
    has["error"] = has["imdb_estimate"] - has["imdb_actual"]
    has["abs_error"] = has["error"].abs()

    out.append(f"\n  Onde o erro é maior:")
    out.append(f"    Filmes com IMDb real < 5.0:")
    low = has[has["imdb_actual"] < 5.0]
    if len(low) > 0:
        out.append(f"      n={len(low)}, MAE={low['abs_error'].mean():.2f}, "
                   f"viés médio={low['error'].mean():+.2f} (sobreestima)")
        out.append(f"      Índices médios: stab={low['stability'].mean():.3f} "
                   f"tens={low['tension'].mean():.3f} shift={low['ending_shift'].mean():.3f}")

    out.append(f"\n    Filmes com IMDb real > 8.0:")
    high = has[has["imdb_actual"] > 8.0]
    if len(high) > 0:
        out.append(f"      n={len(high)}, MAE={high['abs_error'].mean():.2f}, "
                   f"viés médio={high['error'].mean():+.2f}")
        out.append(f"      Índices médios: stab={high['stability'].mean():.3f} "
                   f"tens={high['tension'].mean():.3f} shift={high['ending_shift'].mean():.3f}")

    return has


# ═══════════════════════════════════════════════════════════════════
# RODADA 2 — RECALIBRAÇÃO
# ═══════════════════════════════════════════════════════════════════

def recalibrate(df):
    """Recalibra scoring.py com base nos percentis observados."""
    section("RODADA 2 — RECALIBRAÇÃO")

    # Novos limiares baseados em percentis
    out.append(f"\n  Recalibração dos perfis:")

    stab_p25, stab_p50, stab_p75 = df["stability"].quantile([0.25, 0.50, 0.75])
    tens_p25, tens_p50, tens_p75 = df["tension"].quantile([0.25, 0.50, 0.75])
    rec_p25, rec_p50, rec_p75 = df["recovery"].quantile([0.25, 0.50, 0.75])
    shift_p50, shift_p75, shift_p90 = df["ending_shift"].quantile([0.50, 0.75, 0.90])
    unres_p25, unres_p75 = df["unresolved"].quantile([0.25, 0.75])

    out.append(f"    stability: P25={stab_p25:.3f} P50={stab_p50:.3f} P75={stab_p75:.3f}")
    out.append(f"    tension:   P25={tens_p25:.3f} P50={tens_p50:.3f} P75={tens_p75:.3f}")
    out.append(f"    recovery:  P25={rec_p25:.3f} P50={rec_p50:.3f} P75={rec_p75:.3f}")
    out.append(f"    ending_shift: P50={shift_p50:.3f} P75={shift_p75:.3f} P90={shift_p90:.3f}")
    out.append(f"    unresolved:   P25={unres_p25:.3f} P75={unres_p75:.3f}")

    # Reclassificar com novos limiares
    def reclassify(row):
        s = row["stability"]
        t = row["tension"]
        r = row["recovery"]
        u = row["unresolved"]
        e = row["ending_shift"]

        # Ordem hierárquica: específicos primeiro, default por último
        # 1. Consistente: top quartil de estabilidade + baixo unresolved
        if s > stab_p75 and abs(u) < 0.15:
            return "Consistente"

        # 2. Prestígio: estabilidade acima da mediana + tensão moderada + final estável
        if s > stab_p50 and tens_p25 < t < tens_p75 and e < shift_p75:
            return "Prestígio"

        # 3. Sombrio controlado: tensão alta + estabilidade acima mediana + pouca recuperação
        if t > tens_p75 and s > stab_p50 and r < rec_p50:
            return "Sombrio controlado"

        # 4. Catártico: tensão alta + recuperação alta
        if t > tens_p50 and r > rec_p75:
            return "Catártico"

        # 5. Comercial: tensão alta + estabilidade baixa (oscilação)
        if t > tens_p50 and s < stab_p25:
            return "Comercial"

        # 6. Intenso: tensão muito alta + recuperação baixa
        if t > tens_p75 and r < rec_p25:
            return "Intenso"

        # 7. Desequilibrado: SÓ quando ending_shift acima P90 E estabilidade abaixo P25
        if e > shift_p90 and s < stab_p25:
            return "Desequilibrado"

        return "Neutro"

    df["profile_v2"] = df.apply(reclassify, axis=1)

    out.append(f"\n  Novos perfis (v2):")
    out.append(f"  {'Perfil':<22s} {'v1':>5s} → {'v2':>5s}")
    out.append(f"  {'-'*40}")
    v1_counts = df["profile"].value_counts()
    v2_counts = df["profile_v2"].value_counts()
    all_profiles = sorted(set(v1_counts.index) | set(v2_counts.index))
    for p in all_profiles:
        n1 = v1_counts.get(p, 0)
        n2 = v2_counts.get(p, 0)
        out.append(f"  {p:<22s} {n1:>5d} → {n2:>5d}")

    # Verificar se v2 separa melhor IMDb
    out.append(f"\n  v2 perfis vs IMDb real:")
    out.append(f"  {'Perfil':<22s} {'n':>5s} {'IMDb':>7s}")
    out.append(f"  {'-'*38}")
    for p in v2_counts.index:
        sub = df[df["profile_v2"] == p]
        imdb = sub["imdb_actual"].dropna()
        if len(imdb) > 0:
            out.append(f"  {p:<22s} {len(sub):>5d} {imdb.mean():>7.2f}")

    # ── Recalibrar ROI ──
    out.append(f"\n  Recalibração do ROI:")

    # Usar percentis do roi_score para definir faixas empíricas
    roi_p33 = df["roi_score"].quantile(0.33)
    roi_p66 = df["roi_score"].quantile(0.66)
    out.append(f"    roi_score: P33={roi_p33:.4f} P66={roi_p66:.4f}")

    def recalibrate_roi(score):
        if score > roi_p66:
            return "favorável"
        if score > roi_p33:
            return "neutro"
        return "desfavorável"

    df["roi_tendency_v2"] = df["roi_score"].apply(recalibrate_roi)

    out.append(f"\n    ROI tendency v2:")
    for t in ["favorável", "neutro", "desfavorável"]:
        n = (df["roi_tendency_v2"] == t).sum()
        out.append(f"      {t:<15s}  {n:>4d}")

    # ── Recalibrar IMDb para cauda baixa ──
    out.append(f"\n  Penalização para cauda baixa de IMDb:")

    def recalibrate_imdb(row):
        est = row["imdb_estimate"]
        s = row["stability"]
        t = row["tension"]
        e = row["ending_shift"]

        # Penalização: se combinação de sinais ruins, puxar para baixo
        penalty = 0.0

        # Estabilidade muito baixa (P10)
        if s < df["stability"].quantile(0.10):
            penalty -= 0.3

        # Tensão muito alta (P90)
        if t > df["tension"].quantile(0.90):
            penalty -= 0.2

        # Ending shift extremo (P90)
        if e > df["ending_shift"].quantile(0.90):
            penalty -= 0.2

        # Combinação: baixa estabilidade + alta tensão
        if s < df["stability"].quantile(0.25) and t > df["tension"].quantile(0.75):
            penalty -= 0.3

        return round(max(2.0, est + penalty), 1)

    df["imdb_estimate_v2"] = df.apply(recalibrate_imdb, axis=1)

    return df


# ═══════════════════════════════════════════════════════════════════
# RODADA 3 — VALIDAÇÃO
# ═══════════════════════════════════════════════════════════════════

def validate(df):
    section("RODADA 3 — VALIDAÇÃO PÓS-CALIBRAÇÃO")

    has = df[df["imdb_actual"].notna()].copy()

    # v1 vs v2
    has["error_v1"] = (has["imdb_estimate"] - has["imdb_actual"]).abs()
    has["error_v2"] = (has["imdb_estimate_v2"] - has["imdb_actual"]).abs()

    mae_v1 = has["error_v1"].mean()
    mae_v2 = has["error_v2"].mean()
    med_v1 = has["error_v1"].median()
    med_v2 = has["error_v2"].median()

    out.append(f"\n  IMDb estimado — v1 vs v2:")
    out.append(f"    {'Métrica':<20s} {'v1':>8s} {'v2':>8s} {'Δ':>8s}")
    out.append(f"    {'-'*45}")
    out.append(f"    {'MAE':<20s} {mae_v1:>8.3f} {mae_v2:>8.3f} {mae_v2-mae_v1:>+8.3f}")
    out.append(f"    {'Erro mediano':<20s} {med_v1:>8.3f} {med_v2:>8.3f} {med_v2-med_v1:>+8.3f}")

    # Faixa
    in_range_v1 = ((has["imdb_actual"] >= has["imdb_range_low"]) &
                    (has["imdb_actual"] <= has["imdb_range_high"])).mean()
    out.append(f"    {'Dentro da faixa':<20s} {in_range_v1:>8.1%}     —")

    # Por faixa de IMDb real
    out.append(f"\n  MAE por faixa de IMDb real:")
    out.append(f"    {'Faixa':<15s} {'n':>5s} {'MAE v1':>8s} {'MAE v2':>8s}")
    out.append(f"    {'-'*40}")
    for lo, hi, label in [(2, 5, "2.0-5.0"), (5, 6, "5.0-6.0"), (6, 7, "6.0-7.0"),
                           (7, 8, "7.0-8.0"), (8, 10, "8.0-9.0")]:
        sub = has[(has["imdb_actual"] >= lo) & (has["imdb_actual"] < hi)]
        if len(sub) > 5:
            out.append(f"    {label:<15s} {len(sub):>5d} {sub['error_v1'].mean():>8.3f} "
                       f"{sub['error_v2'].mean():>8.3f}")

    # Por gênero
    out.append(f"\n  MAE por gênero (v2):")
    for genre in ["Thriller", "Horror", "Comedy", "Drama", "Action", "Crime"]:
        gsub = has[has["genre"].str.contains(genre, na=False)]
        if len(gsub) > 10:
            out.append(f"    {genre:<12s}  n={len(gsub):>4d}  MAE={gsub['error_v2'].mean():.3f}")

    # Correlação estimativa vs real
    rho_v1, _ = stats.spearmanr(has["imdb_estimate"], has["imdb_actual"])
    rho_v2, _ = stats.spearmanr(has["imdb_estimate_v2"], has["imdb_actual"])
    out.append(f"\n  Spearman (estimativa vs real):")
    out.append(f"    v1: ρ={rho_v1:+.3f}")
    out.append(f"    v2: ρ={rho_v2:+.3f}")

    # ROI v2 vs real
    if "revenue_multiple_actual" in has.columns:
        rm = has[["roi_tendency_v2", "revenue_multiple_actual"]].dropna(subset=["revenue_multiple_actual"])
        out.append(f"\n  ROI tendency v2 vs revenue_multiple real:")
        for t in ["desfavorável", "neutro", "favorável"]:
            sub = rm[rm["roi_tendency_v2"] == t]["revenue_multiple_actual"]
            if len(sub) > 0:
                out.append(f"    {t:<15s}  n={len(sub):>4d}  median={sub.median():.2f}x")

    # Perfis v2 — separação de IMDb
    out.append(f"\n  Perfis v2 — power de separação:")
    profiles = df["profile_v2"].unique()
    imdbs = {p: df[df["profile_v2"] == p]["imdb_actual"].dropna() for p in profiles}
    imdbs = {p: v for p, v in imdbs.items() if len(v) > 5}
    if len(imdbs) >= 2:
        groups = list(imdbs.values())
        h_stat, p_val = stats.kruskal(*groups)
        out.append(f"    Kruskal-Wallis: H={h_stat:.2f}, p={p_val:.4f} "
                   f"{'— perfis separam IMDb significativamente' if p_val < 0.05 else '— sem separação significativa'}")


def main():
    ensure_dirs()
    df = pd.read_csv(FEATURES / "scored_dataset.csv")

    # Verificar se revenue_multiple_actual existe, senão calcular
    if "revenue_multiple_actual" not in df.columns:
        roi_df = pd.read_csv(FEATURES / "analysis_dataset_with_roi.csv")
        if "revenue_multiple" in roi_df.columns:
            rm_map = roi_df.set_index("title")["revenue_multiple"]
            df["revenue_multiple_actual"] = df["title"].map(rm_map)

    # Rodada 1
    has = diagnose(df)

    # Rodada 2
    df = recalibrate(df)

    # Rodada 3
    validate(df)

    # Salvar
    report = "\n".join(out)
    print(report)

    df.to_csv(FEATURES / "scored_dataset_v2.csv", index=False)
    (RESULTS / "calibration_report.txt").write_text(report, encoding="utf-8")

    print(f"\nSalvo:")
    print(f"  {FEATURES / 'scored_dataset_v2.csv'}")
    print(f"  {RESULTS / 'calibration_report.txt'}")


if __name__ == "__main__":
    main()
