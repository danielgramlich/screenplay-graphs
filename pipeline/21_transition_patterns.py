#!/usr/bin/env python3
"""Etapa 21: Testar hipóteses sobre padrões de transição emocional.

H1 — Final feliz após vale: filmes que terminam positivo após cenas negativas/neutras
     são melhor avaliados?

H2 — Velocidade de mudança: quanto mais lenta a transição entre tons opostos,
     melhor o filme? Passagens graduais vs saltos bruscos.

H3 — Saltos extremos: transições diretas Pos→Neg ou Neg→Pos (sem passar pelo Neutro)
     indicam filme bom ou ruim?
"""

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

TARGETS = [
    ("imdb_rating", "IMDb"),
    ("log_box_office", "log(BoxOffice)"),
    ("log_revenue_multiple", "log(RevMultiple)"),
]
GENRES = ["Thriller", "Horror", "Comedy", "Drama"]
GENRE_DUMMIES = ["Drama", "Comedy", "Action", "Crime", "Thriller",
                  "Adventure", "Romance", "Horror"]
EMOTIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "intermediate" / "emotions"


def to_v3(c):
    if c > 0.05: return "pos"
    if c < -0.05: return "neg"
    return "neu"


def extract_transition_features(title):
    """Extrai features de transição detalhadas para um filme."""
    path = EMOTIONS_DIR / f"{title}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "vader_compound" not in df.columns or len(df) < 10:
        return None

    compounds = df["vader_compound"].values
    n = len(compounds)
    states = [to_v3(c) for c in compounds]

    # ═══════════════════════════════════════════════════════════
    # H1 — FINAL FELIZ APÓS VALE
    # ═══════════════════════════════════════════════════════════
    # Últimas 5 cenas
    last5 = states[-5:]
    last5_vals = compounds[-5:]

    # O final é positivo?
    final_positive = last5[-1] == "pos"
    final_positive_ratio = sum(1 for s in last5 if s == "pos") / len(last5)

    # Houve vale (negativo) no último terço, ANTES do final?
    lt_start = 2 * n // 3
    pre_final = states[lt_start:-5]  # último terço excluindo últimas 5
    pre_final_vals = compounds[lt_start:-5]

    had_valley = "neg" in pre_final if pre_final else False
    valley_then_happy = had_valley and final_positive_ratio > 0.5

    # Magnitude da virada: diferença entre mínimo do pré-final e média do final
    if len(pre_final_vals) > 0:
        turnaround = float(np.mean(last5_vals) - np.min(pre_final_vals))
    else:
        turnaround = 0.0

    # ═══════════════════════════════════════════════════════════
    # H2 — VELOCIDADE DE MUDANÇA / GRADUALIDADE
    # ═══════════════════════════════════════════════════════════
    # Contar transições que passam pelo neutro vs saltos diretos
    direct_jumps = 0      # pos→neg ou neg→pos (sem neutro no meio)
    gradual_transitions = 0  # pos→neu→neg ou neg→neu→pos (com neutro intermediário)
    total_transitions = 0

    for i in range(len(states) - 1):
        total_transitions += 1
        a, b = states[i], states[i + 1]
        if (a == "pos" and b == "neg") or (a == "neg" and b == "pos"):
            direct_jumps += 1

    # Procurar sequências de 3 cenas com transição gradual
    for i in range(len(states) - 2):
        a, b, c = states[i], states[i + 1], states[i + 2]
        if a == "pos" and b == "neu" and c == "neg":
            gradual_transitions += 1
        elif a == "neg" and b == "neu" and c == "pos":
            gradual_transitions += 1

    direct_jump_ratio = direct_jumps / total_transitions if total_transitions > 0 else 0
    gradual_ratio = gradual_transitions / max(1, direct_jumps + gradual_transitions)

    # Tempo médio entre mudanças de tom (duração dos "blocos" emocionais)
    block_lengths = []
    current_state = states[0]
    current_length = 1
    for i in range(1, len(states)):
        if states[i] == current_state:
            current_length += 1
        else:
            block_lengths.append(current_length)
            current_state = states[i]
            current_length = 1
    block_lengths.append(current_length)
    avg_block_length = np.mean(block_lengths)
    median_block_length = np.median(block_lengths)

    # ═══════════════════════════════════════════════════════════
    # H3 — SALTOS EXTREMOS (SEM NEUTRO)
    # ═══════════════════════════════════════════════════════════
    # Proporção de saltos diretos pos↔neg vs total de transições não-autoloop
    non_autoloop = sum(1 for i in range(len(states) - 1) if states[i] != states[i + 1])
    extreme_jump_ratio = direct_jumps / non_autoloop if non_autoloop > 0 else 0

    # Tom predominante (neutro vs extremos)
    pct_neutral = sum(1 for s in states if s == "neu") / n

    # Volatilidade emocional (diferenças absolutas entre cenas)
    diffs = np.abs(np.diff(compounds))
    mean_diff = float(np.mean(diffs))
    max_diff = float(np.max(diffs))

    return {
        # H1
        "final_positive": int(final_positive),
        "final_positive_ratio": round(final_positive_ratio, 3),
        "had_valley_then_happy": int(valley_then_happy),
        "turnaround_magnitude": round(turnaround, 4),
        # H2
        "direct_jump_ratio": round(direct_jump_ratio, 4),
        "gradual_ratio": round(gradual_ratio, 4),
        "avg_block_length": round(avg_block_length, 2),
        "median_block_length": round(median_block_length, 1),
        # H3
        "extreme_jump_ratio": round(extreme_jump_ratio, 4),
        "pct_neutral": round(pct_neutral, 4),
        "mean_scene_diff": round(mean_diff, 4),
        "max_scene_diff": round(max_diff, 4),
    }


def main():
    ensure_dirs()
    df = pd.read_csv(FEATURES / "analysis_dataset_with_roi.csv")
    for g in GENRE_DUMMIES:
        df[f"g_{g}"] = df["genre"].str.contains(g, na=False).astype(int)

    out = []
    out.append("=" * 70)
    out.append("  HIPÓTESES SOBRE PADRÕES DE TRANSIÇÃO EMOCIONAL")
    out.append("=" * 70)

    # Extrair features
    out.append("\n  Extraindo features de transição...")
    feat_rows = []
    for _, row in df.iterrows():
        feats = extract_transition_features(row["title"])
        if feats:
            feat_rows.append({"title": row["title"], **feats})

    feat_df = pd.DataFrame(feat_rows)
    df = df.merge(feat_df, on="title", how="inner")
    out.append(f"  {len(feat_df)} filmes com features extraídas")

    # Distribuições
    out.append(f"\n{'='*70}")
    out.append("  DISTRIBUIÇÕES DAS NOVAS FEATURES")
    out.append("=" * 70)
    new_feats = ["final_positive_ratio", "had_valley_then_happy", "turnaround_magnitude",
                 "direct_jump_ratio", "gradual_ratio", "avg_block_length",
                 "extreme_jump_ratio", "pct_neutral", "mean_scene_diff"]
    for f in new_feats:
        s = df[f].dropna()
        out.append(f"  {f:<28s}  mean={s.mean():.3f}  std={s.std():.3f}  "
                   f"[{s.quantile(.25):.3f}, {s.quantile(.75):.3f}]")

    # ═══════════════════════════════════════════════════════════
    # H1 — FINAL FELIZ APÓS VALE
    # ═══════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  H1 — FINAL FELIZ APÓS VALE EMOCIONAL")
    out.append("=" * 70)
    out.append("  Pergunta: filmes que caem e depois terminam positivo são melhor avaliados?")

    # Comparar grupos
    for tcol, tname in TARGETS:
        valid = df[df[tcol].notna()]
        g_yes = valid[valid["had_valley_then_happy"] == 1][tcol]
        g_no = valid[valid["had_valley_then_happy"] == 0][tcol]
        if len(g_yes) > 10 and len(g_no) > 10:
            u, p = stats.mannwhitneyu(g_yes, g_no, alternative="two-sided")
            out.append(f"\n  {tname}:")
            out.append(f"    Com virada (vale→feliz): n={len(g_yes)}, média={g_yes.mean():.2f}")
            out.append(f"    Sem virada:              n={len(g_no)}, média={g_no.mean():.2f}")
            out.append(f"    Δ = {g_yes.mean() - g_no.mean():+.3f}  p={p:.4f} {'*' if p < 0.05 else ''}")

    # Correlação com turnaround_magnitude
    out.append(f"\n  Magnitude da virada (correlação com targets):")
    for tcol, tname in TARGETS:
        v = df[[tcol, "turnaround_magnitude"]].dropna()
        if len(v) > 30:
            rho, p = stats.spearmanr(v["turnaround_magnitude"], v[tcol])
            out.append(f"    {tname}: ρ={rho:+.3f} (p={p:.4f}) {'*' if p < 0.05 else ''}")

    # Por gênero
    out.append(f"\n  Virada por gênero (IMDb):")
    for genre in GENRES:
        gsub = df[(df["genre"].str.contains(genre, na=False)) & (df["imdb_rating"].notna())]
        g_yes = gsub[gsub["had_valley_then_happy"] == 1]["imdb_rating"]
        g_no = gsub[gsub["had_valley_then_happy"] == 0]["imdb_rating"]
        if len(g_yes) > 5 and len(g_no) > 5:
            diff = g_yes.mean() - g_no.mean()
            _, p = stats.mannwhitneyu(g_yes, g_no, alternative="two-sided")
            out.append(f"    {genre:<12s}  virada={g_yes.mean():.2f} (n={len(g_yes)})  "
                       f"sem={g_no.mean():.2f} (n={len(g_no)})  Δ={diff:+.2f}  p={p:.3f} {'*' if p < 0.05 else ''}")

    # ═══════════════════════════════════════════════════════════
    # H2 — VELOCIDADE DE MUDANÇA
    # ═══════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  H2 — VELOCIDADE DE MUDANÇA / GRADUALIDADE")
    out.append("=" * 70)
    out.append("  Pergunta: transições graduais (com neutro intermediário) são melhores")
    out.append("  que saltos bruscos? Blocos emocionais mais longos = melhor?")

    h2_feats = ["direct_jump_ratio", "gradual_ratio", "avg_block_length", "mean_scene_diff"]
    h2_labels = {
        "direct_jump_ratio": "Saltos diretos (%)",
        "gradual_ratio": "Gradualidade (%)",
        "avg_block_length": "Duração média do bloco",
        "mean_scene_diff": "Diferença média entre cenas",
    }

    out.append(f"\n  Correlações globais:")
    out.append(f"  {'Feature':<30s} {'IMDb':>12s} {'BoxOffice':>12s} {'Revenue':>12s}")
    out.append(f"  {'-'*70}")
    for feat in h2_feats:
        vals = []
        for tcol, tname in TARGETS:
            v = df[[feat, tcol]].dropna()
            if len(v) > 30:
                rho, p = stats.spearmanr(v[feat], v[tcol])
                sig = "*" if p < 0.05 else ""
                vals.append(f"{rho:+.3f}{sig}")
            else:
                vals.append("—")
        out.append(f"  {h2_labels.get(feat, feat):<30s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s}")

    # Quartis de avg_block_length vs IMDb
    out.append(f"\n  IMDb por duração dos blocos emocionais:")
    valid = df[df["imdb_rating"].notna()].copy()
    try:
        valid["block_q"] = pd.qcut(valid["avg_block_length"], 4,
                                    labels=["Q1 (curtos)", "Q2", "Q3", "Q4 (longos)"])
        out.append(f"  {'Quartil':<18s} {'n':>5s} {'IMDb':>8s}")
        out.append(f"  {'-'*35}")
        for q in ["Q1 (curtos)", "Q2", "Q3", "Q4 (longos)"]:
            qs = valid[valid["block_q"] == q]["imdb_rating"]
            if len(qs) > 5:
                out.append(f"  {q:<18s} {len(qs):>5d} {qs.mean():>8.2f}")
    except ValueError:
        out.append("  (quartis não disponíveis — valores muito concentrados)")

    # Por gênero
    out.append(f"\n  Duração de bloco vs IMDb por gênero:")
    for genre in GENRES:
        gsub = df[(df["genre"].str.contains(genre, na=False)) & (df["imdb_rating"].notna())]
        if len(gsub) > 20:
            rho, p = stats.spearmanr(gsub["avg_block_length"], gsub["imdb_rating"])
            sig = "*" if p < 0.05 else ""
            out.append(f"    {genre:<12s}  ρ={rho:+.3f}  p={p:.3f} {sig}  n={len(gsub)}")

    # ═══════════════════════════════════════════════════════════
    # H3 — SALTOS EXTREMOS
    # ═══════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  H3 — SALTOS EXTREMOS (POS↔NEG SEM NEUTRO)")
    out.append("=" * 70)
    out.append("  Pergunta: saltos diretos positivo↔negativo (sem passar pelo neutro)")
    out.append("  indicam filme bom ou ruim?")

    out.append(f"\n  Correlações globais:")
    for tcol, tname in TARGETS:
        v = df[["extreme_jump_ratio", tcol]].dropna()
        if len(v) > 30:
            rho, p = stats.spearmanr(v["extreme_jump_ratio"], v[tcol])
            out.append(f"    {tname}: ρ={rho:+.3f} (p={p:.4f}) {'*' if p < 0.05 else ''}")

    # Complementar: pct_neutral
    out.append(f"\n  Tom neutro dominante vs targets:")
    for tcol, tname in TARGETS:
        v = df[["pct_neutral", tcol]].dropna()
        if len(v) > 30:
            rho, p = stats.spearmanr(v["pct_neutral"], v[tcol])
            out.append(f"    {tname}: ρ={rho:+.3f} (p={p:.4f}) {'*' if p < 0.05 else ''}")

    # Quartis de extreme_jump_ratio
    out.append(f"\n  IMDb por nível de saltos extremos:")
    try:
        valid["jump_q"] = pd.qcut(valid["extreme_jump_ratio"], 4,
                                   labels=["Q1 (poucos)", "Q2", "Q3", "Q4 (muitos)"],
                                   duplicates="drop")
        out.append(f"  {'Quartil':<18s} {'n':>5s} {'IMDb':>8s}")
        out.append(f"  {'-'*35}")
        for q in sorted(valid["jump_q"].unique()):
            qs = valid[valid["jump_q"] == q]["imdb_rating"]
            if len(qs) > 5:
                out.append(f"  {str(q):<18s} {len(qs):>5d} {qs.mean():>8.2f}")
    except ValueError:
        out.append("  (quartis não disponíveis)")

    # Por gênero
    out.append(f"\n  Saltos extremos vs IMDb por gênero:")
    for genre in GENRES:
        gsub = df[(df["genre"].str.contains(genre, na=False)) & (df["imdb_rating"].notna())]
        if len(gsub) > 20:
            rho, p = stats.spearmanr(gsub["extreme_jump_ratio"], gsub["imdb_rating"])
            sig = "*" if p < 0.05 else ""
            out.append(f"    {genre:<12s}  ρ={rho:+.3f}  p={p:.3f} {sig}  n={len(gsub)}")

    # ═══════════════════════════════════════════════════════════
    # MODELO CONTROLADO COM TODAS AS NOVAS FEATURES
    # ═══════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  MODELO CONTROLADO (OLS)")
    out.append("=" * 70)

    genre_cols = [f"g_{g}" for g in GENRE_DUMMIES]
    selected = ["turnaround_magnitude", "direct_jump_ratio", "avg_block_length",
                 "extreme_jump_ratio", "pct_neutral"]
    sel_labels = {
        "turnaround_magnitude": "Virada final",
        "direct_jump_ratio": "Saltos diretos",
        "avg_block_length": "Duração do bloco",
        "extreme_jump_ratio": "Saltos extremos",
        "pct_neutral": "% Neutro",
    }

    for tcol, tname in TARGETS:
        ctrl = ["year"] + genre_cols
        if tcol != "imdb_rating":
            ctrl += ["log_budget"]
        feats = selected + ctrl
        sub = df[feats + [tcol]].dropna()
        if len(sub) < 30:
            continue

        X = sm.add_constant(sub[feats])
        model = sm.OLS(sub[tcol], X).fit()

        out.append(f"\n  {tname} (n={len(sub)}, R²={model.rsquared:.4f}):")
        for feat in selected:
            coef = model.params[feat]
            p = model.pvalues[feat]
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("~" if p < 0.10 else ""))
            out.append(f"    {sel_labels.get(feat, feat):<25s}  β={coef:>+8.4f}  p={p:.4f} {sig}")

    # ═══════════════════════════════════════════════════════════
    # INTERPRETAÇÃO FINAL
    # ═══════════════════════════════════════════════════════════
    out.append(f"\n{'='*70}")
    out.append("  INTERPRETAÇÃO")
    out.append("=" * 70)

    # Coletar evidências
    v_turn = df[["turnaround_magnitude", "imdb_rating"]].dropna()
    rho_turn, p_turn = stats.spearmanr(v_turn["turnaround_magnitude"], v_turn["imdb_rating"])
    v_block = df[["avg_block_length", "imdb_rating"]].dropna()
    rho_block, p_block = stats.spearmanr(v_block["avg_block_length"], v_block["imdb_rating"])
    v_jump = df[["extreme_jump_ratio", "imdb_rating"]].dropna()
    rho_jump, p_jump = stats.spearmanr(v_jump["extreme_jump_ratio"], v_jump["imdb_rating"])
    v_neu = df[["pct_neutral", "imdb_rating"]].dropna()
    rho_neu, p_neu = stats.spearmanr(v_neu["pct_neutral"], v_neu["imdb_rating"])

    out.append(f"""
  H1 — Final feliz após vale:
    Virada magnitude vs IMDb: ρ={rho_turn:+.3f} (p={p_turn:.4f})
    {'SUSTENTADA' if rho_turn > 0.05 and p_turn < 0.05 else 'PARCIAL' if p_turn < 0.10 else 'NÃO SUSTENTADA'}
    {'Filmes com virada emocional no final tendem a ser melhor avaliados.' if rho_turn > 0 and p_turn < 0.05 else 'A virada no final não tem efeito significativo global.' if p_turn > 0.10 else 'Efeito fraco — verificar por gênero.'}

  H2 — Velocidade de mudança:
    Duração do bloco vs IMDb: ρ={rho_block:+.3f} (p={p_block:.4f})
    {'SUSTENTADA' if rho_block > 0.05 and p_block < 0.05 else 'PARCIAL' if p_block < 0.10 else 'NÃO SUSTENTADA'}
    {'Blocos emocionais mais longos (mudanças mais lentas) associam-se a melhor avaliação.' if rho_block > 0 and p_block < 0.05 else 'A velocidade da mudança não impacta significativamente.' if p_block > 0.10 else 'Efeito marginal — verificar por gênero.'}

  H3 — Saltos extremos:
    Saltos extremos vs IMDb: ρ={rho_jump:+.3f} (p={p_jump:.4f})
    % Neutro vs IMDb: ρ={rho_neu:+.3f} (p={p_neu:.4f})
    {'SUSTENTADA' if (rho_jump < -0.05 and p_jump < 0.05) or (rho_neu > 0.05 and p_neu < 0.05) else 'PARCIAL' if p_jump < 0.10 or p_neu < 0.10 else 'NÃO SUSTENTADA'}
    {'Mais saltos extremos prejudicam a avaliação.' if rho_jump < 0 and p_jump < 0.05 else ''}
    {'Tom neutro dominante favorece a avaliação.' if rho_neu > 0 and p_neu < 0.05 else ''}
""")

    report = "\n".join(out)
    print(report)
    (RESULTS / "transition_patterns_report.txt").write_text(report, encoding="utf-8")

    # Salvar features para uso futuro
    feat_df.to_csv(FEATURES / "emotional_graphs" / "transition_features.csv", index=False)

    print(f"\nSalvo: transition_patterns_report.txt + transition_features.csv")


if __name__ == "__main__":
    main()
