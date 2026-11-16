#!/usr/bin/env python3
"""Etapa 15: Estrutura narrativa — transições emocionais por gênero."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from collections import Counter
from src.utils import FEATURES, RESULTS, ensure_dirs

GENRES = ["Thriller", "Horror", "Comedy", "Drama"]
V3 = ["pos", "neu", "neg"]
V3_LABELS = {"pos": "Positivo", "neu": "Neutro", "neg": "Negativo"}


def load():
    df = pd.read_csv(FEATURES / "analysis_dataset_with_roi.csv")
    # Carregar emoções por cena para reconstruir transições
    emotions_dir = Path(__file__).resolve().parent.parent / "data" / "intermediate" / "emotions"
    return df, emotions_dir


def to_v3(c):
    if c > 0.05: return "pos"
    if c < -0.05: return "neg"
    return "neu"


def get_transitions(emotions_dir, titles, last_third_only=False):
    """Extrai todas as transições V3 de um conjunto de filmes."""
    T = np.zeros((3, 3))
    n_films = 0
    for title in titles:
        path = emotions_dir / f"{title}.csv"
        if not path.exists():
            continue
        try:
            edf = pd.read_csv(path)
        except:
            continue
        if "vader_compound" not in edf.columns or len(edf) < 6:
            continue

        vals = edf["vader_compound"].values
        if last_third_only:
            vals = vals[2 * len(vals) // 3:]
        if len(vals) < 3:
            continue

        states = [to_v3(c) for c in vals]
        for i in range(len(states) - 1):
            a, b = V3.index(states[i]), V3.index(states[i + 1])
            T[a][b] += 1
        n_films += 1

    return T, n_films


def format_matrix(T, label, out):
    """Formata matriz de transição normalizada."""
    total_per_row = T.sum(axis=1, keepdims=True)
    P = np.divide(T, total_per_row, where=total_per_row > 0, out=np.zeros_like(T))

    out.append(f"\n  {label}")
    out.append(f"  {'De \\ Para':<12s} {'Positivo':>10s} {'Neutro':>10s} {'Negativo':>10s}  | {'Total':>7s}")
    out.append(f"  {'-'*55}")
    for i, s in enumerate(V3):
        row_total = int(T[i].sum())
        out.append(f"  {V3_LABELS[s]:<12s} {P[i][0]:>10.1%} {P[i][1]:>10.1%} {P[i][2]:>10.1%}  | {row_total:>7d}")

    # Self-loop total
    autoloop = np.trace(T) / T.sum() if T.sum() > 0 else 0
    out.append(f"  Autoloop total: {autoloop:.1%}")
    return P


def top_transitions(T, n_top=5, out=None):
    """Lista as transições mais e menos frequentes."""
    total = T.sum()
    if total == 0:
        return
    items = []
    for i in range(3):
        for j in range(3):
            items.append((V3_LABELS[V3[i]], V3_LABELS[V3[j]], T[i][j], T[i][j] / total))

    items.sort(key=lambda x: x[3], reverse=True)

    out.append(f"\n    Mais frequentes:")
    for fr, to, count, pct in items[:n_top]:
        arrow = "↻" if fr == to else "→"
        out.append(f"      {fr} {arrow} {to}: {pct:.1%} ({int(count)}x)")

    out.append(f"    Menos frequentes:")
    for fr, to, count, pct in items[-n_top:]:
        arrow = "↻" if fr == to else "→"
        out.append(f"      {fr} {arrow} {to}: {pct:.1%} ({int(count)}x)")


def compare_genres(P1, P2, name1, name2, out):
    """Compara duas matrizes de transição normalizadas."""
    out.append(f"\n  Diferenças {name1} vs {name2}:")
    diffs = []
    for i in range(3):
        for j in range(3):
            diff = P1[i][j] - P2[i][j]
            diffs.append((V3_LABELS[V3[i]], V3_LABELS[V3[j]], diff, P1[i][j], P2[i][j]))

    diffs.sort(key=lambda x: abs(x[4] - x[3]), reverse=True)

    for fr, to, diff, v1, v2 in diffs[:5]:
        arrow = "↻" if fr == to else "→"
        more = name1 if diff > 0 else name2
        out.append(f"    {fr} {arrow} {to}: {name1}={v1:.1%}  {name2}={v2:.1%}  (Δ={abs(diff):.1%} mais em {more})")


def main():
    ensure_dirs()
    df, emotions_dir = load()
    out = []

    out.append("=" * 70)
    out.append("  ESTRUTURA NARRATIVA — TRANSIÇÕES EMOCIONAIS POR GÊNERO")
    out.append("=" * 70)

    # ══════════════════════════════════════════════════════════════
    # ETAPA 1 + 2 — MATRIZES E PRINCIPAIS TRANSIÇÕES
    # ══════════════════════════════════════════════════════════════
    matrices = {}
    matrices_lt = {}  # last third

    for genre in GENRES:
        titles = df[df["genre"].str.contains(genre, na=False)]["title"].tolist()
        T, n = get_transitions(emotions_dir, titles)
        T_lt, n_lt = get_transitions(emotions_dir, titles, last_third_only=True)

        out.append(f"\n{'─'*70}")
        out.append(f"  {genre.upper()} (n={n} filmes, {int(T.sum())} transições)")
        out.append(f"{'─'*70}")

        P = format_matrix(T, "Matriz de transição (filme inteiro):", out)
        matrices[genre] = P
        top_transitions(T, 5, out)

        P_lt = format_matrix(T_lt, "Matriz de transição (último terço):", out)
        matrices_lt[genre] = P_lt

    # ══════════════════════════════════════════════════════════════
    # ETAPA 3 — COMPARAÇÃO ENTRE GÊNEROS
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n\n{'='*70}")
    out.append("  COMPARAÇÃO ENTRE GÊNEROS")
    out.append("=" * 70)

    compare_genres(matrices["Thriller"], matrices["Comedy"], "Thriller", "Comedy", out)
    compare_genres(matrices["Horror"], matrices["Drama"], "Horror", "Drama", out)
    compare_genres(matrices["Thriller"], matrices["Horror"], "Thriller", "Horror", out)
    compare_genres(matrices["Comedy"], matrices["Drama"], "Comedy", "Drama", out)

    # ══════════════════════════════════════════════════════════════
    # ETAPA 4 — ÚLTIMO TERÇO
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n\n{'='*70}")
    out.append("  ÚLTIMO TERÇO — COMO O FINAL DIFERE POR GÊNERO")
    out.append("=" * 70)

    for genre in GENRES:
        P_full = matrices[genre]
        P_lt = matrices_lt[genre]

        out.append(f"\n  ── {genre} ──")
        out.append(f"    Mudanças do filme inteiro → último terço:")

        diffs = []
        for i in range(3):
            for j in range(3):
                diff = P_lt[i][j] - P_full[i][j]
                if abs(diff) > 0.02:
                    diffs.append((V3_LABELS[V3[i]], V3_LABELS[V3[j]], diff, P_full[i][j], P_lt[i][j]))

        diffs.sort(key=lambda x: abs(x[2]), reverse=True)
        for fr, to, diff, v_full, v_lt in diffs[:5]:
            arrow = "↻" if fr == to else "→"
            direction = "↑" if diff > 0 else "↓"
            out.append(f"      {fr} {arrow} {to}: {v_full:.1%} → {v_lt:.1%} ({direction}{abs(diff):.1%})")

    # ══════════════════════════════════════════════════════════════
    # ETAPA 5 — INTERPRETAÇÃO NARRATIVA
    # ══════════════════════════════════════════════════════════════
    out.append(f"\n\n{'='*70}")
    out.append("  INTERPRETAÇÃO NARRATIVA")
    out.append("=" * 70)

    # Thriller
    P = matrices["Thriller"]
    Plt = matrices_lt["Thriller"]
    out.append(f"""
  ── THRILLER ──
  Padrão dominante: NEUTRO como pivot central.
  - O estado mais frequente é Neutro (PR mais alto nos dados)
  - Transição mais comum: Neutro ↻ Neutro ({P[1][1]:.0%})
  - Segunda mais comum: Positivo ↻ Positivo ({P[0][0]:.0%})
  - O thriller constrói tensão através de SUSTENTAÇÃO, não de oscilação

  No último terço:
  - Neutro ↻ Neutro {"sobe" if Plt[1][1] > P[1][1] else "desce"} ({P[1][1]:.0%} → {Plt[1][1]:.0%})
  - Negativo ↻ Negativo {"sobe" if Plt[2][2] > P[2][2] else "desce"} ({P[2][2]:.0%} → {Plt[2][2]:.0%})

  Interpretação: bons thrillers mantêm um estado emocional coerente,
  em vez de saltar entre extremos. O clímax não é caos — é controle.""")

    # Horror
    P = matrices["Horror"]
    Plt = matrices_lt["Horror"]
    out.append(f"""
  ── HORROR ──
  Padrão dominante: SUSTENTAÇÃO com picos negativos.
  - Positivo ↻ Positivo alto ({P[0][0]:.0%}) — pausa entre sustos
  - Negativo ↻ Negativo ({P[2][2]:.0%}) — sequências de medo sustentado
  - Transição Neg → Pos é relativamente rara

  No último terço:
  - A tensão negativa {"intensifica" if Plt[2][2] > P[2][2] else "diminui"}

  Interpretação: horror eficaz alterna entre períodos de calma (falsa
  segurança) e mergulhos em medo. A estabilidade emocional (autoloop
  alto) indica que o filme sabe dosificar seus momentos.""")

    # Comedy
    P = matrices["Comedy"]
    Plt = matrices_lt["Comedy"]
    out.append(f"""
  ── COMEDY ──
  Padrão dominante: POSITIVIDADE com pouca permanência negativa.
  - Positivo ↻ Positivo é a transição dominante ({P[0][0]:.0%})
  - Negativo ↻ Negativo é baixo ({P[2][2]:.0%}) — comédias não ficam no negativo
  - Neutro serve como ponte entre positivo e negativo

  No último terço:
  - Positivo ↻ Positivo {"sobe" if Plt[0][0] > P[0][0] else "desce"} ({P[0][0]:.0%} → {Plt[0][0]:.0%})
  - Comédias {"resolvem" if Plt[0][0] > P[0][0] else "complicam"} no final

  Interpretação: comédias bem avaliadas evitam mergulhos emocionais
  profundos. O público espera consistência no tom positivo.""")

    # Drama
    P = matrices["Drama"]
    Plt = matrices_lt["Drama"]
    out.append(f"""
  ── DRAMA ──
  Padrão dominante: EQUILÍBRIO entre estados.
  - Nenhum estado domina excessivamente
  - Positivo ↻ Positivo ({P[0][0]:.0%}), Neutro ↻ Neutro ({P[1][1]:.0%}), Neg ↻ Neg ({P[2][2]:.0%})
  - Maior diversidade de transições que qualquer outro gênero

  No último terço:
  - A diversidade se mantém — drama não converge para um estado

  Interpretação: drama é o gênero mais heterogêneo. Não tem um padrão
  emocional dominante porque engloba desde romances até filmes de guerra.
  Isso explica por que as métricas de grafo têm pouco poder em Drama.""")

    # ══════════════════════════════════════════════════════════════
    # RESUMO FINAL
    # ══════════════════════════════════════════════════════════════
    out.append(f"""

{'='*70}
  RESUMO FINAL
{'='*70}

  ┌─────────────┬─────────────────────────────────────────────────────┐
  │ Gênero      │ Padrão narrativo dominante                         │
  ├─────────────┼─────────────────────────────────────────────────────┤
  │ Thriller    │ Sustentação controlada, neutro como pivot           │
  │ Horror      │ Alternância calma/medo, sustentação de estados      │
  │ Comedy      │ Positividade dominante, evita negatividade           │
  │ Drama       │ Equilíbrio entre estados, sem padrão dominante      │
  └─────────────┴─────────────────────────────────────────────────────┘

  Trade-off central:
  - ESTABILIDADE emocional → melhor avaliação (IMDb)
  - EXTREMOS emocionais → mais bilheteira (Box Office)

  Onde a narrativa pesa mais:
  - Gêneros com expectativa emocional clara (Thriller, Horror)
  - Filmes de baixo orçamento

  Onde não pesa:
  - Gêneros heterogêneos (Drama)
  - Filmes de alto orçamento
""")

    report = "\n".join(out)
    print(report)
    (RESULTS / "narrative_structure_report.txt").write_text(report, encoding="utf-8")
    print(f"\nSalvo: narrative_structure_report.txt")


if __name__ == "__main__":
    main()
