#!/usr/bin/env python3
"""Etapa 19: Aplicar camada de scoring a todos os filmes do dataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils import FEATURES, RESULTS, ensure_dirs
from src.scoring import score_film


def main():
    ensure_dirs()

    # Carregar dataset com métricas + metadata
    df = pd.read_csv(FEATURES / "analysis_dataset_with_roi.csv")
    metric_cols = [c for c in df.columns if c.startswith("gv")]

    print(f"Dataset: {len(df)} filmes, {len(metric_cols)} métricas")

    # Scoring por filme
    scored_rows = []
    explanations = []

    for _, row in df.iterrows():
        title = row["title"]
        genre = str(row.get("genre", "default"))
        metrics = {col: row[col] for col in metric_cols if pd.notna(row[col])}

        result = score_film(metrics, genre)

        # Flatten para CSV
        idx = result["indices"]
        imdb = result["imdb_estimate"]
        roi = result["roi_estimate"]
        bo = result["box_office_tendency"]
        prof = result["profile"]
        qual = result["quality"]

        scored_rows.append({
            "title": title,
            "genre": genre,
            "year": row.get("year"),

            # Targets reais (para comparação)
            "imdb_actual": row.get("imdb_rating"),
            "box_office_actual": row.get("box_office"),
            "revenue_multiple_actual": row.get("revenue_multiple") if "revenue_multiple" in row.index else None,

            # Índices
            "stability": idx["stability"],
            "tension": idx["tension"],
            "recovery": idx["recovery"],
            "unresolved": idx["unresolved"],
            "ending_shift": idx["ending_shift"],

            # Perfil
            "profile": prof["profile"],

            # Estimativas
            "imdb_estimate": imdb["estimate"],
            "imdb_range_low": imdb["range"][0],
            "imdb_range_high": imdb["range"][1],
            "imdb_anchor": imdb["anchor"],
            "imdb_adjustment": imdb["adjustment"],
            "imdb_confidence": imdb["confidence"],

            # ROI
            "roi_score": roi["score"],
            "roi_tendency": roi["tendency"],

            # Box office
            "bo_score": bo["score"],
            "bo_tendency": bo["tendency"],

            # Quality score
            "quality_score": qual["score"],
            "quality_label": qual["label"],
        })

        # Explicação
        explanation_lines = result["explanation"]
        explanations.append({
            "title": title,
            "genre": genre,
            "explanation": " | ".join(explanation_lines),
        })

    scored_df = pd.DataFrame(scored_rows)
    explain_df = pd.DataFrame(explanations)

    # Salvar
    scored_df.to_csv(FEATURES / "scored_dataset.csv", index=False)
    explain_df.to_csv(RESULTS / "film_explanations.csv", index=False)

    # ── Relatório ──
    out = []
    out.append("=" * 70)
    out.append("  SCORING — RESULTADOS")
    out.append("=" * 70)

    out.append(f"\n  Filmes processados: {len(scored_df)}")

    # Distribuição de perfis
    out.append(f"\n  Perfis narrativos:")
    for profile, count in scored_df["profile"].value_counts().items():
        pct = count / len(scored_df) * 100
        out.append(f"    {profile:<20s}  {count:>4d}  ({pct:.1f}%)")

    # Distribuição de tendências
    out.append(f"\n  Tendência comercial:")
    for t, count in scored_df["bo_tendency"].value_counts().items():
        out.append(f"    {t:<15s}  {count:>4d}")

    out.append(f"\n  Tendência de retorno:")
    for t, count in scored_df["roi_tendency"].value_counts().items():
        out.append(f"    {t:<15s}  {count:>4d}")

    # Comparar estimativa vs real (onde disponível)
    has_imdb = scored_df[scored_df["imdb_actual"].notna()]
    if len(has_imdb) > 0:
        diff = has_imdb["imdb_estimate"] - has_imdb["imdb_actual"]
        out.append(f"\n  IMDb estimado vs real (n={len(has_imdb)}):")
        out.append(f"    Erro médio: {diff.mean():+.2f}")
        out.append(f"    Erro absoluto médio: {diff.abs().mean():.2f}")
        out.append(f"    Erro mediano: {diff.abs().median():.2f}")

        # % dentro da faixa
        in_range = ((has_imdb["imdb_actual"] >= has_imdb["imdb_range_low"]) &
                     (has_imdb["imdb_actual"] <= has_imdb["imdb_range_high"]))
        out.append(f"    Dentro da faixa estimada: {in_range.sum()}/{len(has_imdb)} ({in_range.mean():.1%})")

    # Top 5 e bottom 5 mais bem estimados
    if len(has_imdb) > 10:
        has_imdb = has_imdb.copy()
        has_imdb["abs_error"] = (has_imdb["imdb_estimate"] - has_imdb["imdb_actual"]).abs()

        out.append(f"\n  Exemplos — melhor estimados:")
        for _, r in has_imdb.nsmallest(5, "abs_error").iterrows():
            out.append(f"    {r['title']:<30s}  real={r['imdb_actual']:.1f}  est={r['imdb_estimate']:.1f}  "
                       f"perfil={r['profile']}")

        out.append(f"\n  Exemplos — pior estimados:")
        for _, r in has_imdb.nlargest(5, "abs_error").iterrows():
            out.append(f"    {r['title']:<30s}  real={r['imdb_actual']:.1f}  est={r['imdb_estimate']:.1f}  "
                       f"erro={r['abs_error']:+.1f}")

    # Exemplo de explicação
    out.append(f"\n  Exemplo de explicação:")
    sample = explain_df[explain_df["title"] == "fargo"]
    if not sample.empty:
        out.append(f"    Filme: Fargo")
        for line in sample.iloc[0]["explanation"].split(" | "):
            out.append(f"    → {line}")
    else:
        # Pegar primeiro disponível
        sample = explain_df.iloc[0]
        out.append(f"    Filme: {sample['title']}")
        for line in sample["explanation"].split(" | "):
            out.append(f"    → {line}")

    report = "\n".join(out)
    print(report)
    (RESULTS / "scoring_report.txt").write_text(report, encoding="utf-8")

    print(f"\nSalvo:")
    print(f"  {FEATURES / 'scored_dataset.csv'}")
    print(f"  {RESULTS / 'film_explanations.csv'}")
    print(f"  {RESULTS / 'scoring_report.txt'}")


if __name__ == "__main__":
    main()
