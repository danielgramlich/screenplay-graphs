#!/usr/bin/env python3
"""Etapa 12: Coletar orçamento (budget) via TMDb e calcular ROI."""

import os
import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests

from src.utils import FEATURES, METADATA, RESULTS, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ─── Configuração TMDb ───
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_FIND_URL = "https://api.themoviedb.org/3/find/{imdb_id}"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
RATE_LIMIT_DELAY = 0.25  # TMDb permite ~40 req/s, 0.25s é conservador
CACHE_PATH = FEATURES / "tmdb_budget_cache.json"


# ─── Funções de coleta ───

def load_cache() -> dict:
    """Carrega cache local para retomada parcial."""
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict):
    """Salva cache local."""
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_tmdb_budget(imdb_id: str, cache: dict) -> dict:
    """Busca budget e revenue de um filme via TMDb API.

    Retorna dict com: tmdb_id, budget, revenue, budget_source, match_confidence.
    Usa cache local para não repetir consultas.
    """
    if imdb_id in cache:
        return cache[imdb_id]

    result = {
        "tmdb_id": None, "budget": None, "revenue": None,
        "budget_source": None, "match_confidence": None,
    }

    try:
        # Passo 1: encontrar o filme no TMDb pelo IMDb ID
        r = requests.get(
            TMDB_FIND_URL.format(imdb_id=imdb_id),
            params={"api_key": TMDB_API_KEY, "external_source": "imdb_id"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

        movies = data.get("movie_results", [])
        if not movies:
            result["budget_source"] = "tmdb_not_found"
            cache[imdb_id] = result
            return result

        if len(movies) > 1:
            result["match_confidence"] = "ambiguous"
        else:
            result["match_confidence"] = "exact"

        tmdb_id = movies[0]["id"]
        result["tmdb_id"] = tmdb_id

        time.sleep(RATE_LIMIT_DELAY)

        # Passo 2: buscar detalhes (budget + revenue)
        r2 = requests.get(
            TMDB_MOVIE_URL.format(tmdb_id=tmdb_id),
            params={"api_key": TMDB_API_KEY},
            timeout=10,
        )
        r2.raise_for_status()
        details = r2.json()

        budget_val = details.get("budget", 0)
        revenue_val = details.get("revenue", 0)

        # TMDb retorna 0 quando não tem dado — tratar como missing
        result["budget"] = budget_val if budget_val > 0 else None
        result["revenue"] = revenue_val if revenue_val > 0 else None
        result["budget_source"] = "tmdb"

        time.sleep(RATE_LIMIT_DELAY)

    except requests.exceptions.RequestException as e:
        logging.warning("Erro HTTP para %s: %s", imdb_id, e)
        result["budget_source"] = f"error: {str(e)[:50]}"

    cache[imdb_id] = result
    return result


# ─── Funções de cálculo ───

def parse_box_office(val) -> float | None:
    """Converte string de box office ($1,234,567) para float."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = str(val).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def compute_roi_metrics(row: pd.Series) -> dict:
    """Calcula métricas de retorno financeiro."""
    budget = row.get("budget")
    box_office = row.get("box_office_clean")

    result = {
        "log_budget": None,
        "roi": None,
        "revenue_multiple": None,
        "log_revenue_multiple": None,
    }

    if pd.notna(budget) and budget > 0:
        result["log_budget"] = np.log1p(budget)

        if pd.notna(box_office) and box_office > 0:
            result["roi"] = (box_office - budget) / budget
            result["revenue_multiple"] = box_office / budget

            if result["revenue_multiple"] > 0:
                result["log_revenue_multiple"] = np.log(result["revenue_multiple"])

    return result


# ─── Pipeline principal ───

def main():
    ensure_dirs()

    # Carregar dataset e metadata
    df = pd.read_csv(FEATURES / "analysis_dataset.csv")
    meta = pd.read_csv(METADATA / "film_list.csv")
    meta = meta[meta["matched"] == True]

    # Adicionar imdb_id ao dataset
    df = df.merge(
        meta[["source_file", "imdb_id"]],
        left_on="title", right_on="source_file", how="left",
    ).drop(columns=["source_file"])

    n_total = len(df)
    n_with_imdb_id = df["imdb_id"].notna().sum()
    logging.info("Dataset: %d filmes, %d com IMDb ID", n_total, n_with_imdb_id)

    # Carregar cache
    cache = load_cache()
    cached_count = sum(1 for iid in df["imdb_id"].dropna() if iid in cache)
    logging.info("Cache: %d filmes já consultados", cached_count)

    # Coletar budgets
    budgets = []
    processed = 0
    for _, row in df.iterrows():
        imdb_id = row.get("imdb_id")
        if pd.isna(imdb_id):
            budgets.append({
                "tmdb_id": None, "budget": None, "revenue": None,
                "budget_source": "no_imdb_id", "match_confidence": None,
            })
            continue

        result = fetch_tmdb_budget(imdb_id, cache)
        budgets.append(result)
        processed += 1

        # Salvar cache periodicamente
        if processed % 50 == 0:
            save_cache(cache)
            found = sum(1 for b in budgets if b.get("budget") is not None)
            logging.info("Progresso: %d/%d processados, %d budgets encontrados",
                         processed, n_with_imdb_id, found)

    # Salvar cache final
    save_cache(cache)

    # Montar colunas de budget
    budget_df = pd.DataFrame(budgets)
    df["tmdb_id"] = budget_df["tmdb_id"].values
    df["budget_raw"] = budget_df["budget"].values
    df["budget"] = budget_df["budget"].values
    df["tmdb_revenue"] = budget_df["revenue"].values
    df["budget_source"] = budget_df["budget_source"].values
    df["match_confidence"] = budget_df["match_confidence"].values

    # Limpar box_office para cálculos
    df["box_office_clean"] = df["box_office"].apply(parse_box_office)

    # Calcular ROI e métricas derivadas
    roi_rows = df.apply(compute_roi_metrics, axis=1, result_type="expand")
    df = pd.concat([df, roi_rows], axis=1)

    # ─── Relatório ───
    n_budget = df["budget"].notna().sum()
    n_bo = df["box_office_clean"].notna().sum()
    n_roi = df["roi"].notna().sum()

    report = []
    report.append("=" * 60)
    report.append("  RELATÓRIO: ORÇAMENTO E ROI")
    report.append("=" * 60)
    report.append(f"\nTotal de filmes:           {n_total}")
    report.append(f"Com IMDb ID:               {n_with_imdb_id}")
    report.append(f"Com box office:            {n_bo}")
    report.append(f"Com budget (TMDb):         {n_budget} ({n_budget/n_total*100:.1f}%)")
    report.append(f"Com ROI calculável:        {n_roi} ({n_roi/n_total*100:.1f}%)")

    report.append(f"\n--- Fontes de budget ---")
    for src, count in df["budget_source"].value_counts().items():
        report.append(f"  {src}: {count}")

    if n_budget > 0:
        b = df["budget"].dropna()
        report.append(f"\n--- Budget (n={n_budget}) ---")
        report.append(f"  min:    ${b.min():,.0f}")
        report.append(f"  Q1:     ${b.quantile(.25):,.0f}")
        report.append(f"  median: ${b.median():,.0f}")
        report.append(f"  Q3:     ${b.quantile(.75):,.0f}")
        report.append(f"  max:    ${b.max():,.0f}")
        report.append(f"  mean:   ${b.mean():,.0f}")

    if n_roi > 0:
        r = df["roi"].dropna()
        report.append(f"\n--- ROI (n={n_roi}) ---")
        report.append(f"  min:    {r.min():.2f}")
        report.append(f"  Q1:     {r.quantile(.25):.2f}")
        report.append(f"  median: {r.median():.2f}")
        report.append(f"  Q3:     {r.quantile(.75):.2f}")
        report.append(f"  max:    {r.max():.2f}")
        report.append(f"  mean:   {r.mean():.2f}")

        rm = df["revenue_multiple"].dropna()
        report.append(f"\n--- Revenue Multiple (n={len(rm)}) ---")
        report.append(f"  min:    {rm.min():.2f}x")
        report.append(f"  median: {rm.median():.2f}x")
        report.append(f"  max:    {rm.max():.2f}x")

    # Exemplos de falha
    failed = df[df["budget"].isna() & df["imdb_id"].notna()].head(10)
    if len(failed) > 0:
        report.append(f"\n--- Exemplos sem budget ({len(df[df['budget'].isna()])} total) ---")
        for _, row in failed.iterrows():
            report.append(f"  {row['title']} ({row.get('year','?')}) — {row['budget_source']}")

    report.append(f"\n--- Recomendações para análise ---")
    report.append("  * ROI é muito assimétrico → usar log(revenue_multiple) como target")
    report.append("  * revenue_multiple < 0 não existe (box_office >= 0)")
    report.append("  * Filtrar budget > $1M para evitar filmes independentes com ROI extremo")
    report.append("  * Cuidado: budget=0 no TMDb significa 'desconhecido', não 'sem custo'")

    report_text = "\n".join(report)
    print(report_text)

    # Salvar
    out_path = FEATURES / "analysis_dataset_with_roi.csv"

    # Colunas finais organizadas
    id_cols = ["title", "imdb_id", "tmdb_id", "year", "genre"]
    target_cols = ["imdb_rating", "box_office", "box_office_clean", "log_box_office",
                   "budget_raw", "budget", "log_budget", "tmdb_revenue",
                   "roi", "revenue_multiple", "log_revenue_multiple",
                   "budget_source", "match_confidence"]
    metric_cols = [c for c in df.columns if c.startswith("gv")]
    final_cols = [c for c in id_cols + target_cols + metric_cols if c in df.columns]
    df[final_cols].to_csv(out_path, index=False)

    (RESULTS / "budget_roi_report.txt").write_text(report_text, encoding="utf-8")

    logging.info("Salvo: %s (%d filmes, %d colunas)", out_path, len(df), len(final_cols))
    logging.info("Relatório: %s", RESULTS / "budget_roi_report.txt")


if __name__ == "__main__":
    main()
