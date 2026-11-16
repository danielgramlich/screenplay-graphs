#!/usr/bin/env python3
"""Etapa 27: Coletar metadata (IMDb rating, genre, year) dos novos filmes MovieSum.

Usa os imdb_ids do MovieSum para buscar direto no OMDb API.
Merge com film_list.csv existente.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests
from src.utils import METADATA, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "")
OMDB_URL = "http://www.omdbapi.com/"
RATE_LIMIT = 0.15  # segundos entre chamadas
CACHE_PATH = METADATA / "omdb_cache_expanded.json"


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_omdb(imdb_id, cache):
    """Busca metadata de um filme pelo IMDb ID."""
    if imdb_id in cache:
        return cache[imdb_id]

    try:
        r = requests.get(OMDB_URL, params={"i": imdb_id, "apikey": OMDB_API_KEY}, timeout=10)
        r.raise_for_status()
        data = r.json()

        if data.get("Response") == "True":
            result = {
                "title_omdb": data.get("Title", ""),
                "imdb_id": imdb_id,
                "imdb_rating": data.get("imdbRating", "N/A"),
                "box_office": data.get("BoxOffice", "N/A"),
                "year": data.get("Year", "N/A"),
                "genre": data.get("Genre", ""),
                "matched": True,
            }
        else:
            result = {"imdb_id": imdb_id, "matched": False, "error": data.get("Error", "")}

        cache[imdb_id] = result
        time.sleep(RATE_LIMIT)
        return result

    except Exception as e:
        result = {"imdb_id": imdb_id, "matched": False, "error": str(e)[:50]}
        cache[imdb_id] = result
        return result


def main():
    ensure_dirs()

    # Carregar film_list existente
    film_list_path = METADATA / "film_list.csv"
    existing = pd.read_csv(film_list_path)
    existing_ids = set(existing["imdb_id"].dropna())
    logging.info("Film list existente: %d filmes, %d com imdb_id", len(existing), len(existing_ids))

    # Carregar mapa MovieSum
    imdb_map_path = METADATA / "moviesum_imdb_map.json"
    if not imdb_map_path.exists():
        logging.error("moviesum_imdb_map.json não encontrado")
        return

    imdb_map = json.loads(imdb_map_path.read_text(encoding="utf-8"))
    logging.info("MovieSum map: %d filmes com imdb_id", len(imdb_map))

    # Filtrar: só buscar os que não existem no film_list
    to_fetch = {}
    for filename, imdb_id in imdb_map.items():
        if imdb_id and imdb_id not in existing_ids:
            to_fetch[filename] = imdb_id

    logging.info("Novos a buscar: %d filmes", len(to_fetch))

    if not to_fetch:
        logging.info("Nenhum filme novo para buscar.")
        return

    # Carregar cache
    cache = load_cache()
    cached = sum(1 for iid in to_fetch.values() if iid in cache)
    logging.info("Já no cache: %d", cached)

    # Buscar
    rows = []
    success = 0
    fail = 0

    for i, (filename, imdb_id) in enumerate(to_fetch.items()):
        result = fetch_omdb(imdb_id, cache)

        if result.get("matched"):
            # Converter rating
            rating_str = result.get("imdb_rating", "N/A")
            try:
                rating = float(rating_str)
            except (ValueError, TypeError):
                rating = None

            year_str = result.get("year", "N/A")
            try:
                year = float(year_str[:4])
            except (ValueError, TypeError):
                year = None

            rows.append({
                "source_file": filename,
                "title_query": result.get("title_omdb", filename),
                "imdb_id": imdb_id,
                "imdb_rating": rating,
                "box_office": result.get("box_office", "N/A"),
                "year": year,
                "genre": result.get("genre", ""),
                "matched": True,
            })
            success += 1
        else:
            rows.append({
                "source_file": filename,
                "title_query": filename,
                "imdb_id": imdb_id,
                "imdb_rating": None,
                "box_office": "N/A",
                "year": None,
                "genre": "",
                "matched": False,
            })
            fail += 1

        if (i + 1) % 100 == 0:
            save_cache(cache)
            logging.info("Progresso: %d/%d (sucesso=%d, falha=%d)",
                         i + 1, len(to_fetch), success, fail)

    save_cache(cache)

    # Merge com existente
    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Deduplicar por source_file (manter primeiro)
    combined = combined.drop_duplicates(subset="source_file", keep="first")

    # Salvar
    combined.to_csv(film_list_path, index=False)

    logging.info("=" * 60)
    logging.info("  RESULTADO DA COLETA DE METADATA")
    logging.info("=" * 60)
    logging.info("  Novos buscados: %d", len(to_fetch))
    logging.info("  Sucesso: %d", success)
    logging.info("  Falha: %d", fail)
    logging.info("  Film list total: %d filmes", len(combined))
    logging.info("  Com IMDb rating: %d", combined["imdb_rating"].notna().sum())
    logging.info("  Matched: %d", (combined["matched"] == True).sum())
    logging.info("  Salvo: %s", film_list_path)


if __name__ == "__main__":
    main()
