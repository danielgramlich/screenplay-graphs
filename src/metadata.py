"""Coleta de metadados IMDb/box office via OMDb API."""

import difflib
import logging
import os
import time

import pandas as pd
import requests

from src.utils import METADATA, RAW, ensure_dirs

logger = logging.getLogger(__name__)

OMDB_URL = "http://www.omdbapi.com/"


def fetch_omdb(title: str, api_key: str) -> dict | None:
    """Busca metadados de um filme na OMDb API.

    Returns:
        Dict com imdb_rating, imdb_id, budget, revenue, ou None se nao encontrado.
    """
    params = {"t": title, "apikey": api_key, "type": "movie"}
    try:
        resp = requests.get(OMDB_URL, params=params, timeout=10)
        data = resp.json()
        if data.get("Response") == "False":
            return None

        imdb_rating = data.get("imdbRating", "N/A")
        box_office = data.get("BoxOffice", "N/A")

        return {
            "imdb_id": data.get("imdbID", ""),
            "imdb_rating": float(imdb_rating) if imdb_rating != "N/A" else None,
            "box_office": box_office,
            "year": data.get("Year", ""),
            "genre": data.get("Genre", ""),
        }
    except (requests.RequestException, ValueError) as e:
        logger.warning("Erro ao buscar %s: %s", title, e)
        return None


def normalize_title(title: str) -> str:
    """Normaliza titulo para matching."""
    import re
    t = title.lower().strip()
    t = re.sub(r"^(the|a|an)\s+", "", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t.strip()


def _title_variants(title: str) -> list[str]:
    """Gera variantes de titulo para melhorar match na OMDb.

    Ex: "World Is Not Enough The" -> ["World Is Not Enough The", "The World Is Not Enough"]
    """
    import re
    variants = [title]

    # Mover artigo do final para o inicio: "Godfather The" -> "The Godfather"
    match = re.match(r"^(.+?)\s+(The|A|An)$", title, re.IGNORECASE)
    if match:
        variants.append(f"{match.group(2)} {match.group(1)}")

    # Remover artigo do final se presente
    stripped = re.sub(r"\s+(The|A|An)$", "", title, flags=re.IGNORECASE)
    if stripped != title:
        variants.append(stripped)

    return variants


def collect_metadata(api_key: str | None = None) -> pd.DataFrame:
    """Coleta metadados para todos os roteiros em data/raw/.

    Args:
        api_key: Chave da OMDb API. Se None, tenta ler de OMDB_API_KEY env var.

    Returns:
        DataFrame com titulo, imdb_id, imdb_rating, box_office, etc.
    """
    ensure_dirs()
    api_key = api_key or os.environ.get("OMDB_API_KEY")
    if not api_key:
        logger.error(
            "OMDb API key nao encontrada. Defina OMDB_API_KEY ou passe como argumento. "
            "Obtenha uma key gratuita em https://www.omdbapi.com/apikey.aspx"
        )
        return pd.DataFrame()

    raw_files = sorted(RAW.glob("*.txt"))
    rows = []

    for path in raw_files:
        title = path.stem.replace("-", " ").title()
        # Gerar variantes de titulo para tentar match
        variants = _title_variants(title)
        result = None
        used_title = title
        for variant in variants:
            result = fetch_omdb(variant, api_key)
            if result is not None:
                used_title = variant
                break
            time.sleep(0.3)

        if result is None:
            logger.warning("Nao encontrado na OMDb: %s", title)
            rows.append({
                "source_file": path.stem,
                "title_query": title,
                "imdb_id": "",
                "imdb_rating": None,
                "box_office": "",
                "year": "",
                "genre": "",
                "matched": False,
            })
        else:
            rows.append({
                "source_file": path.stem,
                "title_query": used_title,
                **result,
                "matched": True,
            })
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    out_path = METADATA / "film_list.csv"
    df.to_csv(out_path, index=False)

    matched = df["matched"].sum()
    logger.info("Metadados coletados: %d/%d filmes matched", matched, len(df))
    return df
