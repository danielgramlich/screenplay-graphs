"""Download de roteiros do HuggingFace (mattismegevand/IMSDb) com fallback IMSDb."""

import logging
import time

import requests
from bs4 import BeautifulSoup

from src.utils import RAW, ensure_dirs, sanitize_title

logger = logging.getLogger(__name__)


def load_from_huggingface(max_scripts: int | None = None) -> list[dict]:
    """Carrega roteiros do dataset HuggingFace mattismegevand/IMSDb.

    Baixa apenas o data.jsonl (sem posters/imagens).

    Returns:
        Lista de dicts com keys: title, script, genres.
    """
    import json
    from huggingface_hub import hf_hub_download

    jsonl_path = hf_hub_download(
        repo_id="mattismegevand/IMSDb",
        filename="data.jsonl",
        repo_type="dataset",
    )

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_scripts and i >= max_scripts:
                break
            row = json.loads(line)
            script_text = row.get("script", "") or ""
            if len(script_text.strip()) < 500:
                continue
            records.append({
                "title": row.get("title", f"unknown_{i}"),
                "script": script_text,
                "genres": row.get("genres", ""),
            })
    logger.info("Carregados %d roteiros do HuggingFace", len(records))
    return records


def save_scripts(records: list[dict]) -> list[str]:
    """Salva roteiros como .txt em data/raw/. Retorna lista de filenames salvos."""
    ensure_dirs()
    saved = []
    for rec in records:
        fname = sanitize_title(rec["title"]) + ".txt"
        path = RAW / fname
        path.write_text(rec["script"], encoding="utf-8")
        saved.append(fname)
    logger.info("Salvos %d roteiros em %s", len(saved), RAW)
    return saved


def scrape_imsdb_fallback(title: str) -> str | None:
    """Fallback: baixa roteiro diretamente do IMSDb. Retorna texto ou None."""
    url = f"https://imsdb.com/scripts/{title.replace(' ', '-')}.html"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        pre = soup.find("pre")
        if pre:
            return pre.get_text()
    except requests.RequestException as e:
        logger.warning("Falha ao baixar %s: %s", title, e)
    time.sleep(1)
    return None
