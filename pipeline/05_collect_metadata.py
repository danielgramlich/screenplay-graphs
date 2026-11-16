#!/usr/bin/env python3
"""Etapa 6: Coleta de metadados IMDb/box office via OMDb API."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metadata import collect_metadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    df = collect_metadata()
    if df.empty:
        print("Nenhum metadado coletado. Verifique a OMDB_API_KEY.")
        return
    matched = df["matched"].sum()
    print(f"\nResumo da coleta de metadados:")
    print(f"  Total: {len(df)} filmes")
    print(f"  Matched: {matched}")
    print(f"  Sem match: {len(df) - matched}")


if __name__ == "__main__":
    main()
