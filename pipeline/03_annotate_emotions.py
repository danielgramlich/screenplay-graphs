#!/usr/bin/env python3
"""Etapa 3: Anotacao emocional (VADER + NRC) por cena."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.emotions import annotate_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    results = annotate_all()
    print(f"\nResumo da anotacao emocional:")
    print(f"  Filmes anotados: {len(results)}")
    for title, df in sorted(results.items()):
        dist = df["dominant_emotion"].value_counts().to_dict()
        print(f"  {title}: {len(df)} cenas — {dist}")


if __name__ == "__main__":
    main()
