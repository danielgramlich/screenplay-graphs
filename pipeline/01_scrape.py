#!/usr/bin/env python3
"""Etapa 1: Download de roteiros do HuggingFace IMSDb dataset."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scraper import load_from_huggingface, save_scripts

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    records = load_from_huggingface()
    saved = save_scripts(records)
    print(f"Total de roteiros salvos: {len(saved)}")


if __name__ == "__main__":
    main()
