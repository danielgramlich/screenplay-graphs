#!/usr/bin/env python3
"""Etapa 2: Segmentacao de roteiros em cenas."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parser import parse_all_screenplays

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    results = parse_all_screenplays()
    print(f"\nResumo do parsing:")
    print(f"  Roteiros parseados: {len(results)}")
    for title, scenes in sorted(results.items()):
        print(f"  {title}: {len(scenes)} cenas")


if __name__ == "__main__":
    main()
