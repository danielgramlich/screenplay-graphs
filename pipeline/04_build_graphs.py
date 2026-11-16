#!/usr/bin/env python3
"""Etapa 4: Construcao do Grafo de Transicao Emocional (G2) por filme."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import build_all_graphs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    results = build_all_graphs()
    print(f"\nResumo da construcao de grafos:")
    print(f"  Grafos construidos: {len(results)}")
    for title, data in sorted(results.items()):
        m = data["metrics"]
        print(
            f"  {title}: entropy={m['entropy']:.2f}, "
            f"autoloop={m['autoloop_ratio']:.2f}, "
            f"polarization={m['polarization_index']:.2f}"
        )


if __name__ == "__main__":
    main()
