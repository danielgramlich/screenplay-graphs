#!/usr/bin/env python3
"""Etapa 8: Validacao — compara anotacoes manuais com pipeline automatico."""

import logging
import sys
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import EMOTIONS, VALIDATION

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    annotations_path = VALIDATION / "manual_annotations.csv"
    if not annotations_path.exists():
        print("Arquivo manual_annotations.csv nao encontrado.")
        print("Crie-o com colunas: title, scene_idx, human_valence, human_emotion")
        print(f"Caminho esperado: {annotations_path}")
        return

    manual = pd.read_csv(annotations_path)
    required = {"title", "scene_idx", "human_valence", "human_emotion"}
    if not required.issubset(manual.columns):
        print(f"Colunas faltando: {required - set(manual.columns)}")
        return

    # Juntar com anotacoes automaticas
    results = []
    for title in manual["title"].unique():
        auto_path = EMOTIONS / f"{title}.csv"
        if not auto_path.exists():
            print(f"  Aviso: emocoes automaticas nao encontradas para '{title}'")
            continue
        auto = pd.read_csv(auto_path)
        merged = manual[manual["title"] == title].merge(auto, on="scene_idx", how="inner")
        results.append(merged)

    if not results:
        print("Nenhum match entre anotacoes manuais e automaticas.")
        return

    df = pd.concat(results, ignore_index=True)

    # Pearson: valencia humana vs VADER
    valid_valence = df[["human_valence", "vader_compound"]].dropna()
    if len(valid_valence) >= 5:
        r, p = stats.pearsonr(valid_valence["human_valence"], valid_valence["vader_compound"])
        print(f"\nValencia — Pearson r: {r:.3f} (p={p:.4f})")
        print(f"  Alvo: r > 0.6 → {'PASSOU' if r > 0.6 else 'ABAIXO DO ALVO'}")

    # Cohen's kappa: emocao humana vs NRC dominante
    valid_emotion = df[["human_emotion", "dominant_emotion"]].dropna()
    if len(valid_emotion) >= 5:
        kappa = cohen_kappa_score(valid_emotion["human_emotion"], valid_emotion["dominant_emotion"])
        print(f"\nEmocao — Cohen's kappa: {kappa:.3f}")
        print(f"  Alvo: kappa > 0.4 → {'PASSOU' if kappa > 0.4 else 'ABAIXO DO ALVO'}")

    print(f"\nTotal de cenas validadas: {len(df)}")


if __name__ == "__main__":
    main()
