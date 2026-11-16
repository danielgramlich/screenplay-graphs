"""Classificacao de emocao discreta por cena usando NRC Emotion Lexicon."""

import json
import logging

import nltk
import pandas as pd
from nrclex import NRCLex

from src.sentiment import vader_scene
from src.utils import EMOTIONS, EMOTIONS_LIST, SCENES, ensure_dirs

logger = logging.getLogger(__name__)

# NRCLex 4.x depende de wordnet para lematizacao
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def nrc_scene(text: str) -> dict:
    """Classifica emocao discreta de uma cena via NRCLex.

    Returns:
        Dict com scores para cada uma das 6 emocoes + dominant_emotion.
    """
    emotion = NRCLex("init")
    emotion.load_raw_text(text)
    freqs = emotion.affect_frequencies

    scores = {e: freqs.get(e, 0.0) for e in EMOTIONS_LIST}
    total = sum(scores.values())

    if total == 0:
        return {**scores, "dominant_emotion": None}

    dominant = max(scores, key=scores.get)
    return {**scores, "dominant_emotion": dominant}


def classify_scene(text: str) -> dict:
    """Combina VADER e NRC para uma cena. Resolve cenas sem match NRC."""
    vader = vader_scene(text)
    nrc = nrc_scene(text)

    if nrc["dominant_emotion"] is None:
        compound = vader["vader_compound"]
        if compound > 0.05:
            nrc["dominant_emotion"] = "joy"
        elif compound < -0.05:
            nrc["dominant_emotion"] = "sadness"
        else:
            nrc["dominant_emotion"] = "trust"

    return {**vader, **nrc}


def annotate_all() -> dict[str, pd.DataFrame]:
    """Anota emocoes em todas as cenas parseadas. Retorna dict titulo->DataFrame."""
    ensure_dirs()
    results = {}
    scene_files = sorted(SCENES.glob("*.json"))

    if not scene_files:
        logger.warning("Nenhum arquivo de cenas encontrado em %s", SCENES)
        return results

    for path in scene_files:
        title = path.stem
        scenes = json.loads(path.read_text(encoding="utf-8"))

        rows = []
        for scene in scenes:
            annotation = classify_scene(scene["text"])
            rows.append({"scene_idx": scene["scene_idx"], **annotation})

        df = pd.DataFrame(rows)
        out_path = EMOTIONS / f"{title}.csv"
        df.to_csv(out_path, index=False)
        results[title] = df
        logger.info("%s: %d cenas anotadas", title, len(df))

    return results
