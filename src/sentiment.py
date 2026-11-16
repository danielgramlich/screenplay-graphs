"""Analise de sentimento VADER por cena."""

import logging

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

_analyzer = SentimentIntensityAnalyzer()


MAX_SCENE_CHARS = 2000


def vader_scene(text: str) -> dict:
    """Calcula sentimento VADER para uma cena.

    Aplica clipping a 2000 caracteres para evitar diluicao em cenas longas.
    Divide em sentencas, pontua cada uma, e faz media ponderada por tamanho.

    Returns:
        Dict com vader_compound, vader_pos, vader_neg, vader_neu.
    """
    text = text[:MAX_SCENE_CHARS]
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return {"vader_compound": 0.0, "vader_pos": 0.0, "vader_neg": 0.0, "vader_neu": 1.0}

    total_weight = 0
    weighted = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}

    for sent in sentences:
        weight = len(sent)
        scores = _analyzer.polarity_scores(sent)
        for key in weighted:
            weighted[key] += scores[key] * weight
        total_weight += weight

    if total_weight == 0:
        return {"vader_compound": 0.0, "vader_pos": 0.0, "vader_neg": 0.0, "vader_neu": 1.0}

    return {
        "vader_compound": weighted["compound"] / total_weight,
        "vader_pos": weighted["pos"] / total_weight,
        "vader_neg": weighted["neg"] / total_weight,
        "vader_neu": weighted["neu"] / total_weight,
    }
