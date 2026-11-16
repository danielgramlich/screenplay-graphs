"""Análise de sentimento por cena usando RoBERTa.

Versão otimizada com score contínuo baseado nas probabilidades das 3 classes.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

_pipeline = None
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_SCENE_CHARS = 2000
BATCH_SIZE = 64


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        # top_k=None retorna probabilidades de TODAS as classes, não só a top
        _pipeline = pipeline("sentiment-analysis", model=MODEL_NAME,
                              truncation=True, max_length=512,
                              batch_size=BATCH_SIZE, device=-1,
                              top_k=None)
        logger.info("RoBERTa loaded (batch=%d, top_k=all)", BATCH_SIZE)
    return _pipeline


def _probs_to_compound(result_list):
    """Converte probabilidades das 3 classes em score contínuo [-1, +1].

    Fórmula: compound = P(positive) - P(negative)
    Isso garante score contínuo mesmo quando 'neutral' domina.
    Exemplos:
      P(pos)=0.1, P(neu)=0.8, P(neg)=0.1 → compound = 0.0  (neutro balanceado)
      P(pos)=0.2, P(neu)=0.7, P(neg)=0.1 → compound = +0.1 (levemente positivo)
      P(pos)=0.05, P(neu)=0.6, P(neg)=0.35 → compound = -0.3 (negativo)
    """
    probs = {r["label"]: r["score"] for r in result_list}
    p_pos = probs.get("positive", 0)
    p_neg = probs.get("negative", 0)
    p_neu = probs.get("neutral", 0)

    compound = p_pos - p_neg  # range [-1, +1]

    # Label dominante
    dominant = max(probs, key=probs.get)
    confidence = probs[dominant]

    return compound, dominant, confidence


def roberta_scene(text: str) -> dict:
    """Sentimento de uma cena — score contínuo baseado em probabilidades."""
    pipe = _get_pipeline()
    text = text[:MAX_SCENE_CHARS].strip()
    if len(text) < 5:
        return {"roberta_compound": 0.0, "roberta_label": "neutral", "roberta_confidence": 0.0}
    try:
        result = pipe(text)[0]  # lista de {label, score} para cada classe
        compound, label, conf = _probs_to_compound(result)
        return {"roberta_compound": round(compound, 4), "roberta_label": label,
                "roberta_confidence": round(conf, 4)}
    except Exception:
        return {"roberta_compound": 0.0, "roberta_label": "neutral", "roberta_confidence": 0.0}


def roberta_scenes_batch(texts: list[str]) -> list[dict]:
    """Processa múltiplas cenas num único batch — score contínuo."""
    pipe = _get_pipeline()

    clean = []
    valid_mask = []
    for t in texts:
        t = t[:MAX_SCENE_CHARS].strip()
        if len(t) >= 5:
            clean.append(t)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    if not clean:
        return [{"roberta_compound": 0.0, "roberta_label": "neutral", "roberta_confidence": 0.0}
                for _ in texts]

    try:
        # Com top_k=None, cada resultado é uma lista de 3 dicts
        all_results = pipe(clean)
    except Exception as e:
        logger.warning("Batch failed: %s — fallback individual", e)
        all_results = []
        for t in clean:
            try:
                all_results.append(pipe(t)[0])
            except Exception:
                all_results.append([{"label": "neutral", "score": 1.0}])

    outputs = []
    r_idx = 0
    for is_valid in valid_mask:
        if is_valid:
            result = all_results[r_idx]
            compound, label, conf = _probs_to_compound(result)
            outputs.append({"roberta_compound": round(compound, 4),
                             "roberta_label": label,
                             "roberta_confidence": round(conf, 4)})
            r_idx += 1
        else:
            outputs.append({"roberta_compound": 0.0, "roberta_label": "neutral",
                             "roberta_confidence": 0.0})

    return outputs
