"""Testes para classificacao emocional."""

from src.sentiment import vader_scene
from src.emotions import nrc_scene, classify_scene


def test_vader_positive():
    result = vader_scene("This is wonderful! I love it so much!")
    assert result["vader_compound"] > 0


def test_vader_negative():
    result = vader_scene("This is terrible. I hate everything about it.")
    assert result["vader_compound"] < 0


def test_vader_empty():
    result = vader_scene("")
    assert result["vader_compound"] == 0.0


def test_nrc_returns_emotions():
    result = nrc_scene("I am so happy and joyful today!")
    assert "dominant_emotion" in result
    assert "joy" in result


def test_classify_fallback():
    # Texto sem palavras no NRC mas com sentimento claro
    result = classify_scene("Awesome cool great fantastic!")
    assert result["dominant_emotion"] is not None


def test_classify_negative_fallback():
    result = classify_scene("Bad terrible awful horrible!")
    assert result["dominant_emotion"] is not None
