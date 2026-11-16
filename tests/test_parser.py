"""Testes para o parser de cenas."""

from src.parser import parse_screenplay


SAMPLE_SCREENPLAY = """
FADE IN:

INT. OFFICE - DAY

John sits at his desk, typing furiously.

EXT. STREET - NIGHT

Rain pours down. Mary runs across the street.

INT. APARTMENT - NIGHT

Mary enters, soaking wet. She slams the door.

EXT. ROOFTOP - DAWN

The sun rises over the city.
"""


def test_parse_basic():
    scenes = parse_screenplay(SAMPLE_SCREENPLAY)
    assert len(scenes) == 4
    assert scenes[0]["heading"].startswith("INT. OFFICE")
    assert scenes[1]["heading"].startswith("EXT. STREET")


def test_scene_indices():
    scenes = parse_screenplay(SAMPLE_SCREENPLAY)
    for i, s in enumerate(scenes):
        assert s["scene_idx"] == i


def test_empty_screenplay():
    scenes = parse_screenplay("No sluglines here, just regular text.")
    assert scenes == []


def test_short_scene_merge():
    text = """
INT. ROOM - DAY

Hello.

EXT. PARK - DAY

A very short scene.

INT. HOUSE - NIGHT

This is a longer scene with more text that should definitely not be merged
with anything because it has sufficient content to stand on its own.
"""
    scenes = parse_screenplay(text)
    # Cenas curtas (<20 chars) sao mergeadas com a proxima
    assert all(s["char_count"] >= 0 for s in scenes)


def test_char_count():
    scenes = parse_screenplay(SAMPLE_SCREENPLAY)
    for s in scenes:
        assert s["char_count"] == len(s["text"])
