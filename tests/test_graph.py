"""Testes para construcao do grafo G2."""

import numpy as np

from src.graph import build_transition_matrix, build_digraph, compute_metrics
from src.utils import EMOTIONS_LIST


def test_transition_matrix_basic():
    emotions = ["joy", "joy", "sadness", "anger", "joy"]
    T = build_transition_matrix(emotions)
    assert T.shape == (6, 6)
    # joy -> joy: 1, joy -> sadness: 1, sadness -> anger: 1, anger -> joy: 1
    idx = {e: i for i, e in enumerate(EMOTIONS_LIST)}
    assert T[idx["joy"]][idx["joy"]] == 1
    assert T[idx["joy"]][idx["sadness"]] == 1
    assert T[idx["sadness"]][idx["anger"]] == 1
    assert T[idx["anger"]][idx["joy"]] == 1
    assert T.sum() == 4


def test_transition_matrix_empty():
    T = build_transition_matrix([])
    assert T.sum() == 0


def test_digraph_construction():
    emotions = ["joy", "sadness", "anger", "fear", "trust", "surprise"]
    T = build_transition_matrix(emotions)
    G = build_digraph(T)
    assert len(G.nodes) == 6
    assert G.number_of_edges() == 5  # 5 transicoes sequenciais unicas


def test_metrics_not_empty():
    emotions = ["joy", "sadness", "anger", "fear", "trust", "joy", "sadness"]
    T = build_transition_matrix(emotions)
    G = build_digraph(T)
    m = compute_metrics(T, G)
    assert m["entropy"] > 0
    assert 0 <= m["autoloop_ratio"] <= 1
    assert 0 <= m["polarization_index"] <= 1


def test_metrics_all_same_emotion():
    emotions = ["joy"] * 10
    T = build_transition_matrix(emotions)
    G = build_digraph(T)
    m = compute_metrics(T, G)
    assert m["autoloop_ratio"] == 1.0
    assert m["polarization_index"] == 0.0
