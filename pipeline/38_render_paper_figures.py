#!/usr/bin/env python3
"""Etapa 38: Renderiza Fig.~2 (G^per) e Fig.~3 (G^emo, K=3) do artigo SBPO 2026.

Filme exemplo: panic-room (Thriller, 12 personagens, MEG protagonista).
Saída: PDFs vetoriais em sbpo2026/figs/.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from src.utils import RAW
from src.characters.sbsi_core import (
    ExtractionConfig, discover_and_validate_characters,
    split_script_into_scenes, extract_speaker_turns,
)
from src.characters.methods import graphs_turn_adjacency_per_scene
from src.sentiment_roberta import roberta_scenes_batch

TITLE = "panic-room"
OUT = ROOT / "sbpo2026" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = ExtractionConfig()


def to_v3(c):
    if c > 0.05:
        return "pos"
    if c < -0.05:
        return "neg"
    return "neu"


def to_v5(c):
    if c >= 0.30:
        return "++"
    if c >= 0.05:
        return "+"
    if c <= -0.30:
        return "--"
    if c <= -0.05:
        return "-"
    return "0"


def build_g_per(text):
    """Reconstrói G^per agregado para um filme."""
    name_map, canonical, _ = discover_and_validate_characters(text, CONFIG)
    scenes = split_script_into_scenes(text)
    speakers_per_scene = []
    for sc in scenes:
        sp, _ = extract_speaker_turns(sc.body, name_map, canonical, CONFIG)
        speakers_per_scene.append(sp)
    scene_graphs = graphs_turn_adjacency_per_scene(speakers_per_scene)
    film_G = nx.Graph()
    for G in scene_graphs:
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1)
            if film_G.has_edge(u, v):
                film_G[u][v]["weight"] += w
            else:
                film_G.add_edge(u, v, weight=w)
    return film_G, scenes


def build_g_emo(scenes, K=3):
    """Reconstrói G^emo (K=3 ou K=5) a partir de RoBERTa por cena."""
    texts = [getattr(sc, "body", "") or getattr(sc, "text", "") for sc in scenes]
    results = roberta_scenes_batch(texts)
    compounds = [r["roberta_compound"] for r in results]
    if K == 3:
        states = ["neg", "neu", "pos"]
        seq = [to_v3(c) for c in compounds]
    else:
        states = ["--", "-", "0", "+", "++"]
        seq = [to_v5(c) for c in compounds]
    n = len(states)
    T = np.zeros((n, n))
    idx = {s: i for i, s in enumerate(states)}
    for a, b in zip(seq[:-1], seq[1:]):
        T[idx[a]][idx[b]] += 1
    G = nx.DiGraph()
    for s in states:
        G.add_node(s)
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if T[i][j] > 0:
                G.add_edge(a, b, weight=int(T[i][j]))
    return G, T, states


# ──────────────────────────────────────────────────────────────────
# Fig. 2 — G^per
# ──────────────────────────────────────────────────────────────────

def render_g_per(G, out_path):
    if G.number_of_nodes() == 0:
        raise SystemExit("G^per vazio")

    strengths = dict(G.degree(weight="weight"))
    s_max = max(strengths.values())
    edge_w = [d["weight"] for _, _, d in G.edges(data=True)]
    w_max = max(edge_w) if edge_w else 1.0

    # spring_layout com seed dá distribuição mais espalhada que kamada_kawai
    # para grafos densos como G^per de Panic Room.
    pos = nx.spring_layout(G, weight="weight", k=2.4, iterations=600, seed=11)

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=300)

    # Edges - espessura proporcional ao peso, alpha sempre alto
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        ratio = w / w_max
        lw = 1.2 + 11.0 * ratio
        alpha = 0.65 + 0.30 * ratio
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="0.25",
                linewidth=lw, alpha=alpha, zorder=1, solid_capstyle="round")

    # Nodes
    sizes = {n: 200 + 1000 * (strengths[n] / s_max) for n in G.nodes()}
    colors = ["#c0392b" if n == "MEG" else "#2c3e50" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=[sizes[n] for n in G.nodes()],
                           node_color=colors, edgecolors="white",
                           linewidths=1.2, ax=ax)

    # Labels com offset radial (cada label é empurrado para fora ao longo
    # do vetor centróide→nó, evitando sobreposição com os nós vizinhos).
    cx = float(np.mean([p[0] for p in pos.values()]))
    cy = float(np.mean([p[1] for p in pos.values()]))
    for n, (x, y) in pos.items():
        dx, dy = x - cx, y - cy
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / norm, dy / norm
        # raio visual aproximado do nó em coordenadas de dados
        node_r = 0.022 + 0.052 * (sizes[n] / 1200) ** 0.5
        lx = x + ux * (node_r + 0.045)
        ly = y + uy * (node_r + 0.045)
        # Alinhamento conforme quadrante para o texto ficar "fora" do nó
        ha = "left" if ux > 0.15 else ("right" if ux < -0.15 else "center")
        va = "bottom" if uy > 0.15 else ("top" if uy < -0.15 else "center")
        ax.text(lx, ly, n.title(), ha=ha, va=va,
                fontsize=9.0, fontfamily="serif", color="0.12",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec="none", alpha=0.75))

    ax.set_axis_off()
    ax.margins(0.22)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# Fig. 3 — G^emo (K=3)
# ──────────────────────────────────────────────────────────────────

def render_g_emo(G, T, out_path, states=None):
    if states is None:
        states = ["neg", "neu", "pos"]
    if len(states) == 3:
        labels = {"neg": "Neg.", "neu": "Neu.", "pos": "Pos."}
        colors = {"neg": "#c0392b", "neu": "#7f8c8d", "pos": "#27ae60"}
        pos = {"neg": (-1.6, 0.0), "neu": (0.0, 0.0), "pos": (1.6, 0.0)}
        node_size = 2600
        figsize = (6.2, 2.4)
        node_r = 0.30  # raio visual aproximado em coordenadas de dados
    else:
        labels = {"--": "−−", "-": "−", "0": "0", "+": "+", "++": "++"}
        colors = {"--": "#922b21", "-": "#d98880",
                  "0": "#7f8c8d",
                  "+": "#7dcea0", "++": "#196f3d"}
        pos = {s: (-2.6 + 1.3 * i, 0.0) for i, s in enumerate(states)}
        node_size = 2000
        figsize = (7.0, 2.4)
        node_r = 0.26

    w_max = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # ── Setas curvas entre estados distintos ──
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        w = d["weight"]
        x0, y0 = pos[u]; x1, y1 = pos[v]
        dist = abs(states.index(u) - states.index(v))
        rad_mag = 0.22 + 0.08 * (dist - 1)
        # Convenção: u→v à direita (ix(u)<ix(v)) curva para cima; à esquerda, para baixo.
        rad = rad_mag if (states.index(u) < states.index(v)) else -rad_mag
        lw = 1.0 + 4.0 * (w / w_max)
        # shrink em pontos (units do mutation_scale); aproxima o raio visual do nó
        shrink = 22 if len(states) == 3 else 18
        arrow = mpatches.FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=14,
            shrinkA=shrink, shrinkB=shrink,
            linewidth=lw, color="0.30", alpha=0.85, zorder=1,
        )
        ax.add_patch(arrow)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + (0.22 if rad > 0 else -0.22)
        ax.text(mx, my, str(int(w)), ha="center", va="center",
                fontsize=9.0, fontfamily="serif", color="0.15",
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="none", alpha=0.92), zorder=2)

    # ── Autoloops: badge colorido acima do nó (legenda explica) ──
    for u, v, d in G.edges(data=True):
        if u != v:
            continue
        w = d["weight"]
        x, y = pos[u]
        ax.text(x, y + node_r + 0.18, str(int(w)),
                ha="center", va="center",
                fontsize=10.5, fontfamily="serif",
                color="white", fontweight="bold", zorder=3,
                bbox=dict(boxstyle="round,pad=0.32",
                          fc=colors[u], ec="none", alpha=0.95))

    # ── Nós (por cima das arestas) ──
    for s in states:
        x, y = pos[s]
        ax.scatter([x], [y], s=node_size, c=colors[s], edgecolors="white",
                   linewidths=1.5, zorder=3)
        # fontsize reduzido garante que o texto cabe dentro do círculo
        fs = 10 if len(states) == 3 else 11
        ax.text(x, y, labels[s], ha="center", va="center",
                fontsize=fs, fontfamily="serif", color="white",
                fontweight="bold", zorder=4)

    xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(-0.9, 1.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    raw_path = RAW / f"{TITLE}.txt"
    text = raw_path.read_text(encoding="utf-8", errors="replace")

    print(f"Construindo G^per de {TITLE}...")
    G_per, scenes = build_g_per(text)
    print(f"  V={G_per.number_of_nodes()}, E={G_per.number_of_edges()}, "
          f"|cenas|={len(scenes)}")
    render_g_per(G_per, OUT / "fig_gper.pdf")
    print(f"  ✓ {OUT / 'fig_gper.pdf'}")

    print(f"Construindo G^emo (K=3) de {TITLE}...")
    G3, T3, st3 = build_g_emo(scenes, K=3)
    print(f"  V={G3.number_of_nodes()}, E={G3.number_of_edges()}")
    print(f"  Matriz de transição:\n{T3.astype(int)}")
    render_g_emo(G3, T3, OUT / "fig_gemo.pdf", states=st3)
    print(f"  ✓ {OUT / 'fig_gemo.pdf'}")

    print(f"Construindo G^emo (K=5) de {TITLE}...")
    G5, T5, st5 = build_g_emo(scenes, K=5)
    print(f"  V={G5.number_of_nodes()}, E={G5.number_of_edges()}")
    print(f"  Matriz de transição:\n{T5.astype(int)}")
    render_g_emo(G5, T5, OUT / "fig_gemo_k5.pdf", states=st5)
    print(f"  ✓ {OUT / 'fig_gemo_k5.pdf'}")


if __name__ == "__main__":
    main()
