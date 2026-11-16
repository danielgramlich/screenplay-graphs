#!/usr/bin/env python3
"""Etapa 34: Gerar relatório HTML — mapa visual da dissertação.

Processa 7 filmes (1 por gênero) e gera um relatório HTML completo
com gráficos, explicações e achados da pesquisa.
"""

import sys
import re
import json
import base64
import logging
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "screenpredictor"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy.stats import pearsonr

from src.utils import RAW, RESULTS, INTERMEDIATE, ensure_dirs
from core.parser import parse_screenplay
from core.sentiment import vader_scene
from core.features import to_v5, to_v3, graph_metrics
from src.sentiment_roberta import roberta_scenes_batch
from src.characters.sbsi_core import (
    ExtractionConfig, discover_and_validate_characters,
    split_script_into_scenes, extract_speaker_turns,
)
from src.characters.methods import graphs_turn_adjacency_per_scene

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Filmes selecionados
FILMS = {
    "Thriller": "fight-club",
    "Horror": "psycho",
    "Comedy": "dead-poets-society",
    "Drama": "pulp-fiction",
    "Action": "inception",
    "Crime": "godfather-part-ii",
    "Romance": "good-will-hunting",
}

# Paleta
COLORS = {
    "bg": "#ffffff", "card": "#f6f8fa", "border": "#e1e4e8",
    "text": "#24292e", "muted": "#586069", "dim": "#959da5",
    "accent": "#0366d6", "green": "#28a745", "red": "#d73a49",
    "yellow": "#b08800", "purple": "#6f42c1",
    "v5_high": "#27ae60", "v5_pos": "#28a745", "v5_neu": "#959da5",
    "v5_neg": "#d73a49", "v5_low": "#c0392b",
}


def fig_to_base64(fig):
    """Converte matplotlib figure para base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def process_film(title):
    """Processa um filme completo e retorna todos os dados."""
    path = RAW / f"{title}.txt"
    if not path.exists():
        logging.warning("Não encontrado: %s", path)
        return None

    text = path.read_text(encoding="utf-8", errors="replace")
    scenes = parse_screenplay(text)
    n = len(scenes)
    if n < 6:
        return None

    logging.info("  Processando %s (%d cenas)...", title, n)

    # VADER
    compounds_v = np.array([vader_scene(s["text"])["vader_compound"] for s in scenes])

    # RoBERTa
    logging.info("    RoBERTa...")
    try:
        rob_results = roberta_scenes_batch([s["text"] for s in scenes])
        compounds_r = np.array([r["roberta_compound"] for r in rob_results])
    except Exception:
        compounds_r = np.zeros(n)

    # Discretização + Grafos
    v5_states = [to_v5(c) for c in compounds_v]
    v5_names = ["v5_high", "v5_pos", "v5_neu", "v5_neg", "v5_low"]
    T5 = np.zeros((5, 5))
    for i in range(len(v5_states) - 1):
        T5[v5_names.index(v5_states[i])][v5_names.index(v5_states[i + 1])] += 1
    gv5 = graph_metrics(v5_states, v5_names, "gv5")

    # RoBERTa grafos
    v5r_states = [to_v5(c) for c in compounds_r]
    T5r = np.zeros((5, 5))
    for i in range(len(v5r_states) - 1):
        T5r[v5_names.index(v5r_states[i])][v5_names.index(v5r_states[i + 1])] += 1
    gv5r = graph_metrics(v5r_states, v5_names, "gv5")

    # V3 último terço
    lt = 2 * n // 3
    v3_names = ["pos", "neu", "neg"]

    # Personagens
    logging.info("    Personagens...")
    char_data = {"graph": None, "protagonist": "", "protag_v": [], "protag_r": []}
    try:
        config = ExtractionConfig()
        name_mapping, canonical_set, _ = discover_and_validate_characters(text, config)
        if len(canonical_set) >= 2:
            char_scenes = split_script_into_scenes(text)
            speakers_per_scene = []
            for cs in char_scenes:
                speakers, _ = extract_speaker_turns(cs.body, name_mapping, canonical_set, config)
                speakers_per_scene.append(speakers)

            scene_graphs = graphs_turn_adjacency_per_scene(speakers_per_scene)
            film_G = nx.Graph()
            for G in scene_graphs:
                for u, v, d in G.edges(data=True):
                    w = d.get("weight", 1)
                    if film_G.has_edge(u, v):
                        film_G[u][v]["weight"] += w
                    else:
                        film_G.add_edge(u, v, weight=w)

            char_data["graph"] = film_G
            if film_G.number_of_nodes() >= 2:
                strengths = {nd: sum(dd["weight"] for _, _, dd in film_G.edges(nd, data=True))
                             for nd in film_G.nodes()}
                protag = max(strengths, key=strengths.get)
                char_data["protagonist"] = protag

                # Arco do protagonista
                pv = []
                pr_texts = []
                for cs in char_scenes:
                    current = None
                    pt = []
                    for line in cs.body.split("\n"):
                        s = line.strip()
                        if not s:
                            continue
                        cl = re.sub(r"\(.*?\)", "", s).strip().upper()
                        if cl in name_mapping:
                            current = name_mapping.get(cl, cl)
                        elif current == protag and not (s.startswith("(") and s.endswith(")")):
                            pt.append(s)
                    if pt:
                        full = " ".join(pt)
                        pv.append(vader_scene(full)["vader_compound"])
                        pr_texts.append(full)

                char_data["protag_v"] = pv
                if pr_texts:
                    try:
                        rr = roberta_scenes_batch(pr_texts)
                        char_data["protag_r"] = [r["roberta_compound"] for r in rr]
                    except Exception:
                        pass
    except Exception as e:
        logging.warning("    Erro personagens: %s", e)

    return {
        "title": title, "n": n, "scenes": scenes,
        "compounds_v": compounds_v, "compounds_r": compounds_r,
        "v5_states": v5_states, "v5r_states": v5r_states,
        "T5": T5, "T5r": T5r, "gv5": gv5, "gv5r": gv5r,
        "v5_names": v5_names, "lt": lt,
        "characters": char_data,
    }


# ─── GRÁFICOS ───

def plot_sentiment(data):
    """Gráfico 1. VADER<span id="vader-detail"></span> vs RoBERTa sobrepostos."""
    fig, ax = plt.subplots(figsize=(12, 4))
    n = data["n"]
    lt = data["lt"]

    ax.axvspan(lt, n, alpha=0.08, color=COLORS["purple"])
    ax.plot(range(n), data["compounds_v"], color=COLORS["yellow"], lw=1.0, alpha=0.6, label="VADER")
    ax.plot(range(n), data["compounds_r"], color=COLORS["accent"], lw=1.2, alpha=0.85, label="RoBERTa")
    ax.axhline(y=0, color=COLORS["muted"], ls=":", lw=0.6, alpha=0.4)

    r, _ = pearsonr(data["compounds_v"], data["compounds_r"])
    ax.text(0.02, 0.97, f"Correlação: r = {r:.2f}", transform=ax.transAxes,
            fontsize=9, color=COLORS["muted"], va="top")
    ax.text(lt + 1, ax.get_ylim()[1] * 0.9, "Último terço", fontsize=8,
            color=COLORS["purple"], alpha=0.8)

    ax.set_xlabel("Cena", color=COLORS["muted"], fontsize=10)
    ax.set_ylabel("Sentimento", color=COLORS["muted"], fontsize=10)
    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["border"],
              labelcolor=COLORS["text"], fontsize=9)
    ax.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    for s in ax.spines.values():
        s.set_color(COLORS["border"])
    fig.set_facecolor(COLORS["bg"])
    return fig_to_base64(fig)


def plot_graph(T, states, state_labels, state_colors, title, gmetrics, prefix):
    """Grafo de transição emocional."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    total = T.sum()
    if total == 0:
        ax.text(0.5, 0.5, "Sem dados", ha="center", va="center",
                color=COLORS["muted"], fontsize=14, transform=ax.transAxes)
        ax.axis("off")
        return fig_to_base64(fig)

    G = nx.DiGraph()
    for s in states:
        G.add_node(s)
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            if T[i][j] > 0:
                G.add_edge(s1, s2, weight=T[i][j])

    pos = nx.circular_layout(G)
    pr_vals = [gmetrics.get(f"{prefix}_pr_{s}", 1.0 / len(states)) for s in states]
    max_pr = max(pr_vals) if pr_vals else 1
    node_sizes = [500 + 2000 * (v / max_pr) for v in pr_vals]
    n_colors = [state_colors.get(s, COLORS["muted"]) for s in states]

    edges = list(G.edges(data=True))
    if edges:
        weights = [d["weight"] for _, _, d in edges]
        max_w = max(weights)
        for (u, v, d) in edges:
            ew = 0.5 + 3.0 * (d["weight"] / max_w)
            style = "arc3,rad=0.3" if u == v else "arc3,rad=0.15"
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=ew,
                                    alpha=0.4, edge_color=COLORS["muted"],
                                    connectionstyle=style, ax=ax, arrows=True, arrowsize=10)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=n_colors,
                           edgecolors=COLORS["text"], linewidths=1.5, ax=ax)
    labels = {s: state_labels.get(s, s) for s in states}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold",
                            font_color="white", ax=ax)

    ax.set_title(title, color=COLORS["text"], fontsize=11, pad=8)
    ax.axis("off")
    return fig_to_base64(fig)


def plot_metrics(gv5, gv5r):
    """Métricas comparativas VADER vs RoBERTa."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.set_facecolor(COLORS["bg"])

    metrics = [
        ("Estabilidade (Autoloop)", "gv5_autoloop"),
        ("Previsibilidade (Entropia)", "gv5_entropy"),
        ("Oscilação (Reciprocidade)", "gv5_reciprocity"),
    ]

    for ax, (title, key) in zip(axes, metrics):
        v_val = gv5.get(key, 0)
        r_val = gv5r.get(key, 0)
        items = [("VADER", v_val, COLORS["yellow"]), ("RoBERTa", r_val, COLORS["accent"])]
        bars = ax.barh([i[0] for i in items][::-1], [i[1] for i in items][::-1],
                        color=[i[2] for i in items][::-1], height=0.45)
        for bar, val in zip(bars, [i[1] for i in items][::-1]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=10, fontweight="bold",
                     color=COLORS["text"])
        ax.set_xlim(0, max(v_val, r_val, 0.01) * 1.5 + 0.05)
        ax.set_title(title, fontsize=10, color=COLORS["text"])
        ax.set_facecolor(COLORS["bg"])
        ax.tick_params(colors=COLORS["muted"], labelsize=8)
        for s in ax.spines.values():
            s.set_color(COLORS["border"])

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_characters(char_data):
    """Rede de personagens + arco do protagonista."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                     gridspec_kw={"width_ratios": [3, 2]})
    fig.set_facecolor(COLORS["bg"])

    film_G = char_data.get("graph")
    protag = char_data.get("protagonist", "?")

    if film_G and film_G.number_of_nodes() >= 2:
        pos = nx.spring_layout(film_G, seed=42, k=2.0)
        strengths = {n: sum(d["weight"] for _, _, d in film_G.edges(n, data=True))
                     for n in film_G.nodes()}
        max_s = max(strengths.values()) if strengths else 1
        node_sizes = [100 + 1200 * (strengths.get(n, 0) / max_s) for n in film_G.nodes()]
        node_colors = [COLORS["accent"] if n == protag else COLORS["muted"]
                       for n in film_G.nodes()]

        edges = list(film_G.edges(data=True))
        if edges:
            weights = [d.get("weight", 1) for _, _, d in edges]
            max_w = max(weights)
            widths = [0.3 + 2.5 * (w / max_w) for w in weights]
            nx.draw_networkx_edges(film_G, pos, width=widths, alpha=0.2,
                                    edge_color=COLORS["muted"], ax=ax1)

        nx.draw_networkx_nodes(film_G, pos, node_size=node_sizes, node_color=node_colors,
                               edgecolors=COLORS["text"], linewidths=0.8, ax=ax1, alpha=0.9)
        top = sorted(strengths, key=strengths.get, reverse=True)[:8]
        nx.draw_networkx_labels(film_G, pos, {n: n for n in top}, font_size=6,
                                font_weight="bold", font_color="white", ax=ax1)

        nn = film_G.number_of_nodes()
        ne = film_G.number_of_edges()
        ax1.set_title(f"Rede: {nn} personagens, {ne} interações\nProtagonista: {protag}",
                      color=COLORS["text"], fontsize=10)
    else:
        ax1.text(0.5, 0.5, "Personagens não detectados", ha="center", va="center",
                 fontsize=12, color=COLORS["muted"], transform=ax1.transAxes)

    ax1.axis("off")
    ax1.set_facecolor(COLORS["bg"])

    # Arco do protagonista
    pv = char_data.get("protag_v", [])
    pr = char_data.get("protag_r", [])

    if pv:
        ax2.plot(range(len(pv)), pv, color=COLORS["yellow"], lw=1.0, alpha=0.6, label="VADER")
    if pr:
        ax2.plot(range(len(pr)), pr, color=COLORS["accent"], lw=1.2, alpha=0.85, label="RoBERTa")
    if pv or pr:
        ax2.axhline(y=0, color=COLORS["muted"], ls=":", lw=0.6, alpha=0.4)
        ax2.set_xlabel("Cena", color=COLORS["muted"], fontsize=9)
        ax2.set_ylabel("Sentimento", color=COLORS["muted"], fontsize=9)
        ax2.set_title(f"Arco: {protag}", color=COLORS["text"], fontsize=10)
        ax2.legend(facecolor=COLORS["card"], edgecolor=COLORS["border"],
                   labelcolor=COLORS["text"], fontsize=8)
        ax2.set_facecolor(COLORS["bg"])
        ax2.tick_params(colors=COLORS["muted"], labelsize=8)
        for s in ax2.spines.values():
            s.set_color(COLORS["border"])
    else:
        ax2.text(0.5, 0.5, "Sem dados", ha="center", va="center",
                 fontsize=12, color=COLORS["muted"], transform=ax2.transAxes)
        ax2.axis("off")
        ax2.set_facecolor(COLORS["bg"])

    plt.tight_layout()
    return fig_to_base64(fig)


# ─── HTML ───

def generate_html(film_results):
    """Gera HTML completo do relatório."""

    V5_COLORS = {"v5_high": COLORS["v5_high"], "v5_pos": COLORS["v5_pos"],
                  "v5_neu": COLORS["v5_neu"], "v5_neg": COLORS["v5_neg"], "v5_low": COLORS["v5_low"]}
    V5_LABELS = {"v5_high": "M.Positivo", "v5_pos": "Positivo", "v5_neu": "Neutro",
                  "v5_neg": "Negativo", "v5_low": "M.Negativo"}

    sections = []

    for genre, data in film_results.items():
        if data is None:
            continue

        title_clean = data["title"].replace("-", " ").title()

        # Gerar gráficos
        logging.info("  Gerando gráficos para %s...", title_clean)
        img_sentiment = plot_sentiment(data)
        img_graph_vader = plot_graph(data["T5"], data["v5_names"], V5_LABELS, V5_COLORS,
                                      "VADER — Grafo V5", data["gv5"], "gv5")
        img_graph_roberta = plot_graph(data["T5r"], data["v5_names"], V5_LABELS, V5_COLORS,
                                        "RoBERTa — Grafo V5", data["gv5r"], "gv5")
        img_metrics = plot_metrics(data["gv5"], data["gv5r"])
        img_chars = plot_characters(data["characters"])

        # Métricas texto
        autoloop_v = data["gv5"].get("gv5_autoloop", 0)
        autoloop_r = data["gv5r"].get("gv5_autoloop", 0)
        entropy_v = data["gv5"].get("gv5_entropy", 0)
        entropy_r = data["gv5r"].get("gv5_entropy", 0)
        protag = data["characters"].get("protagonist", "—")
        n_chars = data["characters"]["graph"].number_of_nodes() if data["characters"]["graph"] else 0

        pv = data["characters"].get("protag_v", [])
        pr = data["characters"].get("protag_r", [])
        protag_mean_v = f"{np.mean(pv):+.3f}" if pv else "—"
        protag_mean_r = f"{np.mean(pr):+.3f}" if pr else "—"
        genre_id = genre.lower()

        section = f"""
        <div class="film-section" id="film-{genre_id}">
            <div class="film-header">
                <span class="genre-badge">{genre}</span>
                <h2>{title_clean}</h2>
                <span class="film-stats">{data['n']} cenas • {n_chars} personagens • Protagonista: {protag}</span>
            </div>

            <div class="step">
                <div class="step-number">1</div>
                <div class="step-content">
                    <h3>Curva Emocional — VADER vs RoBERTa</h3>
                    <p>Cada cena recebeu um score de sentimento por dois modelos:
                    <b style="color:{COLORS['yellow']}">VADER</b> (dicionário de palavras) e
                    <b style="color:{COLORS['accent']}">RoBERTa</b> (rede neural).
                    A zona roxa marca o último terço do filme (clímax e desfecho).</p>
                    <img src="{img_sentiment}" class="chart">
                </div>
            </div>

            <div class="step">
                <div class="step-number">2</div>
                <div class="step-content">
                    <h3>Grafos de Transição Emocional</h3>
                    <p>Os scores contínuos são transformados em 5 estados (Muito Positivo a Muito Negativo).
                    Cada seta mostra uma transição entre cenas consecutivas.
                    O tamanho dos círculos reflete o PageRank — qual estado "domina" o fluxo emocional.</p>
                    <div class="graphs-row">
                        <div class="graph-item">
                            <img src="{img_graph_vader}" class="chart-half">
                        </div>
                        <div class="graph-item">
                            <img src="{img_graph_roberta}" class="chart-half">
                        </div>
                    </div>
                </div>
            </div>

            <div class="step">
                <div class="step-number">3</div>
                <div class="step-content">
                    <h3>Métricas dos Grafos — VADER vs RoBERTa</h3>
                    <p>De cada grafo extraímos 4 métricas que quantificam a dinâmica emocional.
                    Cada uma responde a uma pergunta diferente sobre como o filme se comporta emocionalmente:</p>
                    <img src="{img_metrics}" class="chart">

                    <p style="margin-top:12px"><b>Estabilidade Emocional (Autoloop)</b> —
                    <i>"O filme mantém o mesmo clima entre cenas consecutivas?"</i><br>
                    Calculada como a fração das transições que são self-loops (o estado se mantém):
                    autoloop = tr(T) / Σ(T), onde T é a matriz de transição do grafo.
                    Um autoloop de 0.90 significa que em 90% das vezes, a emoção da cena seguinte
                    é a mesma da anterior.
                    Na pesquisa, esta foi a <b>métrica mais forte</b>:
                    em Thriller, autoloop alto correlaciona com melhor IMDb (ρ=+0.25 com RoBERTa).</p>

                    <p><b>Previsibilidade (Entropia de Shannon)</b> —
                    <i>"As transições emocionais seguem um padrão reconhecível?"</i><br>
                    Calculada como H = −Σ(pᵢⱼ · log₂(pᵢⱼ)), onde pᵢⱼ é a probabilidade de transição
                    do estado i para o estado j. Entropia baixa = poucas transições dominam
                    (padrão claro). Entropia alta = todas as transições ocorrem com frequência similar
                    (caos emocional). Com RoBERTa, entropia baixa em Thriller correlaciona
                    com melhor IMDb (ρ=−0.25) — achado que era <b>invisível</b> com VADER.</p>

                    <p><b>Oscilação (Reciprocidade)</b> —
                    <i>"O filme vai e volta entre os mesmos estados?"</i><br>
                    Calculada como Σ min(Tᵢⱼ, Tⱼᵢ) / (Σ(T) − tr(T)), onde Tᵢⱼ é a contagem
                    de transições de i para j. Mede transições bidirecionais — se o filme alterna
                    frequentemente entre positivo e negativo, a reciprocidade é alta.
                    Reciprocidade alta indica "montanha-russa emocional".</p>

                    <div class="metrics-summary">
                        <div class="metric-card">
                            <span class="metric-label">Estabilidade (Autoloop)</span>
                            <span class="metric-values">
                                <span style="color:{COLORS['yellow']}">VADER: {autoloop_v:.3f}</span>
                                <span style="color:{COLORS['accent']}">RoBERTa: {autoloop_r:.3f}</span>
                            </span>
                        </div>
                        <div class="metric-card">
                            <span class="metric-label">Previsibilidade (Entropia)</span>
                            <span class="metric-values">
                                <span style="color:{COLORS['yellow']}">VADER: {entropy_v:.3f}</span>
                                <span style="color:{COLORS['accent']}">RoBERTa: {entropy_r:.3f}</span>
                            </span>
                        </div>
                    </div>
                    <p style="margin-top:12px; font-size:13px; color:#444d56">
                    <b>Interpretação para este filme:</b> o autoloop VADER de {autoloop_v:.1%} indica que
                    em {autoloop_v:.0%} das transições entre cenas consecutivas o estado emocional
                    permaneceu inalterado, sugerindo
                    {'elevada' if autoloop_v > 0.8 else 'moderada' if autoloop_v > 0.5 else 'baixa'}
                    estabilidade emocional. Com RoBERTa, o autoloop é de {autoloop_r:.1%},
                    {'confirmando' if abs(autoloop_r - autoloop_v) < 0.15 else 'divergindo do'}
                    o padrão observado com VADER.</p>
                </div>
            </div>

            <div class="step">
                <div class="step-number">4</div>
                <div class="step-content">
                    <h3>Rede de Personagens e Arco do Protagonista</h3>
                    <p>À esquerda: rede de interações entre personagens, onde cada nó representa
                    um personagem e cada aresta indica diálogo entre eles. O nó
                    <b style="color:{COLORS['accent']}">azul</b> identifica o protagonista
                    (personagem com maior número de interações).
                    À direita: evolução do sentimento das falas do protagonista ao longo do filme.</p>
                    <img src="{img_chars}" class="chart">
                    <div class="metrics-summary">
                        <div class="metric-card">
                            <span class="metric-label">Tom médio do protagonista</span>
                            <span class="metric-values">
                                <span style="color:{COLORS['yellow']}">VADER: {protag_mean_v}</span>
                                <span style="color:{COLORS['accent']}">RoBERTa: {protag_mean_r}</span>
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        sections.append(section)

    films_html = "\n".join(sections)

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mapa da Dissertação — Análise de Roteiros Cinematográficos</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: #f8f9fa;
            color: #1a1a2e;
            line-height: 1.7;
        }}

        /* Sidebar fixa */
        .sidebar {{
            position: fixed;
            top: 0;
            left: 0;
            width: 260px;
            height: 100vh;
            background: #ffffff;
            border-right: 1px solid #e1e4e8;
            padding: 24px 16px;
            overflow-y: auto;
            z-index: 100;
        }}
        .sidebar h3 {{
            font-size: 13px;
            color: #6a737d;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 16px;
        }}
        .sidebar a {{
            display: block;
            padding: 8px 12px;
            color: #24292e;
            text-decoration: none;
            font-size: 13px;
            border-radius: 6px;
            margin-bottom: 2px;
            transition: background 0.15s;
        }}
        .sidebar a:hover {{
            background: #f0f2f5;
            color: #0366d6;
        }}
        .sidebar a.indent {{
            padding-left: 28px;
            font-size: 12px;
            color: #6a737d;
        }}
        .sidebar a.indent:hover {{ color: #0366d6; }}
        .sidebar .sep {{
            height: 1px;
            background: #e1e4e8;
            margin: 12px 0;
        }}

        .container {{ max-width: 900px; margin: 0 auto; padding: 40px 24px 40px 284px; }}

        /* Header */
        .header {{
            text-align: center;
            padding: 48px 0 32px;
            border-bottom: 1px solid #e1e4e8;
            margin-bottom: 36px;
        }}
        .header h1 {{ font-size: 28px; color: #0366d6; margin-bottom: 10px; font-weight: 700; }}
        .header p {{ color: #586069; font-size: 15px; max-width: 700px; margin: 0 auto; }}
        .header .subtitle {{ font-size: 12px; color: #959da5; margin-top: 8px; }}

        /* Seções */
        .intro-section {{
            background: #ffffff;
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 28px;
            margin-bottom: 28px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        .intro-section h2 {{ color: #0366d6; font-size: 20px; margin-bottom: 16px; }}
        .intro-section h3 {{ color: #24292e; font-size: 16px; margin: 16px 0 8px; }}
        .intro-section p {{ color: #444d56; margin-bottom: 12px; font-size: 14px; }}
        .intro-section ol, .intro-section ul {{ color: #444d56; padding-left: 24px; margin-bottom: 12px; }}
        .intro-section li {{ margin-bottom: 6px; font-size: 14px; }}
        .intro-section b {{ color: #24292e; }}
        .intro-section code {{ background: #f0f2f5; padding: 2px 6px; border-radius: 3px;
            font-size: 12px; color: #e36209; }}
        .intro-section table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
        .intro-section td {{ padding: 10px; border-bottom: 1px solid #e1e4e8; font-size: 13px;
            color: #444d56; vertical-align: top; }}
        .intro-section .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 11px; font-weight: 600; }}
        .tag-green {{ background: #dcffe4; color: #22863a; }}
        .tag-yellow {{ background: #fff5b1; color: #b08800; }}
        .tag-red {{ background: #ffdce0; color: #cb2431; }}
        .tag-blue {{ background: #dbedff; color: #0366d6; }}
        .tag-purple {{ background: #f0e0ff; color: #6f42c1; }}

        /* Filme */
        .film-section {{
            background: #ffffff;
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 28px;
            margin-bottom: 28px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        .film-header {{
            border-bottom: 1px solid #e1e4e8;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .film-header h2 {{ font-size: 22px; color: #24292e; margin: 8px 0 4px; }}
        .film-header .film-stats {{ font-size: 12px; color: #959da5; }}
        .genre-badge {{
            display: inline-block;
            background: #0366d6;
            color: white;
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Steps */
        .step {{
            display: flex;
            gap: 16px;
            margin-bottom: 28px;
            padding-bottom: 24px;
            border-bottom: 1px solid #eaecef;
        }}
        .step:last-child {{ border-bottom: none; }}
        .step-number {{
            flex-shrink: 0;
            width: 32px; height: 32px;
            background: #0366d6;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
            margin-top: 4px;
        }}
        .step-content {{ flex: 1; }}
        .step-content h3 {{ color: #24292e; font-size: 16px; margin-bottom: 8px; }}
        .step-content p {{ color: #444d56; font-size: 13px; margin-bottom: 12px; }}

        /* Charts */
        .chart {{ width: 100%; border-radius: 8px; margin: 8px 0;
            border: 1px solid #e1e4e8; }}
        .chart-half {{ width: 100%; border-radius: 8px; border: 1px solid #e1e4e8; }}
        .graphs-row {{ display: flex; gap: 12px; }}
        .graph-item {{ flex: 1; }}

        /* Metrics */
        .metrics-summary {{ display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }}
        .metric-card {{
            background: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 12px 16px;
            flex: 1;
            min-width: 200px;
        }}
        .metric-label {{ display: block; font-size: 11px; color: #959da5;
            text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }}
        .metric-values {{ display: flex; gap: 16px; font-size: 14px; font-weight: 600;
            color: #24292e; }}

        /* Findings */
        .finding {{ padding: 12px 16px; border-left: 3px solid; margin-bottom: 8px;
            border-radius: 0 6px 6px 0; font-size: 14px; color: #444d56; }}
        .finding-green {{ border-color: #28a745; background: #f0fff4; }}
        .finding-red {{ border-color: #d73a49; background: #fff5f5; }}
        .finding-yellow {{ border-color: #e3b341; background: #fffdef; }}
        .finding b {{ color: #24292e; }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 40px 0 20px;
            border-top: 1px solid #e1e4e8;
            color: #959da5;
            font-size: 12px;
        }}

        /* Scroll suave */
        html {{ scroll-behavior: smooth; }}
    </style>
</head>
<body>
    <!-- Sidebar fixa -->
    <nav class="sidebar">
        <h3>Índice</h3>
        <a href="#top">Início</a>
        <a href="#glossario">Glossário de Termos</a>
        <div class="sep"></div>
        <a href="#pipeline">Como Foi Feito</a>
        <a class="indent" href="#pipeline">1. Coleta e Segmentação</a>
        <a class="indent" href="#pipeline">2. Sentimento por Cena</a>
        <a class="indent" href="#pipeline">3. Discretização</a>
        <a class="indent" href="#pipeline">4. Grafos de Transição</a>
        <a class="indent" href="#pipeline">5. Métricas dos Grafos</a>
        <a class="indent" href="#pipeline">6. Personagens</a>
        <a class="indent" href="#pipeline">7. Análise Estatística</a>
        <div class="sep"></div>
        <a href="#ia">Componentes de IA</a>
        <a class="indent" href="#vader-detail">VADER</a>
        <a class="indent" href="#roberta-detail">RoBERTa</a>
        <a class="indent" href="#llm-detail">Claude Haiku (LLM)</a>
        <a class="indent" href="#ridge-detail">Ridge Regression</a>
        <a href="#combinacao">Por Que Combinar</a>
        <div class="sep"></div>
        <a href="#filmes">Análise por Gênero</a>
        <a class="indent" href="#film-thriller">Thriller — Fight Club</a>
        <a class="indent" href="#film-horror">Horror — Psycho</a>
        <a class="indent" href="#film-comedy">Comedy — Dead Poets Society</a>
        <a class="indent" href="#film-drama">Drama — Pulp Fiction</a>
        <a class="indent" href="#film-action">Action — Inception</a>
        <a class="indent" href="#film-crime">Crime — Godfather Part II</a>
        <a class="indent" href="#film-romance">Romance — Good Will Hunting</a>
        <div class="sep"></div>
        <a href="#resultados">Resultados</a>
        <a class="indent" href="#achados">Achados Fortes</a>
        <a class="indent" href="#nao-funciona">O Que Não Funciona</a>
        <a class="indent" href="#modelos">Melhor Modelo</a>
        <a class="indent" href="#conclusao">Conclusão</a>
    </nav>

    <div class="container">

        <div class="header" id="top">
            <h1>Mapa da Dissertação</h1>
            <p>Análise de roteiros cinematográficos por grafos de transição emocional
            e redes de personagens: pode a estrutura narrativa de um roteiro
            prever o sucesso de um filme?</p>
            <p class="subtitle">7 filmes analisados • 2 modelos de sentimento (VADER + RoBERTa) •
            Grafos emocionais + Redes de personagens</p>
        </div>

        <!-- Glossário de termos -->
        <div class="intro-section" id="glossario">
            <h2>Glossário de Termos</h2>
            <p>Os termos abaixo são utilizados ao longo deste relatório. Recomenda-se
            a consulta desta seção antes da leitura das demais.</p>
            <table>
                <tr><td><b>IMDb</b></td><td>Internet Movie Database. Plataforma onde o público
                atribui notas de 1 a 10 a filmes. A nota IMDb é a média dessas avaliações
                e serve como indicador de <b>qualidade percebida pelo público</b>.</td></tr>
                <tr><td><b>Box Office</b></td><td>Bilheteira bruta (em dólares) arrecadada pelo
                filme nos Estados Unidos. Serve como indicador de <b>sucesso comercial</b>.
                Não reflete lucro real, pois não desconta custos de produção e marketing.</td></tr>
                <tr><td><b>R² (R-quadrado)</b></td><td>Coeficiente de determinação. Mede a proporção
                da variância dos dados que o modelo consegue explicar. Varia de 0 a 1.
                Por exemplo, R² = 0.15 significa que o modelo explica 15% da variação.
                Valores acima de 0.10 já são considerados relevantes em ciências sociais.</td></tr>
                <tr><td><b>ρ (rho)</b></td><td>Coeficiente de correlação de Spearman. Mede a força
                e a direção da associação monótona entre duas variáveis. Varia de -1 a +1.
                Valores positivos indicam que quando uma variável sobe, a outra também sobe.
                Valores negativos indicam relação inversa. Por exemplo, ρ = +0.25 indica
                associação positiva moderada.</td></tr>
                <tr><td><b>AUC</b></td><td>Area Under the Curve (Área sob a Curva ROC). Mede a
                capacidade de um modelo de classificação distinguir entre duas classes.
                Varia de 0.5 (aleatório, sem capacidade discriminativa) a 1.0 (perfeito).
                Valores acima de 0.7 são considerados aceitáveis.</td></tr>
                <tr><td><b>Baseline</b></td><td>Modelo de referência simples, usado para comparação.
                Nesta pesquisa, o baseline utiliza apenas o ano de lançamento e o gênero do
                filme para prever a nota IMDb ou a bilheteira. Se um modelo mais complexo
                não supera o baseline, as features adicionais não contribuem com informação
                útil.</td></tr>
                <tr><td><b>Feature</b></td><td>Variável numérica extraída dos dados que serve
                como entrada para um modelo estatístico. Por exemplo, o autoloop do grafo
                emocional de um filme é uma feature. Nesta pesquisa, foram extraídas 44 features
                emocionais e 29 features de rede de personagens por filme.</td></tr>
                <tr><td><b>Overfitting</b></td><td>Situação em que um modelo se ajusta excessivamente
                aos dados de treinamento, capturando ruído em vez de padrões reais. Ocorre
                tipicamente quando o número de features é grande em relação ao número de
                amostras. O modelo parece funcionar bem nos dados de treino, mas falha em dados
                novos.</td></tr>
                <tr><td><b>Validação cruzada</b></td><td>Técnica para avaliar a capacidade de
                generalização de um modelo. Os dados são divididos em partições: o modelo
                é treinado em parte dos dados e testado na parte restante, repetindo-se o
                processo para todas as partições. Nesta pesquisa, utilizou-se validação cruzada
                de 5 partições (5-fold CV) e, em análises iniciais, Leave-One-Out (LOO-CV).</td></tr>
                <tr><td><b>Compound</b></td><td>Score de sentimento normalizado no intervalo [-1, +1],
                produzido pelo VADER ou pelo RoBERTa. Valores próximos de +1 indicam sentimento
                muito positivo; próximos de -1, muito negativo; próximos de 0, neutro.</td></tr>
                <tr><td><b>Transformer</b></td><td>Arquitetura de rede neural proposta por Vaswani
                et al. (2017) que processa texto compreendendo relações entre palavras distantes.
                É a base dos modelos BERT, RoBERTa, GPT e ChatGPT.</td></tr>
                <tr><td><b>Fine-tuning</b></td><td>Processo de ajustar um modelo pré-treinado
                para uma tarefa específica, treinando-o adicionalmente com dados rotulados
                daquela tarefa. O modelo RoBERTa utilizado nesta pesquisa foi pré-treinado
                em textos gerais e posteriormente refinado (fine-tuned) para classificação
                de sentimento.</td></tr>
                <tr><td><b>One-hot encoding</b></td><td>Técnica de representação de variáveis
                categóricas como vetores binários. Por exemplo, o gênero "Thriller" é representado
                como um vetor onde a posição correspondente a Thriller é 1 e todas as demais são 0.
                Nesta pesquisa, utilizado para representar os gêneros dos filmes como
                entrada do modelo Ridge.</td></tr>
            </table>
        </div>

        <!-- Pipeline -->
        <div class="intro-section">
            <h2 id="pipeline">Como a Pesquisa Foi Feita — Passo a Passo</h2>

            <h3>Passo 1: Coleta e Segmentação dos Roteiros</h3>
            <p>Foram coletados <b>2393 roteiros</b> de filmes americanos a partir de duas fontes:
            o Internet Movie Script Database (IMSDb, aproximadamente 1100 roteiros em formato texto puro) e o
            dataset MovieSum disponível no HuggingFace (aproximadamente 1300 roteiros em formato XML estruturado).
            Os roteiros do IMSDb foram obtidos por meio da biblioteca <code>datasets</code> do HuggingFace,
            enquanto os do MovieSum foram convertidos de XML para texto puro, preservando a estrutura de cenas.</p>
            <p>Cada roteiro foi <b>segmentado automaticamente em cenas</b> por meio de uma expressão regular
            que identifica <i>sluglines</i>, os cabeçalhos padronizados de roteiros americanos
            (por exemplo: <code>INT. CASA - DIA</code>, <code>EXT. RUA - NOITE</code>).
            O texto compreendido entre duas sluglines consecutivas constitui uma cena.
            Cenas com menos de 20 caracteres foram incorporadas à cena seguinte, uma vez que
            correspondiam tipicamente a indicações técnicas como "CUT TO:". Após filtragem de qualidade
            (exclusão de roteiros sem sluglines reconhecíveis, muito curtos ou sem metadados associados),
            restaram <b>1537 filmes</b> com roteiro válido, avaliação IMDb e rede de personagens.</p>

            <h3>Passo 2: Medição do Sentimento por Cena</h3>
            <p>Para cada cena de cada filme, o sentimento emocional do texto foi medido
            por meio de <b>dois modelos independentes</b>:</p>
            <ul>
                <li><b style="color:{COLORS['yellow']}">VADER</b>: léxico de sentimento com 7518 palavras
                e scores fixos atribuídos por avaliadores humanos (Hutto & Gilbert, 2014). Cada cena é truncada
                a 2000 caracteres, segmentada em sentenças, e cada sentença recebe um score.
                O score final da cena corresponde à média ponderada pelo comprimento das sentenças,
                resultando em um valor contínuo no intervalo [−1, +1].</li>
                <li><b style="color:{COLORS['accent']}">RoBERTa</b>: modelo neural baseado na arquitetura
                Transformer, com 125 milhões de parâmetros (Liu et al., 2019). Cada cena é submetida
                integralmente ao modelo, que retorna probabilidades para três classes (positivo, neutro, negativo).
                O score contínuo é calculado como a diferença entre as probabilidades das classes positiva e negativa:
                <i>s = P(positivo) − P(negativo)</i>.</li>
            </ul>
            <p>O resultado é uma <b>curva emocional</b> para cada filme, na qual cada ponto corresponde
            ao sentimento de uma cena. Esta representação permite visualizar a evolução emocional
            ao longo da narrativa.</p>

            <h3>Passo 3: Discretização em Estados Emocionais</h3>
            <p>Para a construção dos grafos de transição, os scores contínuos foram transformados em
            <b>estados categóricos</b> por meio de limiares fixos aplicados ao valor <i>compound</i>
            de cada cena:</p>
            <table>
                <tr><td><b>Muito Positivo</b></td><td>compound ≥ 0.30</td>
                    <td>Cenas de alegria intensa, celebração</td></tr>
                <tr><td><b>Positivo</b></td><td>0.05 ≤ compound &lt; 0.30</td>
                    <td>Cenas levemente positivas, diálogos amigáveis</td></tr>
                <tr><td><b>Neutro</b></td><td>−0.05 &lt; compound &lt; 0.05</td>
                    <td>Cenas descritivas, transições, exposição</td></tr>
                <tr><td><b>Negativo</b></td><td>−0.30 &lt; compound ≤ −0.05</td>
                    <td>Cenas de tensão, conflito moderado</td></tr>
                <tr><td><b>Muito Negativo</b></td><td>compound ≤ −0.30</td>
                    <td>Cenas de violência, morte, desespero</td></tr>
            </table>
            <p>Dessa forma, cada filme é convertido em uma <b>sequência de estados</b>, como por exemplo:
            Neutro, Positivo, Positivo, Negativo, Muito Negativo, Neutro, etc.
            Essa sequência constitui a entrada para a construção do grafo de transição.</p>

            <h3>Passo 4: Construção dos Grafos de Transição Emocional</h3>
            <p>Cada filme é representado formalmente como um <b>grafo dirigido ponderado</b>
            G = (V, E, W), onde:</p>
            <ul>
                <li><b>V</b> (vértices) = conjunto dos N = 5 estados emocionais</li>
                <li><b>E</b> (arestas) = transições observadas entre cenas consecutivas.
                Se a cena <i>t</i> pertence ao estado <i>i</i> e a cena <i>t+1</i> pertence
                ao estado <i>j</i>, registra-se uma aresta direcionada de <i>i</i> para <i>j</i>.</li>
                <li><b>W</b> (pesos) = a cada aresta (i, j) é atribuído o peso T<sub>ij</sub>,
                correspondente ao número total de vezes que a transição do estado <i>i</i> para o
                estado <i>j</i> foi observada ao longo do filme.</li>
                <li><b>Self-loops</b> = arestas da forma (i, i), indicando que o estado emocional
                se manteve inalterado entre cenas consecutivas. Uma proporção elevada de self-loops
                indica <b>estabilidade emocional</b> na narrativa.</li>
            </ul>
            <p>O grafo pode ser representado na forma de uma <b>matriz de transição</b> T de dimensão
            N × N, onde cada elemento T<sub>ij</sub> contém a contagem de transições do estado <i>i</i>
            para o estado <i>j</i>. A normalização por linhas gera a matriz de probabilidades
            P, onde P<sub>ij</sub> = T<sub>ij</sub> / Σ<sub>k</sub> T<sub>ik</sub>,
            representando a probabilidade condicional de transição para o estado <i>j</i>
            dado que o estado atual é <i>i</i>.</p>

            <h3>Passo 5: Extração de Métricas dos Grafos</h3>
            <p>De cada grafo foram extraídas <b>métricas quantitativas</b> que sintetizam
            a dinâmica emocional do filme em valores numéricos comparáveis entre filmes.
            As quatro métricas principais são descritas a seguir, com notação consistente:
            T designa a matriz de contagens, P a matriz de probabilidades, e N o número de estados.</p>

            <p><b>Autoloop (Estabilidade Emocional)</b></p>
            <p>Definido como a razão entre a soma dos elementos diagonais de T (o traço) e a soma
            total de todas as transições: <i>A = tr(T) / Σ<sub>ij</sub> T<sub>ij</sub></i>.
            Um valor de A = 0.90 indica que em 90% das transições entre cenas consecutivas o estado
            emocional permaneceu inalterado. Esta métrica quantifica a estabilidade do tom emocional
            ao longo da narrativa. Na presente pesquisa, observou-se que em filmes do gênero Thriller,
            valores elevados de autoloop apresentaram correlação positiva significativa com a avaliação
            IMDb (ρ = +0.25, p &lt; 0.001).</p>

            <p><b>Entropia de Shannon (Previsibilidade)</b></p>
            <p>Calculada sobre a matriz de probabilidades P como
            <i>H = −Σ<sub>ij</sub> P<sub>ij</sub> · log<sub>2</sub>(P<sub>ij</sub>)</i>,
            considerando apenas os elementos P<sub>ij</sub> &gt; 0. A entropia quantifica a
            imprevisibilidade das transições emocionais. Valores baixos indicam que poucas transições
            concentram a maior parte das ocorrências, sugerindo um padrão emocional reconhecível.
            Valores elevados indicam dispersão uniforme entre todas as transições possíveis,
            caracterizando um fluxo emocional mais caótico. Notavelmente, esta métrica não apresentou
            correlação significativa com IMDb quando calculada sobre scores VADER (ρ ≈ 0), mas
            tornou-se significativa com scores RoBERTa (ρ = −0.25 em Thriller, p &lt; 0.001),
            evidenciando a sensibilidade do modelo de sentimento na extração de sinal.</p>

            <p><b>Reciprocidade (Oscilação Bidirecional)</b></p>
            <p>Mede a proporção de transições bidirecionais no grafo, calculada como
            <i>R = Σ<sub>i&lt;j</sub> min(T<sub>ij</sub>, T<sub>ji</sub>) / (Σ<sub>ij</sub> T<sub>ij</sub> − tr(T))</i>,
            onde o numerador captura a menor contagem de cada par bidirecional e o denominador
            exclui os self-loops. Um valor elevado de R indica oscilação emocional frequente
            entre estados opostos. Por exemplo, um filme que apresenta alternâncias recorrentes
            entre os estados Positivo e Negativo em cenas consecutivas terá reciprocidade alta.
            Um valor baixo indica progressão emocional predominantemente unidirecional,
            sem retorno frequente aos estados anteriores.</p>

            <p><b>PageRank (Centralidade por Caminhada Aleatória)</b></p>
            <p>O PageRank, originalmente proposto por Brin e Page (1998) para classificação de
            páginas web, é uma medida de centralidade baseada em caminhadas aleatórias. Aplicado
            ao grafo de transição emocional, quantifica a importância relativa de cada estado
            no fluxo emocional global. O algoritmo é definido iterativamente:
            <i>PR(v) = (1 − d) / N + d · Σ<sub>u</sub> PR(u) · P<sub>uv</sub></i>,
            onde d é o fator de amortecimento (d = 0.85, por convenção), N é o número de estados,
            e P<sub>uv</sub> é a probabilidade de transição do estado u para o estado v.
            O fator d controla o equilíbrio entre seguir as transições observadas (85% do tempo)
            e recomeçar de um estado aleatório (15%), garantindo que estados sem transições de
            entrada não recebam importância zero. Na pesquisa, observou-se que em Thrillers
            com melhor avaliação, o PageRank do estado Negativo no último terço do filme é
            significativamente menor (ρ = −0.27, p &lt; 0.001), indicando que estes filmes
            evitam concentrar o desfecho em emoções negativas.</p>

            <p>No total, foram extraídas <b>44 métricas</b> por filme, incluindo variantes
            calculadas sobre o filme inteiro (discretização V5 com 5 estados) e sobre o
            último terço (discretização V3 com 3 estados).</p>

            <h3>Passo 6: Rede de Personagens e Protagonista</h3>
            <p>Paralelamente aos grafos emocionais, foram construídas <b>redes sociais de personagens</b>
            para cada filme:</p>
            <ul>
                <li>Os personagens foram identificados pelo formato padronizado do roteiro americano
                (nome em letras maiúsculas centralizado acima do diálogo). Nomes similares foram
                agrupados automaticamente por meio da distância de Levenshtein, com limiar de
                similaridade ≥ 0.7 (por exemplo, "VITO" e "VITO CORLEONE" são tratados como o
                mesmo personagem).</li>
                <li>Para cada cena, foi registrada a sequência de falantes. Arestas foram criadas
                entre personagens que dialogam em sequência (adjacência de turno), com peso
                igual ao número total de interações ao longo do filme.</li>
                <li>Foram extraídas 29 métricas de rede (densidade, centralização de força,
                intermediação, coeficiente de clustering, modularidade, entre outras) e
                8 features do protagonista (tom emocional médio das falas, proporção de falas
                positivas e negativas, presença em cenas, arco emocional).</li>
            </ul>

            <h3>Passo 7: Análise Estatística e Modelagem Preditiva</h3>
            <p>Para cada par (feature, target), foi calculado o coeficiente de correlação de Spearman ρ,
            com correção de Benjamini-Hochberg para controle da taxa de falsos positivos em testes
            múltiplos. A análise foi estratificada por 7 gêneros cinematográficos. A modelagem
            preditiva utilizou regressão Ridge (penalização L2, α = 1.0) com validação cruzada
            de 5 partições (5-fold CV) sobre os 1537 filmes do corpus. Foram comparadas 7 configurações
            de modelo: baseline (ano e gênero), apenas features emocionais, apenas features de rede
            de personagens, apenas features do protagonista, e combinações das anteriores.</p>
        </div>

        <!-- ML/LLM -->
        <div class="intro-section">
            <h2 id="ia">Componentes de Inteligência Artificial</h2>

            <p><b>Ambiente de desenvolvimento:</b> todo o projeto foi implementado em
            <b>Python 3.12.7</b>, utilizando ambiente virtual isolado (<code>python -m venv .venv</code>).
            As dependências foram gerenciadas por meio do arquivo <code>pyproject.toml</code> e instaladas
            com <code>pip install</code>. O código foi executado localmente em macOS, sem GPU dedicada.
            Toda a inferência neural (modelo RoBERTa) foi realizada na CPU.</p>

            <h3 style="color:{COLORS['yellow']}">1. VADER<span id="vader-detail"></span> — Valence Aware Dictionary and sEntiment Reasoner
            <span class="tag tag-yellow">Não é IA</span></h3>

            <p><b>O que é:</b> um <b>léxico de sentimento</b> criado por Hutto & Gilbert (2014).
            Funciona como um dicionário: contém 7518 palavras, cada uma com um score de sentimento
            atribuído manualmente por avaliadores humanos. Exemplos: "love" = +0.64, "horrible" = −0.68,
            "okay" = +0.10.</p>

            <p><b>Instalação:</b> <code>pip install vaderSentiment==3.3.2</code>.
            O pacote inclui o léxico completo — não precisa de download adicional.
            Também instalei o <code>nltk==3.8</code> e baixei o tokenizador de sentenças
            com <code>nltk.download('punkt_tab')</code>.</p>

            <p><b>Implementação (arquivo <code>src/sentiment.py</code>):</b></p>
            <ol>
                <li>Para cada cena, o texto é <b>truncado a 2000 caracteres</b>
                (<code>text = text[:2000]</code>) — testei diferentes limites e 2000
                deu o melhor equilíbrio entre informação e diluição. Chamei essa variante
                de "v5_clipped" durante o desenvolvimento.</li>
                <li>O texto truncado é dividido em sentenças com <code>nltk.sent_tokenize(text)</code></li>
                <li>Cada sentença recebe um score via <code>SentimentIntensityAnalyzer().polarity_scores(sent)</code>,
                que retorna 4 valores: pos, neg, neu e compound</li>
                <li>O score final da cena é a <b>média ponderada pelo comprimento</b>:
                <code>weighted[key] += scores[key] * len(sent)</code>, dividido pelo peso total.
                Sentenças mais longas pesam mais porque contêm mais informação.</li>
                <li>O resultado é o <b>compound</b>: número normalizado entre −1 e +1</li>
            </ol>

            <p><b>Decisão de design:</b> testei 6 variantes de medição emocional
            (texto inteiro, só diálogo, máximo absoluto, mix ponderado 70/30, clipping,
            top-k sentenças). A variante "v5_clipped" venceu em validação contra anotações
            humanas (Pearson r = 0.40 vs 0.38 do baseline sem clipping).</p>

            <p><b>Limitação que observei:</b> o VADER não entende contexto. Uma cena com
            "the beautiful sunset cast golden light over the funeral" recebe score positivo
            porque "beautiful", "golden" e "light" são palavras positivas — mesmo que a cena
            seja de um funeral.</p>

            <h3 style="color:{COLORS['accent']}; margin-top:20px">2. RoBERTa<span id="roberta-detail"></span> — Robustly Optimized BERT Approach
            <span class="tag tag-blue">IA pré-treinada</span></h3>

            <p><b>O que é:</b> um <b>modelo de linguagem baseado na arquitetura Transformer</b>,
            com 125 milhões de parâmetros treináveis. O modelo específico que utilizei
            (<code>cardiffnlp/twitter-roberta-base-sentiment-latest</code>) foi pré-treinado
            pela Cardiff University em ~124 milhões de tweets e fine-tuned para classificação
            de sentimento em 3 classes: positivo, neutro e negativo.</p>

            <p><b>Instalação:</b> <code>pip install transformers==5.5.0 torch==2.11.0</code>.
            O modelo (499MB) é baixado automaticamente do HuggingFace na primeira execução
            e fica em cache em <code>~/.cache/huggingface/</code>. Não precisa de GPU —
            roda na CPU, mas significativamente mais lento.</p>

            <p><b>Implementação (arquivo <code>src/sentiment_roberta.py</code>):</b></p>
            <ol>
                <li>O modelo é carregado via <code>pipeline("sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                top_k=None)</code> — o parâmetro <code>top_k=None</code> é essencial:
                faz o modelo retornar as probabilidades de <b>todas as 3 classes</b>,
                não apenas a classe dominante.</li>
                <li>Cada cena (truncada a 2000 chars) é passada inteira ao modelo.
                Inicialmente eu dividia em sentenças como no VADER, mas isso era 10× mais lento
                e o resultado não melhorava — o Transformer já entende contexto de frases longas.</li>
                <li>O score contínuo é calculado como <b>P(positivo) − P(negativo)</b>,
                gerando um valor entre −1 e +1. Na primeira versão eu retornava 0.0 para
                cenas classificadas como "neutral", mas isso causava 80% das cenas com score
                zero — corrigi para usar a diferença de probabilidades, que dá scores
                contínuos mesmo em cenas neutras (ex: P(pos)=0.15, P(neg)=0.05 → +0.10).</li>
                <li>Para processar 2393 filmes × ~80 cenas = ~191.000 textos, usei batch
                processing (<code>batch_size=64</code>). Na CPU, isso rodou a ~10 cenas/segundo.
                O processamento completo levou <b>~12 horas</b>. Implementei salvamento
                incremental a cada 100 filmes para não perder progresso se o processo caísse.</li>
            </ol>

            <p><b>Diferença que observei:</b> nas análises de correlação com IMDb (Spearman),
            o RoBERTa produziu sinais consistentemente mais fortes que o VADER em 6 dos 7 gêneros.
            Em Thriller, autoloop passou de ρ=+0.15 (VADER) para ρ=+0.25 (RoBERTa) — 60% mais forte.
            A entropia, que era invisível com VADER (ρ≈0), tornou-se significativa com RoBERTa (ρ=−0.25).</p>

            <h3 style="color:{COLORS['purple']}; margin-top:20px">3. Claude Haiku<span id="llm-detail"></span> — Análise Semântica do Desfecho
            <span class="tag tag-purple">IA generativa (LLM)</span></h3>

            <p><b>O que é:</b> modelo de linguagem generativo da Anthropic (família Claude).
            Diferente do VADER e RoBERTa que medem sentimento de palavras/frases, um LLM
            compreende <b>significado narrativo</b> — pode dizer se um final é "resolvido",
            "catártico" ou "tragicamente belo".</p>

            <p><b>Instalação:</b> <code>pip install anthropic==0.86.0</code>. Acesso via API
            com chave de autenticação. Modelo utilizado: <code>claude-haiku-4-5-20251001</code>
            (versão mais leve e barata do Claude, suficiente para classificação estruturada).
            Custo: ~$0.001 por filme (578 filmes processados).</p>

            <p><b>Implementação (arquivo <code>archive/vader_pipeline/scripts/23_annotate_endings_llm.py</code>):</b></p>
            <ol>
                <li>Para cada filme, extraí o texto das <b>últimas 3 cenas</b> do roteiro
                (limitado a 8000 caracteres para caber no contexto do modelo)</li>
                <li>Montei um <b>prompt estruturado</b> com system prompt + user prompt.
                O system prompt instrui o modelo a analisar como analista de roteiro.
                O user prompt contém o texto das cenas + rubrica detalhada para cada dimensão.</li>
                <li>Configurei <code>temperature=0</code> (determinístico) e
                <code>max_tokens=256</code> (resposta curta em JSON)</li>
                <li>O modelo retorna JSON com 6 campos. Implementei validação: verifico se
                todos os campos existem, se os valores ordinais estão entre 1–5, e se a
                valência é uma das 4 categorias válidas. Se falhar, tento uma segunda vez.</li>
                <li>Converti para features numéricas: <code>llm_resolution</code> (1–5),
                <code>llm_catharsis</code> (1–5), etc. A valência categórica
                (happy/tragic/bittersweet/ambiguous) foi mapeada para valor numérico
                e também para indicadores binários (one-hot).</li>
            </ol>

            <p><b>Resultado:</b> as features derivadas do LLM melhoraram a predição em Crime
            (AUC de 0.55 para 0.76) e Thriller (de 0.63 para 0.72). O coeficiente mais expressivo
            foi <code>llm_closure</code> (qualidade do encerramento narrativo), com β = +1.29 em Crime.</p>
            <p><b>Justificativa da descontinuação:</b> apesar dos resultados promissores, esta
            abordagem foi descontinuada no pipeline final por duas razões. Primeiro, a dependência
            de uma API externa comercial limita a reprodutibilidade do estudo, uma vez que os
            resultados ficam condicionados à disponibilidade e ao custo do serviço. Segundo,
            os achados mais robustos da pesquisa, particularmente a assinatura emocional do gênero
            Thriller e o trade-off entre avaliação e bilheteira, são adequadamente capturados
            pelos grafos de transição e redes de personagens, sem necessidade de análise generativa.</p>

            <h3 style="color:{COLORS['green']}; margin-top:20px">4. Ridge Regression<span id="ridge-detail"></span> — Modelo Preditivo
            <span class="tag tag-green">ML treinado nos dados</span></h3>

            <p><b>O que é:</b> regressão linear com <b>penalização L2</b> (Ridge).
            A penalização controla a magnitude dos coeficientes para evitar overfitting —
            importante quando o número de features (até 29) é grande em relação
            à amostra (~1537 filmes).</p>

            <p><b>Instalação:</b> <code>pip install scikit-learn==1.8.0</code>.
            Classes utilizadas: <code>Ridge</code>, <code>StandardScaler</code>,
            <code>LeaveOneOut</code>, <code>KFold</code>, <code>r2_score</code>,
            <code>mean_absolute_error</code>.</p>

            <p><b>Implementação (arquivos <code>pipeline/11_baseline_models.py</code> a
            <code>pipeline/33_final_combined_analysis.py</code>):</b></p>
            <ol>
                <li><b>Padronização</b>: antes de treinar, cada feature é normalizada com
                <code>StandardScaler</code> (média=0, desvio=1). Isso é necessário porque
                as features têm escalas diferentes (autoloop vai de 0 a 1, entropia de 0 a 5).</li>
                <li><b>Configuração</b>: <code>Ridge(alpha=1.0, random_state=42)</code>.
                O alpha=1.0 é o padrão — testei outros valores mas a diferença era negligível.</li>
                <li><b>Validação cruzada</b>: no corpus pequeno (574 filmes), usei
                <b>Leave-One-Out</b> (LOO-CV) — treina em 573 filmes, testa em 1, repete 574 vezes.
                No corpus expandido (1537 filmes), passei para <b>5-fold CV</b>
                (<code>KFold(n_splits=5, shuffle=True, random_state=42)</code>)
                por eficiência computacional (LOO com 1537 filmes levava horas).</li>
                <li><b>Métricas</b>: R² (variância explicada), MAE (erro absoluto médio),
                Spearman ρ (correlação de ranking entre real e previsto).</li>
                <li><b>Comparação</b>: sempre comparo contra um <b>baseline</b> que usa apenas
                ano + 8 gêneros (one-hot). Se o modelo com features narrativas não supera
                esse baseline, as features não adicionam valor preditivo.</li>
            </ol>

            <p><b>Resultado:</b> baseline (year+genre) explica R²=0.167 do IMDb.
            Nenhuma combinação de features narrativas superou significativamente.
            Para Box Office, RoBERTa + Rede de Personagens atingiu R²=0.138 (+1.3%),
            o único modelo que adicionou valor incremental consistente.</p>
        </div>

        <!-- Por que redes de personagens -->
        <div class="intro-section">
            <h2 id="combinacao">Por Que Combinamos Grafos Emocionais com Redes de Personagens</h2>
            <p>A pesquisa começou com <b>duas linhas separadas</b>:</p>
            <ul>
                <li><b>Linha 1 — Redes de personagens:</b> investigar se a estrutura social
                do roteiro (quem interage com quem) se relaciona com o sucesso do filme.
                Os resultados iniciais foram fracos — a rede de personagens sozinha explicava
                pouco do IMDb.</li>
                <li><b>Linha 2 — Grafos emocionais:</b> investigar se a dinâmica emocional
                das cenas (como o sentimento evolui ao longo do filme) se relaciona com sucesso.
                Os sinais eram contextuais — fortes em Thriller, fracos globalmente.</li>
            </ul>
            <p>A <b>combinação das duas linhas</b> surgiu da hipótese de que o sucesso
            não depende apenas do <i>que o filme sente</i> (emoção das cenas) nem apenas de
            <i>quem interage</i> (rede social), mas de <b>quem sente o quê</b> —
            o cruzamento entre personagens e emoções.</p>
            <p><b>Como implementei a combinação:</b></p>
            <ol>
                <li>Para cada filme, extraí as <b>falas do protagonista</b> (personagem com
                mais interações na rede) cena a cena</li>
                <li>Passei essas falas pelo VADER e RoBERTa separadamente</li>
                <li>Calculei features como: tom médio do protagonista, % de falas positivas/negativas,
                volatilidade emocional, arco emocional (diferença entre último terço e início),
                e contraste protagonista vs restante do elenco</li>
                <li>Estas features foram combinadas com as métricas da rede (densidade, centralização,
                modularidade) e as métricas emocionais (autoloop, entropia, PageRank) num modelo
                Ridge com 5-fold CV</li>
            </ol>
            <p><b>Resultado da combinação:</b> para Box Office, o modelo RoBERTa + Rede
            superou todos os outros (R²=0.138 vs 0.125 do baseline). Para IMDb, a combinação
            não melhorou — mas revelou que <b>redes menos densas</b> (ρ=−0.12) e
            <b>protagonistas com tom mais positivo</b> (ρ=+0.12) são os sinais mais fortes
            entre as features de personagens.</p>
        </div>

        <!-- Filmes -->
        <div class="intro-section" style="background: transparent; border: none; padding: 0;">
            <h2 style="text-align: center; margin-bottom: 24px;">Análise por Gênero — 7 Filmes</h2><span id="filmes"></span>
        </div>

        {films_html}

        <!-- Resultados -->
        <div class="intro-section">
            <h2 id="resultados">Resultados Consolidados</h2>
            <p>Esta seção apresenta os principais achados da pesquisa, organizados em quatro partes:
            os resultados confirmados como robustos, os resultados que não se sustentaram,
            a comparação dos modelos preditivos, e a conclusão geral.</p>
            <p>Todos os resultados foram obtidos a partir de um corpus de <b>1537 filmes</b>,
            utilizando dois modelos de sentimento (VADER e RoBERTa), três fontes de features
            (grafos emocionais, rede de personagens e tom do protagonista), e validação cruzada
            de 5 partições com regressão Ridge.</p>

            <h3 style="color:{COLORS['green']}">Achados Fortes</h3><span id="achados"></span>
            <p>Os achados abaixo foram considerados robustos por atenderem a pelo menos três critérios:
            (1) significância estatística (p &lt; 0.05) após correção para múltiplos testes,
            (2) confirmação com dois modelos de sentimento independentes (VADER e RoBERTa), e
            (3) estabilidade ao triplicar o tamanho do corpus (de 574 para 1537 filmes).</p>

            <div class="finding finding-green">
                <b>Achado 1: O gênero Thriller apresenta assinatura emocional consistente.</b><br>
                Foram identificadas três métricas de grafo com correlação significativa com a nota IMDb
                em filmes de Thriller (n = 260):
                <ul style="margin: 8px 0 0 20px">
                    <li><b>Estabilidade emocional (autoloop)</b>: ρ = +0.25, p &lt; 0.001.
                    Thrillers melhor avaliados tendem a manter o estado emocional entre cenas consecutivas,
                    em vez de oscilar entre estados diferentes. Na prática, isso significa que filmes como
                    <i>No Country for Old Men</i> (autoloop = 0.99) mantêm um tom emocional coeso,
                    enquanto filmes com autoloop baixo alternam entre positivo e negativo sem consistência.</li>
                    <li><b>Centralidade negativa (PageRank do estado Negativo)</b>: ρ = −0.27, p &lt; 0.001.
                    Thrillers melhor avaliados apresentam menor dominância do estado negativo no fluxo emocional.
                    Isso não significa ausência de tensão, mas sim que a tensão é dosada e intercalada com
                    momentos de neutralidade ou positividade.</li>
                    <li><b>Previsibilidade (entropia)</b>: ρ = −0.25, p &lt; 0.001.
                    Thrillers melhor avaliados possuem transições emocionais mais previsíveis, ou seja,
                    seguem padrões reconhecíveis em vez de alternar caoticamente. Este achado é
                    particularmente relevante porque foi detectado apenas com o modelo RoBERTa.
                    Com VADER, a entropia não apresentava correlação significativa (ρ ≈ 0).</li>
                </ul>
            </div>

            <div class="finding finding-green">
                <b>Achado 2: Existe um trade-off entre qualidade percebida e sucesso comercial.</b><br>
                As mesmas características narrativas que favorecem a avaliação IMDb tendem a desfavorecer
                a bilheteira, e vice-versa. Concretamente:
                <ul style="margin: 8px 0 0 20px">
                    <li>Filmes com <b>maior estabilidade emocional</b> (autoloop alto) recebem
                    melhores notas IMDb, mas tendem a arrecadar menos.</li>
                    <li>Filmes com <b>protagonistas de tom mais positivo</b> recebem melhores notas,
                    mas protagonistas mais intensos e sombrios estão associados a maior bilheteira.</li>
                    <li>Filmes com <b>redes de personagens distribuídas</b> (elencos corais, sem um
                    protagonista dominante) tendem a ter maior bilheteira, enquanto redes centralizadas
                    (um protagonista forte) estão associadas a melhor avaliação.</li>
                </ul>
                <p style="margin-top: 8px">Este trade-off foi confirmado em múltiplas análises e com diferentes
                conjuntos de features, constituindo o achado mais consistente da pesquisa.</p>
            </div>

            <div class="finding finding-green">
                <b>Achado 3: O modelo RoBERTa produz sinais mais fortes que o VADER.</b><br>
                Ao comparar os dois modelos de sentimento, o RoBERTa apresentou correlações mais fortes
                com a nota IMDb em 6 dos 7 gêneros testados. Em Thriller, a correlação do autoloop
                passou de ρ = +0.15 (VADER) para ρ = +0.25 (RoBERTa), um aumento de 60%.
                Além disso, a entropia, que não apresentava sinal com VADER, tornou-se uma das
                métricas mais informativas com RoBERTa. Estes resultados indicam que a qualidade
                do modelo de sentimento tem impacto direto na capacidade de extração de padrões
                dos grafos emocionais.
            </div>

            <div class="finding finding-green">
                <b>Achado 4: Features de rede de personagens contribuem para a predição de bilheteira.</b><br>
                As métricas da rede social de personagens (densidade, modularidade, centralização)
                mostraram-se mais informativas que as métricas emocionais para prever a bilheteira.
                A combinação de features emocionais (RoBERTa) com features de rede de personagens
                produziu o único modelo que superou consistentemente o baseline (ver seção "Melhor Modelo"
                abaixo).
            </div>

            <h3 style="color:{COLORS['red']}; margin-top: 28px;">O que NÃO Funciona</h3><span id="nao-funciona"></span>
            <p>Tão importante quanto os achados positivos são os resultados negativos, que delimitam
            o que este tipo de análise <b>não</b> consegue fazer.</p>

            <div class="finding finding-red">
                <b>Predição global de IMDb não é viável com features narrativas.</b><br>
                Nenhuma combinação de features (emocionais, de rede, do protagonista, ou todas juntas)
                superou significativamente um modelo baseline que utiliza apenas o ano de lançamento
                e o gênero do filme. O baseline já explica R² = 0.165 (16.5%) da variação nas notas IMDb.
                Os modelos com features narrativas atingem no máximo R² = 0.166 (16.6%), um ganho
                de apenas 0.1 ponto percentual, estatisticamente irrelevante. Isso indica que o sinal
                narrativo é <b>contextual</b> (aparece em gêneros específicos como Thriller) e não
                <b>universal</b> (não funciona para todos os filmes).
            </div>

            <div class="finding finding-red">
                <b>O gênero Horror apresentou resultados instáveis.</b><br>
                Na análise inicial com 69 filmes de Horror, observaram-se correlações fortes
                (autoloop ρ = +0.35, centralização ρ = +0.29). Porém, ao triplicar o corpus para
                199 filmes de Horror, esses efeitos desapareceram (autoloop ρ = +0.08, p = 0.26).
                Este é um exemplo clássico de <b>overfitting</b>: com amostras pequenas, o ruído
                estatístico pode simular padrões que não existem na população real. A triplicação
                do corpus revelou que os achados iniciais em Horror eram artefatos do tamanho da amostra.
            </div>

            <h3 style="color:{COLORS['accent']}; margin-top: 28px;">Melhor Modelo por Variável de Sucesso</h3>
            <span id="modelos"></span>
            <p>Foram testadas 7 configurações de modelo para cada variável de sucesso. Os modelos
            são comparados pelo R² obtido em validação cruzada de 5 partições (5-fold CV),
            que estima a capacidade do modelo de generalizar para filmes que não foram usados no
            treinamento.</p>

            <table>
                <tr>
                    <td style="min-width: 120px"><b>Nota IMDb</b><br>
                    <span style="font-size:11px; color:{COLORS['dim']}">Qualidade percebida<br>pelo público</span></td>
                    <td>
                    <p><b>Melhor modelo: Rede de personagens</b> (R² = 0.166)</p>
                    <p>O R² de 0.166 indica que o modelo explica 16.6% da variação nas notas IMDb.
                    Para contextualizar: o baseline, que utiliza apenas o ano de lançamento e o gênero
                    do filme (sem nenhuma informação do roteiro), já atinge R² = 0.165.
                    A diferença de 0.1 ponto percentual é estatisticamente irrelevante.</p>
                    <p><b>Interpretação:</b> a estrutura narrativa do roteiro, conforme medida
                    neste estudo, não adiciona poder explicativo relevante para prever como o público
                    avaliará o filme de forma global. Porém, dentro de gêneros específicos (como Thriller),
                    as correlações individuais das métricas são significativas e informativas.</p>
                    </td>
                </tr>
                <tr>
                    <td><b>Bilheteira</b><br>
                    <span style="font-size:11px; color:{COLORS['dim']}">Receita bruta nos<br>EUA (Box Office)</span></td>
                    <td>
                    <p><b style="color:{COLORS['green']}">Melhor modelo: RoBERTa + Rede de Personagens</b>
                    (R² = 0.138)</p>
                    <p>O R² de 0.138 indica que o modelo explica 13.8% da variação na bilheteira
                    (em escala logarítmica). O baseline atinge R² = 0.125 (12.5%). O ganho de
                    <b>1.3 pontos percentuais</b> pode parecer pequeno em termos absolutos, mas
                    é relevante por dois motivos:</p>
                    <ul>
                        <li>É o <b>único modelo</b> que superou o baseline de forma consistente
                        ao longo de todas as partições da validação cruzada.</li>
                        <li>O ganho provém exclusivamente de informações extraídas do roteiro,
                        demonstrando que a estrutura narrativa contém sinal informativo para a
                        bilheteira que não é capturado apenas por ano e gênero.</li>
                    </ul>
                    <p>As features utilizadas neste modelo são: 5 métricas emocionais do grafo RoBERTa
                    (estabilidade, centralidade negativa, centralidade positiva, entropia e frequência
                    de transições de recuperação), 9 métricas da rede de personagens (densidade,
                    centralização, intermediação, clustering, modularidade, entropia de interações,
                    mudança de elenco, dominância e presença do protagonista), o ano de lançamento
                    e 8 indicadores de gênero.</p>
                    </td>
                </tr>
            </table>

            <h3 style="color:#0366d6; margin-top: 28px;" id="conclusao">Conclusão</h3>
            <p>A pergunta central desta pesquisa era: <i>pode a estrutura narrativa de um roteiro
            cinematográfico prever o sucesso de um filme?</i></p>
            <p>A resposta, com base na análise de 1537 filmes, é <b>parcialmente sim, com
            ressalvas importantes</b>:</p>
            <ol>
                <li><b>A predição global não é viável.</b> Nenhum modelo baseado em features narrativas
                superou significativamente um baseline simples para prever a nota IMDb de forma geral.
                A nota que o público atribui a um filme depende de fatores que vão além do roteiro
                (elenco, direção, marketing, contexto cultural).</li>
                <li><b>Existem padrões contextuais genuínos.</b> Dentro do gênero Thriller, a estrutura
                emocional do roteiro apresenta correlações robustas com a avaliação. Thrillers bem
                avaliados tendem a manter estabilidade emocional, evitar excesso de negatividade
                e apresentar transições previsíveis.</li>
                <li><b>Qualidade percebida e sucesso comercial respondem a padrões opostos.</b>
                O que agrada o público avaliador do IMDb (controle emocional, protagonista positivo,
                rede centralizada) é o oposto do que gera bilheteira (intensidade emocional,
                protagonista sombrio, elencos corais).</li>
                <li><b>A escolha do modelo de sentimento importa.</b> O RoBERTa, por compreender
                contexto, produz sinais mais fortes que o VADER. Métricas que eram invisíveis com
                VADER tornaram-se significativas com RoBERTa, reforçando a importância de utilizar
                modelos de sentimento baseados em redes neurais em estudos deste tipo.</li>
                <li><b>A combinação de emoção e personagens é mais informativa que cada dimensão isolada.</b>
                Para a bilheteira, o modelo que combina grafos emocionais (RoBERTa) com redes de
                personagens foi o único a superar o baseline de forma consistente.</li>
            </ol>
        </div>

        <div class="footer">
            <p>Dissertação de Mestrado — Análise de Roteiros Cinematográficos<br>
            Corpus: 1537 filmes • Pipeline: VADER + RoBERTa + Grafos + Personagens<br>
            Gerado automaticamente</p>
        </div>

    </div>
</body>
</html>"""

    return html


def main():
    ensure_dirs()
    logging.info("Gerando relatório HTML...")

    # Processar filmes
    film_results = {}
    for genre, title in FILMS.items():
        logging.info("Processando %s (%s)...", title, genre)
        data = process_film(title)
        film_results[genre] = data

    # Gerar HTML
    logging.info("Montando HTML...")
    html = generate_html(film_results)

    out_path = RESULTS / "relatorio_dissertacao.html"
    out_path.write_text(html, encoding="utf-8")
    logging.info("Relatório salvo: %s", out_path)
    logging.info("Abra no navegador: file://%s", out_path.resolve())


if __name__ == "__main__":
    main()
