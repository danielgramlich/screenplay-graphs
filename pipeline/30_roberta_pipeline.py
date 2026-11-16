#!/usr/bin/env python3
"""Etapa 30: Pipeline RoBERTa — anotar emoções, construir grafos, extrair métricas.

Roda o pipeline emocional completo com RoBERTa em vez de VADER.
Salva em pasta separada para não perder dados VADER.
Compara resultados no final.
"""

import sys
import json
import logging
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from src.utils import INTERMEDIATE, FEATURES, METADATA, RESULTS, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Pasta separada para dados RoBERTa
ROBERTA_DIR = FEATURES / "roberta"
ROBERTA_DIR.mkdir(parents=True, exist_ok=True)

SCENES_DIR = INTERMEDIATE / "scenes"


def to_v5(c):
    if c >= 0.3: return "v5_high"
    if c >= 0.05: return "v5_pos"
    if c > -0.05: return "v5_neu"
    if c > -0.3: return "v5_neg"
    return "v5_low"

V5_STATES = ["v5_high", "v5_pos", "v5_neu", "v5_neg", "v5_low"]

def to_v3(c):
    if c > 0.05: return "pos"
    if c < -0.05: return "neg"
    return "neu"

V3_STATES = ["pos", "neu", "neg"]


def build_tm(seq, states):
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    T = np.zeros((n, n))
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a in idx and b in idx:
            T[idx[a]][idx[b]] += 1
    return T


def graph_metrics(T, states, prefix):
    """Mesmas métricas do pipeline VADER."""
    import networkx as nx
    total = T.sum()
    if total == 0:
        return {}
    n = len(states)
    P = T / total
    nz = P[P > 0]
    result = {
        f"{prefix}_entropy": float(-np.sum(nz * np.log2(nz))),
        f"{prefix}_autoloop": float(np.trace(T) / total),
    }
    recip = sum(min(T[i][j], T[j][i]) for i in range(n) for j in range(n) if i != j)
    denom = total - np.trace(T)
    result[f"{prefix}_reciprocity"] = float(recip / denom) if denom > 0 else 0.0

    G = nx.DiGraph()
    for s in states: G.add_node(s)
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            if T[i][j] > 0: G.add_edge(s1, s2, weight=T[i][j])
    try:
        pr = nx.pagerank(G, weight="weight")
    except:
        pr = {s: 1.0/n for s in states}
    for s in states:
        result[f"{prefix}_pr_{s}"] = float(pr.get(s, 0))

    return result


def process_film(title, scenes, roberta_batch_func):
    """Processa um filme: emoção → discretização → grafos → métricas."""
    texts = [sc.get("text", "") for sc in scenes]
    results = roberta_batch_func(texts)
    compounds = [r["roberta_compound"] for r in results]

    v = np.array(compounds)
    n = len(v)

    if n < 6:
        return None

    feats = {"title": title}

    # V5 (filme inteiro)
    v5_seq = [to_v5(c) for c in compounds]
    T5 = build_tm(v5_seq, V5_STATES)
    feats.update(graph_metrics(T5, V5_STATES, "gv5"))

    # V3 (filme inteiro)
    v3_seq = [to_v3(c) for c in compounds]
    T3 = build_tm(v3_seq, V3_STATES)
    feats.update(graph_metrics(T3, V3_STATES, "gv3"))

    # V3 último terço
    act3 = compounds[2 * n // 3:]
    if len(act3) >= 4:
        v3a3_seq = [to_v3(c) for c in act3]
        T3a3 = build_tm(v3a3_seq, V3_STATES)
        feats.update(graph_metrics(T3a3, V3_STATES, "gv3a3"))

    # Stats da curva (para comparação)
    feats["roberta_mean"] = float(np.mean(v))
    feats["roberta_std"] = float(np.std(v))
    feats["roberta_min"] = float(np.min(v))
    feats["n_scenes"] = n

    return feats


def main():
    ensure_dirs()

    # Lazy load do modelo RoBERTa (versão batch)
    from src.sentiment_roberta import roberta_scenes_batch

    # Forçar carregamento do modelo
    logging.info("Carregando modelo RoBERTa...")
    _ = roberta_scenes_batch(["test"])
    logging.info("Modelo carregado.")

    # Processar todos os filmes que têm cenas parseadas
    scene_files = sorted(SCENES_DIR.glob("*.json"))
    out_path = ROBERTA_DIR / "roberta_graph_features.csv"

    # Retomada: carregar progresso anterior
    done_titles = set()
    rows = []
    if out_path.exists():
        old = pd.read_csv(out_path)
        done_titles = set(old["title"])
        rows = old.to_dict("records")
        logging.info("Retomando: %d filmes já processados", len(done_titles))

    remaining = [p for p in scene_files if p.stem not in done_titles]
    logging.info("Filmes a processar: %d (restantes: %d)", len(scene_files), len(remaining))

    success = len(done_titles)
    t0 = time.time()

    for i, path in enumerate(remaining):
        title = path.stem
        try:
            scenes = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        feats = process_film(title, scenes, roberta_scenes_batch)
        if feats:
            rows.append(feats)
            success += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate / 60
            logging.info("Progresso: %d/%d (total_sucesso=%d, %.1f filmes/min, ETA %.0f min)",
                         i + 1, len(remaining), success, rate * 60, eta)

        # Salvamento incremental a cada 100 filmes
        if (i + 1) % 100 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            logging.info("Salvo incrementalmente: %d filmes", len(rows))

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logging.info("Salvo final: %d filmes → %s", len(df), out_path)

    # ═══════════════════════════════════════════════════════
    # COMPARAÇÃO VADER vs RoBERTa
    # ═══════════════════════════════════════════════════════
    vader_path = FEATURES / "emotional_graphs" / "refined_graph_features.csv"
    if vader_path.exists():
        vader = pd.read_csv(vader_path)
        meta = pd.read_csv(METADATA / "film_list.csv")
        meta = meta[meta["matched"] == True]

        # Merge ambos com metadata
        v_merged = vader.merge(meta[["source_file", "imdb_rating"]],
                                left_on="title", right_on="source_file", how="inner")
        r_merged = df.merge(meta[["source_file", "imdb_rating"]],
                             left_on="title", right_on="source_file", how="inner")

        both = v_merged.merge(r_merged, on="title", suffixes=("_vader", "_roberta"))
        both = both[both["imdb_rating_vader"].notna()]

        out = []
        out.append("=" * 70)
        out.append("  COMPARAÇÃO VADER vs RoBERTa")
        out.append("=" * 70)
        out.append(f"\n  VADER: {len(v_merged)} filmes")
        out.append(f"  RoBERTa: {len(r_merged)} filmes")
        out.append(f"  Em comum (com IMDb): {len(both)}")

        # Correlações top features vs IMDb
        compare_feats = ["gv5_autoloop", "gv3_pr_neg", "gv5_pr_v5_pos", "gv3_entropy"]
        out.append(f"\n  {'Feature':<25s} {'VADER ρ':>10s} {'RoBERTa ρ':>10s} {'Melhor':>8s}")
        out.append(f"  {'-'*58}")

        for feat in compare_feats:
            v_col = f"{feat}_vader"
            r_col = f"{feat}_roberta"
            if v_col in both.columns and r_col in both.columns:
                v_valid = both[[v_col, "imdb_rating_vader"]].dropna()
                r_valid = both[[r_col, "imdb_rating_vader"]].dropna()

                rho_v, p_v = stats.spearmanr(v_valid[v_col], v_valid["imdb_rating_vader"])
                rho_r, p_r = stats.spearmanr(r_valid[r_col], r_valid["imdb_rating_vader"])

                melhor = "RoBERTa" if abs(rho_r) > abs(rho_v) else "VADER"
                out.append(f"  {feat:<25s} {rho_v:>+10.3f} {rho_r:>+10.3f} {melhor:>8s}")

        # Correlação entre VADER e RoBERTa
        out.append(f"\n  Concordância VADER ↔ RoBERTa:")
        for feat in compare_feats:
            v_col = f"{feat}_vader"
            r_col = f"{feat}_roberta"
            if v_col in both.columns and r_col in both.columns:
                valid = both[[v_col, r_col]].dropna()
                rho, p = stats.pearsonr(valid[v_col], valid[r_col])
                out.append(f"    {feat:<25s}  r={rho:.3f} (p={p:.4f})")

        report = "\n".join(out)
        print(report)
        (RESULTS / "vader_vs_roberta_report.txt").write_text(report, encoding="utf-8")
    else:
        logging.info("VADER features não encontradas — pulando comparação")

    total_time = (time.time() - t0) / 60
    logging.info("Tempo total: %.1f minutos", total_time)


if __name__ == "__main__":
    main()
