
"""
io_utils.py

Filesystem I/O helpers for writing outputs in a reviewer-friendly, reproducible way.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import networkx as nx

from metrics import MetricRow


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_metric_rows_csv(path: Path, metric_rows: Sequence[MetricRow]) -> None:
    rows = []
    for r in metric_rows:
        rows.append(
            {
                "t": r.t,
                "W": r.W,
                "D": r.D,
                "C": r.C,
                "L": r.L,
                "Delta": r.Delta,
                "n_nodes": r.n_nodes,
                "n_edges": r.n_edges,
            }
        )
    write_csv(path, rows, fieldnames=["t", "W", "D", "C", "L", "Delta", "n_nodes", "n_edges"])


def write_edge_list_csv(path: Path, G: nx.Graph) -> None:
    rows = []
    for u, v, data in sorted(G.edges(data=True), key=lambda x: (x[0], x[1])):
        rows.append({"u": u, "v": v, "weight": float(data.get("weight", 1.0))})
    write_csv(path, rows, fieldnames=["u", "v", "weight"])


def runtime_environment_info() -> Dict[str, Any]:
    """
    Capture versions for reproducibility logs.
    """
    try:
        import numpy as np  # noqa
        numpy_version = np.__version__
    except Exception:
        numpy_version = None

    try:
        import Levenshtein  # noqa
        levenshtein_version = getattr(Levenshtein, "__version__", None)
    except Exception:
        levenshtein_version = None

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "networkx_version": nx.__version__,
        "numpy_version": numpy_version,
        "python_Levenshtein_version": levenshtein_version,
    }
