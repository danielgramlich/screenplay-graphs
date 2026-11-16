
#!/usr/bin/env python3
"""
sbsi2026_run.py

Unified runner:
- reads one screenplay (.txt),
- extracts scenes and speaker turns once,
- runs multiple baseline methods,
- writes results into separate folders.

It can also run a single method (to integrate with a subprocess-based experiment runner):
  python3 sbsi2026_run.py --input <script.txt> --output <outdir> --method <method_id>

Outputs:
- For scene-aligned methods: <outdir>/scene_metrics.csv with columns W,D,C,L,Delta.
- For film-level method: <outdir>/film_metrics.csv (single row) + edge list.

For "all":
  <out_base>/
    _shared/  (scene list, character mapping, metadata)
    <method_id>/
      scene_metrics.csv  (or film_metrics.csv)
      edges.csv
      method_metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from sbsi_core import (
    ExtractionConfig,
    STRING_SIM_BACKEND,

    Scene,
    compute_cast_jaccard_series,
    compute_cast_sets,
    discover_and_validate_characters,
    extract_speaker_turns,
    file_md5,
    split_script_into_scenes,
)
from methods import (
    MethodSpec,
    aggregate_over_all_scenes,
    get_default_methods,
    graphs_copresence_binary_per_scene,
    graphs_copresence_min_turns_per_scene,
    graphs_cumulative,
    graphs_event_decay_end_of_scene,
    graphs_multilayer_collapse,
    graphs_sliding_window,
    graphs_turn_adjacency_per_scene,
    graphs_turn_window_k_per_scene,
)
from metrics import (
    MetricRow,
    compute_metrics_for_scene_aligned_series,
    compute_metrics_for_single_graph,
)
from io_utils import (
    ensure_dir,
    runtime_environment_info,
    write_csv,
    write_edge_list_csv,
    write_json,
    write_metric_rows_csv,
)


CODE_VERSION = "sbsi2026_temporal_networks_v1_1"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_slug(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))[:80] or "script"


def _write_shared_artifacts(
    out_base: Path,
    script_path: Path,
    scenes: Sequence[Scene],
    cfg: ExtractionConfig,
    name_mapping: Dict[str, str],
    canonical_set: Sequence[str],
    raw_counts: Any,
    speakers_per_scene: Sequence[Sequence[str]],
) -> None:
    shared = out_base / "_shared"
    ensure_dir(shared)

    # scenes.csv
    scene_rows: List[Dict[str, Any]] = []
    for sc, speakers in zip(scenes, speakers_per_scene):
        scene_rows.append(
            {
                "scene": sc.index,
                "heading": sc.heading,
                "n_turns": len(speakers),
                "cast_size": len(set(speakers)),
            }
        )
    write_csv(shared / "scenes.csv", scene_rows, fieldnames=["scene", "heading", "n_turns", "cast_size"])

    # name_mapping.csv (raw -> canonical)
    mapping_rows = [{"raw": k, "canonical": v} for k, v in sorted(name_mapping.items(), key=lambda kv: (kv[1], kv[0]))]
    write_csv(shared / "name_mapping.csv", mapping_rows, fieldnames=["raw", "canonical"])

    # character_counts.csv (raw counts before clustering)
    counts_rows = [{"raw": k, "count": int(v)} for k, v in raw_counts.most_common()]
    write_csv(shared / "character_counts.csv", counts_rows, fieldnames=["raw", "count"])

    # metadata.json
    meta = {
        "code_version": CODE_VERSION,
        "input_script": script_path.name,
        "input_md5": file_md5(str(script_path)),
        "num_scenes": len(scenes),
        "num_canonical_characters": len(set(canonical_set)),
        "extraction_config": cfg.__dict__,
        "runtime": runtime_environment_info(),
        "string_similarity_backend": STRING_SIM_BACKEND,
    }
    write_json(shared / "run_metadata.json", meta)


def _method_outdir(out_base: Path, method_id: str, mode: str) -> Path:
    # mode: "all" or "single"
    if mode == "single":
        return out_base
    return out_base / method_id


def _run_method(
    method: MethodSpec,
    *,
    outdir: Path,
    scenes: Sequence[Scene],
    speakers_per_scene: Sequence[Sequence[str]],
    turn_counts_per_scene,
    cast_jaccard: Sequence[float],
    tang_window_size: int,
    half_life_events: int,
) -> None:
    ensure_dir(outdir)

    # Build graphs according to method id
    graphs: Optional[List[nx.Graph]] = None
    film_graph: Optional[nx.Graph] = None
    is_scene_aligned = True

    # Base per-scene graphs (adjacency, copresence, window) reused by several methods.
    if method.id == "turn_adj_scene":
        graphs = graphs_turn_adjacency_per_scene(speakers_per_scene)

    elif method.id == "copresence_min_turns_scene":
        graphs = graphs_copresence_min_turns_per_scene(turn_counts_per_scene)

    elif method.id == "turn_window_k2_scene":
        graphs = graphs_turn_window_k_per_scene(speakers_per_scene, k=2)

    elif method.id == "aggregate_film_copresence_min_turns":
        # Aggregate the scene-level copresence rule into one film graph.
        scene_graphs = graphs_copresence_min_turns_per_scene(turn_counts_per_scene)
        film_graph = aggregate_over_all_scenes(scene_graphs)
        is_scene_aligned = False

    elif method.id == "cumulative_turn_adj":
        base_scene_graphs = graphs_turn_adjacency_per_scene(speakers_per_scene)
        graphs = graphs_cumulative(base_scene_graphs)

    elif method.id.startswith("tang_window_w"):
        base_scene_graphs = graphs_turn_adjacency_per_scene(speakers_per_scene)
        graphs = graphs_sliding_window(base_scene_graphs, window_size=tang_window_size)

    elif method.id.startswith("event_decay_half"):
        graphs = graphs_event_decay_end_of_scene(speakers_per_scene, half_life_events=half_life_events)

    elif method.id == "multilayer_collapse_dialogue+copresence":
        layer_dialogue = graphs_turn_adjacency_per_scene(speakers_per_scene)
        layer_copres = graphs_copresence_min_turns_per_scene(turn_counts_per_scene)
        graphs = graphs_multilayer_collapse(layer_dialogue, layer_copres, beta_a=1.0, beta_b=1.0)

    else:
        raise ValueError(f"Unknown method id: {method.id}")

    # Save method metadata
    method_meta = {
        "method": {
            "id": method.id,
            "label": method.label,
            "family": method.family,
            "temporal_unit": method.temporal_unit,
            "description": method.description,
            "params": method.params,
        }
    }
    write_json(outdir / "method_metadata.json", method_meta)

    # Compute + save metrics
    if is_scene_aligned:
        assert graphs is not None
        metric_rows = compute_metrics_for_scene_aligned_series(graphs, cast_jaccard)
        write_metric_rows_csv(outdir / "scene_metrics.csv", metric_rows)

        # Also export an aggregated edge list for transparency (sum of weights over all snapshots)
        agg = aggregate_over_all_scenes(graphs)
        write_edge_list_csv(outdir / "edges_aggregated.csv", agg)

    else:
        assert film_graph is not None
        metric_row = compute_metrics_for_single_graph(film_graph)
        write_metric_rows_csv(outdir / "film_metrics.csv", [metric_row])
        write_edge_list_csv(outdir / "edges.csv", film_graph)


def main() -> None:
    ap = argparse.ArgumentParser(description="SBSI 2026 – Temporal character network runner (multi-method).")

    ap.add_argument("input", nargs="?", default=None, help="Path to script .txt (positional).")
    ap.add_argument("--input", dest="input_flag", default=None, help="Path to script .txt.")
    ap.add_argument("--output", default="output_sbsi2026", help="Output folder (base).")
    ap.add_argument("--flat-output", action="store_true", help="Write directly into --output (no script subfolder). Useful for subprocess runners.")

    ap.add_argument(
        "--method",
        default="all",
        help="Method id to run, or 'all' to run the full suite.",
    )

    ap.add_argument("--tang-window", type=int, default=5, help="Scene window size for Tang-style snapshots.")
    ap.add_argument("--half-life-events", type=int, default=50, help="Half-life (in events) for event-decay method.")

    ap.add_argument("--legacy-mode", action="store_true", help="Use legacy character-line heuristics (less strict).")
    ap.add_argument("--min-appearances", type=int, default=2, help="Minimum times a character cue must appear.")
    ap.add_argument("--max-name-words", type=int, default=5, help="Max words in a character cue line.")
    ap.add_argument("--similarity-threshold", type=float, default=0.7, help="Levenshtein ratio threshold for clustering.")

    args = ap.parse_args()

    in_path_str = args.input_flag or args.input
    if not in_path_str:
        ap.error("You must provide the script path either as positional or via --input.")
    script_path = Path(in_path_str).expanduser().resolve()
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    out_base = Path(args.output).expanduser().resolve()
    script_slug = _safe_slug(script_path.stem)
    if not args.flat_output:
        # Default: isolate outputs per script to avoid collisions.
        out_base = out_base / script_slug
    ensure_dir(out_base)

    cfg = ExtractionConfig(
        character_max_words=int(args.max_name_words),
        similarity_threshold=float(args.similarity_threshold),
        minimum_appearances=int(args.min_appearances),
        legacy_mode=bool(args.legacy_mode),
    )

    raw_text = _read_text(script_path)

    # Shared extraction
    scenes = split_script_into_scenes(raw_text)
    name_mapping, canonical_set, raw_counts = discover_and_validate_characters(raw_text, cfg)

    speakers_per_scene = []
    turn_counts_per_scene = []
    for sc in scenes:
        speakers, counts = extract_speaker_turns(sc.text, name_mapping, canonical_set, cfg)
        speakers_per_scene.append(speakers)
        turn_counts_per_scene.append(counts)

    cast_sets = compute_cast_sets(speakers_per_scene)
    cast_jaccard = compute_cast_jaccard_series(cast_sets)

    # Write shared artifacts (scene list, mapping, metadata)
    _write_shared_artifacts(
        out_base=out_base,
        script_path=script_path,
        scenes=scenes,
        cfg=cfg,
        name_mapping=name_mapping,
        canonical_set=sorted(set(canonical_set)),
        raw_counts=raw_counts,
        speakers_per_scene=speakers_per_scene,
    )

    # Build methods list
    all_methods = get_default_methods(tang_window_size=int(args.tang_window), half_life_events=int(args.half_life_events))
    if args.method == "all":
        methods_to_run = all_methods
        mode = "all"
    else:
        methods_to_run = [m for m in all_methods if m.id == args.method]
        if not methods_to_run:
            known = ", ".join(m.id for m in all_methods)
            raise ValueError(f"Unknown --method '{args.method}'. Known: {known}")
        mode = "single"

    # Run
    for m in methods_to_run:
        outdir = _method_outdir(out_base, m.id, mode)
        _run_method(
            m,
            outdir=outdir,
            scenes=scenes,
            speakers_per_scene=speakers_per_scene,
            turn_counts_per_scene=turn_counts_per_scene,
            cast_jaccard=cast_jaccard,
            tang_window_size=int(args.tang_window),
            half_life_events=int(args.half_life_events),
        )

    print(f"Done. Outputs at: {out_base}")


if __name__ == "__main__":
    main()
