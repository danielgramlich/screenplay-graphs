"""Segmentacao de roteiros em cenas via regex de sluglines."""

import json
import logging
import re

from src.utils import RAW, SCENES, ensure_dirs, sanitize_title

logger = logging.getLogger(__name__)

SLUGLINE_RE = re.compile(
    r"^\s*(INT\.|EXT\.|INT/EXT\.|INT\.\/EXT\.|I/E\.)\s*.+",
    re.IGNORECASE | re.MULTILINE,
)

MIN_SCENE_CHARS = 20


def parse_screenplay(text: str) -> list[dict]:
    """Segmenta texto de roteiro em cenas usando sluglines.

    Returns:
        Lista de dicts: {scene_idx, heading, text, char_count}
    """
    matches = list(SLUGLINE_RE.finditer(text))
    if not matches:
        return []

    scenes = []
    for i, match in enumerate(matches):
        heading = match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        scene_text = text[start:end].strip()

        scenes.append({
            "scene_idx": i,
            "heading": heading,
            "text": scene_text,
            "char_count": len(scene_text),
        })

    # Mergear cenas muito curtas com a proxima
    merged = []
    buffer = None
    for scene in scenes:
        if buffer is not None:
            scene = {
                **scene,
                "text": buffer["text"] + "\n" + scene["text"],
                "char_count": buffer["char_count"] + scene["char_count"],
                "heading": buffer["heading"],
                "scene_idx": buffer["scene_idx"],
            }
            buffer = None

        if scene["char_count"] < MIN_SCENE_CHARS:
            buffer = scene
        else:
            merged.append(scene)

    if buffer:
        if merged:
            merged[-1]["text"] += "\n" + buffer["text"]
            merged[-1]["char_count"] += buffer["char_count"]
        else:
            merged.append(buffer)

    # Re-indexar
    for i, s in enumerate(merged):
        s["scene_idx"] = i

    return merged


def parse_all_screenplays() -> dict[str, list[dict]]:
    """Parseia todos os roteiros em data/raw/. Retorna dict titulo->cenas."""
    ensure_dirs()
    results = {}
    raw_files = sorted(RAW.glob("*.txt"))

    if not raw_files:
        logger.warning("Nenhum roteiro encontrado em %s", RAW)
        return results

    for path in raw_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        scenes = parse_screenplay(text)
        title = path.stem

        if len(scenes) < 10:
            logger.warning(
                "%s: apenas %d cenas encontradas (possivel problema de formato)",
                title, len(scenes),
            )
        elif len(scenes) > 300:
            logger.warning("%s: %d cenas (numero incomum)", title, len(scenes))

        # Salvar JSON
        out_path = SCENES / f"{title}.json"
        out_path.write_text(
            json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results[title] = scenes

    logger.info(
        "Parseados %d roteiros, total de %d cenas",
        len(results),
        sum(len(s) for s in results.values()),
    )
    return results
