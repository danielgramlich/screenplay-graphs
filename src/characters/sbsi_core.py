
"""
sbsi_core.py

Core parsing and normalization utilities for extracting:
- scenes (via sluglines INT/EXT),
- canonical character names,
- per-scene speaker turn sequences.

Design goals (for SBSI/NIRE + reviewer fidelity):
- deterministic outputs given (script text, parameters),
- explicit heuristics and parameters exported to metadata,
- shared parsing across all baseline methods to ensure fairness.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple, Optional

import numpy as np
import networkx as nx
# Optional dependency: python-Levenshtein (fast, true Levenshtein).
# If unavailable, we fall back to difflib.SequenceMatcher (still deterministic).
try:
    from Levenshtein import ratio as levenshtein_ratio  # type: ignore
    _STRING_SIM_BACKEND = "python-Levenshtein"
except Exception:  # pragma: no cover
    from difflib import SequenceMatcher
    def levenshtein_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()
    _STRING_SIM_BACKEND = "difflib"



# ==========================================================
#   CONFIG
# ==========================================================

@dataclass(frozen=True)
class ExtractionConfig:
    """Parameters controlling character discovery and normalization."""
    character_max_words: int = 5
    similarity_threshold: float = 0.7
    minimum_appearances: int = 2

    # If True, apply a more permissive legacy filter close to the older code.
    # If False, use stricter slugline/transition exclusion rules (recommended).
    legacy_mode: bool = False


@dataclass(frozen=True)
class Scene:
    """A scene extracted from a script."""
    index: int  # 1-based
    heading: str
    text: str   # full raw scene block (including heading)
    body: str   # scene without heading line


# ==========================================================
#   BASIC HELPERS
# ==========================================================

_SLUGLINE_RE = re.compile(r"^\s*(?:INT|EXT)(?:\./|/\.|\.|/)?(?:EXT|INT)?[\.\s/]", re.IGNORECASE)
_TRANSITION_RE = re.compile(
    r"^\s*(?:CUT TO:|DISSOLVE TO:|FADE IN:|FADE OUT:|SMASH CUT TO:|MATCH CUT TO:|WIPE TO:)\s*$",
    re.IGNORECASE,
)

# A conservative "ALL CAPS" detector for screenplay lines.
# Accepts digits and some punctuation frequently present in character names.
_ALLCAPS_LINE_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \.\'\-(),/]*$")

# Disallow colon inside character names; colons strongly suggest sluglines ("INT:") or metadata.
_HAS_COLON_RE = re.compile(r":")


def file_md5(path: str) -> str:
    """Compute MD5 of an input file for reproducibility logs."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def normalize_whitespace(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def is_slugline(line: str) -> bool:
    return bool(_SLUGLINE_RE.match(line.strip()))


def is_transition(line: str) -> bool:
    return bool(_TRANSITION_RE.match(line.strip()))


def is_allcaps_line(line: str) -> bool:
    """True if line looks like a typical ALL CAPS screenplay line."""
    s = line.strip()
    if not s:
        return False
    # Fast path
    if not _ALLCAPS_LINE_RE.match(s):
        return False
    # Ensure there is at least one letter (avoid lines of "----" or "123")
    return any("A" <= ch <= "Z" for ch in s)


# Patterns of cues that should NOT be treated as character speakers
# (voicemails, recordings, broadcast voices, narrators, anonymous off-screen).
_BLACKLIST_PATTERNS = [
    re.compile(r"\bVOICEMAIL\b",         re.IGNORECASE),
    re.compile(r"\bANSWERING MACHINE\b", re.IGNORECASE),
    re.compile(r"\bMESSAGE\b",           re.IGNORECASE),
    re.compile(r"\bRECORDING\b",         re.IGNORECASE),
    re.compile(r"\bRADIO\b",             re.IGNORECASE),
    re.compile(r"\bTV\b",                re.IGNORECASE),
    re.compile(r"\bNARRATOR\b",          re.IGNORECASE),
    re.compile(r"\bP\.?A\.? SYSTEM\b",   re.IGNORECASE),
    re.compile(r"\bINTERCOM\b",          re.IGNORECASE),
    re.compile(r"\bANSWERING\b",         re.IGNORECASE),
]


def is_blacklisted_character(name: str) -> bool:
    """True if the name matches patterns we want to exclude (voicemail, TV, narrator, etc.)."""
    for pat in _BLACKLIST_PATTERNS:
        if pat.search(name):
            return True
    return False


def clean_and_normalize_special_cases(name: str) -> str:
    """
    Remove stage directions and voice markers from character names.

    Examples removed:
      - (CONT'D), (O.S.), (V.O.), (ON PHONE) — including parentheticals
        that appear in the MIDDLE of the cue (e.g., "FERNANDO WOOD (CONT'D) ROBERT LATHAM").
      - "'S VOICE", etc.
    """
    name = name.strip()
    # Remove ANY parenthetical notes (not just trailing).
    # Handles "FERNANDO WOOD (CONT'D) ROBERT LATHAM" -> "FERNANDO WOOD ROBERT LATHAM"
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    # Remove "'S VOICE"
    name = re.sub(r"'S VOICE\b", "", name, flags=re.IGNORECASE)
    # Normalize internal whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def decompose_compound_cue(cue: str, simple_cues: Set[str]) -> List[str]:
    """
    If a cue line concatenates multiple known character names (e.g.,
    "CHRIS ROSE" with both CHRIS and ROSE being known speakers, or
    "FERNANDO WOOD ROBERT LATHAM" with both as known multi-word names),
    split it into the underlying speakers.

    Returns the list of decomposed names if successful, or [cue] otherwise.

    Strategy: greedy longest-prefix match against simple_cues.
    """
    if " " not in cue or cue in simple_cues:
        return [cue]
    tokens = cue.split()
    if len(tokens) < 2:
        return [cue]
    # Each token alone is a known cue → split token-by-token
    if all(tok in simple_cues for tok in tokens):
        return tokens
    # Try multi-word decomposition (greedy, longest-first)
    result: List[str] = []
    i = 0
    while i < len(tokens):
        match_len = 0
        for j in range(len(tokens), i, -1):
            candidate = " ".join(tokens[i:j])
            if candidate in simple_cues:
                match_len = j - i
                result.append(candidate)
                break
        if match_len == 0:
            return [cue]  # could not fully decompose; keep original
        i += match_len
    return result if len(result) > 1 else [cue]


def _is_probable_character_line(
    current_line: str,
    next_line: str,
    cfg: ExtractionConfig,
) -> bool:
    """
    Heuristic: a character cue is typically ALL CAPS and followed by a non-ALL-CAPS line (dialogue).
    We also exclude sluglines and common transitions.

    Note: We *intentionally* keep this heuristic close to the paper description
    but make it deterministic and safer against false positives.
    """
    cur = current_line.strip()
    nxt = next_line.strip()

    if not cur:
        return False

    # Must look like ALL CAPS (character cues in scripts are usually in caps)
    if not is_allcaps_line(cur):
        return False

    # Next line must exist and should NOT be all caps (likely dialogue text)
    if not nxt or is_allcaps_line(nxt):
        return False

    # Exclude obvious non-character lines
    if is_transition(cur):
        return False

    if cfg.legacy_mode:
        # Legacy: exclude only the classic prefixes; closer to the older script_core.py.
        if cur.startswith(("INT.", "EXT.", "FADE", "CUT TO", "DISSOLVE TO:")):
            return False
    else:
        # Strict: exclude any slugline-like headings and lines with colon.
        if is_slugline(cur):
            return False
        if _HAS_COLON_RE.search(cur):
            return False

    # Limit name length
    words = cur.split()
    if not (0 < len(words) <= cfg.character_max_words):
        return False

    return True


# ==========================================================
#   SCENE SPLITTING
# ==========================================================

def split_script_into_scenes(full_script_text: str) -> List[Scene]:
    """
    Split script into scenes using sluglines (INT/EXT...) as delimiters.

    - Uses a lookahead regex so the slugline is kept as the first line of each scene block.
    - Accepts 'INT.'/'EXT.' or 'INT '/'EXT '.

    Returns a list of Scene objects (1-based index).
    """
    text = normalize_whitespace(full_script_text)

    # Keep delimiter line by splitting with lookahead
    raw_scenes = re.split(r"(?=^\s*(?:INT|EXT)[\.\s/])", text, flags=re.MULTILINE)
    raw_scenes = [s.strip("\n") for s in raw_scenes if s.strip()]

    # Drop any preamble blocks *before* the first real slugline.
    # This avoids treating title pages / cast lists as a scene and producing false turns.
    raw_scenes_kept = list(raw_scenes)
    while raw_scenes_kept:
        first_line = raw_scenes_kept[0].lstrip().split('\n', 1)[0].strip()
        if first_line and is_slugline(first_line):
            break
        raw_scenes_kept.pop(0)
    if raw_scenes_kept:
        raw_scenes = raw_scenes_kept

    scenes: List[Scene] = []
    for idx, block in enumerate(raw_scenes, start=1):
        lines = block.split("\n")
        heading = lines[0].strip() if lines else ""
        body = "\n".join(lines[1:]).lstrip("\n")
        scenes.append(Scene(index=idx, heading=heading, text=block, body=body))
    return scenes


# ==========================================================
#   CHARACTER DISCOVERY + CANONICALIZATION
# ==========================================================

def build_normalization_map(names: Iterable[str], similarity_threshold: float) -> Dict[str, str]:
    """
    Deterministically cluster similar names and map each raw name to a canonical name.

    Similarity criteria:
      - substring containment (A in B or B in A), OR
      - Levenshtein ratio > threshold, with a conservative *token guard* to avoid
        collapsing distinct role-like names that only share a prefix (e.g., 'AGENT JONES' vs 'AGENT BROWN').

    Canonical name: the *longest* string in the cluster; ties broken lexicographically.
    """

    def _token_guard(a: str, b: str) -> bool:
        """Return True if string-similarity merging is plausible given whole-token overlap.

        We keep substring merging as-is (covers 'VITO' -> 'VITO CORLEONE').
        For ratio-based merges, we require either:
          - last token matches (common in 'DR. SMITH' vs 'DR SMITH'), OR
          - at least 2 shared tokens, OR
          - one is a single token that appears as a whole token in the other.

        This blocks merges like 'AGENT JONES' ~ 'AGENT BROWN' (only shares 'AGENT').
        """
        ta = a.split()
        tb = b.split()
        if not ta or not tb:
            return False
        if ta[-1] == tb[-1]:
            return True
        inter = set(ta) & set(tb)
        if len(inter) >= 2:
            return True
        if len(ta) == 1 and ta[0] in tb:
            return True
        if len(tb) == 1 and tb[0] in ta:
            return True
        return False
    names_sorted = sorted(set(names), key=lambda n: (-len(n), n))

    clusters: List[List[str]] = []

    for name in names_sorted:
        matched_cluster_idx: Optional[int] = None
        for ci, cluster in enumerate(clusters):
            rep = max(cluster, key=lambda s: (len(s), s))  # canonical candidate
            if name in rep or rep in name or (levenshtein_ratio(name, rep) > similarity_threshold and _token_guard(name, rep)):
                matched_cluster_idx = ci
                break

        if matched_cluster_idx is None:
            clusters.append([name])
        else:
            clusters[matched_cluster_idx].append(name)

    mapping: Dict[str, str] = {}
    for cluster in clusters:
        canonical = max(cluster, key=lambda s: (len(s), s))
        for raw in cluster:
            mapping[raw] = canonical

    return mapping


def discover_and_validate_characters(full_script_text: str, cfg: ExtractionConfig) -> Tuple[Dict[str, str], Set[str], Counter]:
    """
    Pass 1: discover and validate characters from the full script text.

    Strategy (matches the paper description):
      - scan line by line;
      - a line is a character cue if it is ALL CAPS, followed by non-ALL-CAPS,
        not a slugline/transition, and has <= character_max_words;
      - clean markers like (V.O.) / (CONT'D);
      - count frequencies, keep those with >= minimum_appearances;
      - cluster similar names and map to canonical forms.

    Returns:
      - name_mapping: dict {raw_name -> canonical_name}
      - canonical_set: set of canonical names
      - raw_counts: Counter of validated (cleaned) raw names before clustering
    """
    text = normalize_whitespace(full_script_text)
    lines = text.split("\n")
    candidates: List[str] = []

    for i in range(len(lines) - 1):
        cur = lines[i].strip()
        nxt = lines[i + 1].strip()

        if _is_probable_character_line(cur, nxt, cfg):
            cleaned = clean_and_normalize_special_cases(cur)
            if cleaned and not is_blacklisted_character(cleaned):
                candidates.append(cleaned)

    raw_counts = Counter(candidates)

    validated_raw = {name for name, c in raw_counts.items() if c >= cfg.minimum_appearances}

    # Identify compound cues: cues that can be split into >= 2 atomic cues.
    # An atomic cue is one that does NOT decompose into other validated cues.
    # We discover atomic cues iteratively (handles "PRESTON BLAIR LINCOLN" =
    # ["PRESTON BLAIR", "LINCOLN"] when "PRESTON BLAIR" itself is atomic).
    atomic_cues: Set[str] = {n for n in validated_raw if " " not in n}
    # Process by ascending token count so shorter cues become atomic first
    # (needed for "PRESTON BLAIR LINCOLN" to see "PRESTON BLAIR" already atomic).
    ordered = sorted(validated_raw, key=lambda n: (len(n.split()), n))
    changed = True
    while changed:
        changed = False
        for name in ordered:
            if name in atomic_cues or " " not in name:
                continue
            parts = decompose_compound_cue(name, atomic_cues)
            if len(parts) == 1:  # cannot decompose → it's atomic
                atomic_cues.add(name)
                changed = True

    compound_cues: Set[str] = set()
    for name in validated_raw - atomic_cues:
        parts = decompose_compound_cue(name, atomic_cues)
        if len(parts) > 1:
            compound_cues.add(name)
    # Exclude compound cues from clustering input — we'll decompose them at runtime.
    clustering_input = validated_raw - compound_cues

    name_mapping = build_normalization_map(clustering_input, similarity_threshold=cfg.similarity_threshold)
    canonical_set = set(name_mapping.values())

    # Store compound cues in the mapping too, so we can detect them later.
    # They point to a special sentinel "__COMPOUND__" so extract_speaker_turns can split.
    for cue in compound_cues:
        name_mapping[cue] = "__COMPOUND__:" + cue

    return name_mapping, canonical_set, raw_counts


# ==========================================================
#   TURN EXTRACTION (PER SCENE)
# ==========================================================

def extract_speaker_turns(
    scene_text: str,
    name_mapping: Dict[str, str],
    canonical_set: Set[str],
    cfg: ExtractionConfig,
) -> Tuple[List[str], Counter]:
    """
    Extract the sequence of speaker turns from a scene.

    Returns:
      - speakers: list of canonical speaker names (one per detected turn)
      - turn_counts: Counter of turns per canonical speaker
    """
    text = normalize_whitespace(scene_text)
    lines = text.split("\n")

    speakers: List[str] = []
    for i in range(len(lines) - 1):
        cur = lines[i].strip()
        nxt = lines[i + 1].strip()

        if _is_probable_character_line(cur, nxt, cfg):
            cleaned = clean_and_normalize_special_cases(cur)
            if not cleaned or is_blacklisted_character(cleaned):
                continue
            if cleaned in name_mapping:
                canonical = name_mapping[cleaned]
                if isinstance(canonical, str) and canonical.startswith("__COMPOUND__:"):
                    # Compound cue: decompose into known canonical speakers
                    raw_compound = canonical[len("__COMPOUND__:"):]
                    parts = decompose_compound_cue(raw_compound, canonical_set)
                    for part in parts:
                        if part in canonical_set:
                            speakers.append(part)
                elif canonical in canonical_set:
                    speakers.append(canonical)

    return speakers, Counter(speakers)


def compute_cast_sets(speakers_per_scene: Sequence[Sequence[str]]) -> List[Set[str]]:
    return [set(speakers) for speakers in speakers_per_scene]


def jaccard_distance(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    inter = len(a & b)
    return 1.0 - (inter / union) if union else 0.0


def compute_cast_jaccard_series(cast_sets: Sequence[Set[str]]) -> List[float]:
    """D_t (paper): Jaccard distance between speaking casts in consecutive scenes; D_1 = 0."""
    out: List[float] = []
    prev: Set[str] = set()
    for i, cur in enumerate(cast_sets):
        if i == 0:
            out.append(0.0)
        else:
            out.append(jaccard_distance(prev, cur))
        prev = cur
    return out


# Expose similarity backend for reproducibility logs.
STRING_SIM_BACKEND = _STRING_SIM_BACKEND
