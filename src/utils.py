"""Constantes de paths e helpers de I/O."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
METADATA = DATA / "metadata"
VALIDATION = DATA / "validation"

# Dados intermediários (cenas, emoções, grafos)
INTERMEDIATE = DATA / "intermediate"
SCENES = INTERMEDIATE / "scenes"
EMOTIONS = INTERMEDIATE / "emotions"
GRAPHS = INTERMEDIATE / "graphs"
LLM_ANNOTATIONS = INTERMEDIATE / "llm_annotations"

# Features consolidadas
FEATURES = DATA / "features"
FEATURES_EMOTIONAL = FEATURES / "emotional_graphs"
FEATURES_CHARACTERS = FEATURES / "characters"

# Resultados finais
RESULTS = DATA / "results"

# Alias para compatibilidade (scripts antigos usam PROCESSED)
PROCESSED = INTERMEDIATE

ALL_DIRS = [RAW, METADATA, VALIDATION, INTERMEDIATE, SCENES, EMOTIONS,
            GRAPHS, LLM_ANNOTATIONS, FEATURES, FEATURES_EMOTIONAL,
            FEATURES_CHARACTERS, RESULTS]

EMOTIONS_LIST = ["joy", "sadness", "anger", "fear", "surprise", "trust"]


def ensure_dirs():
    """Cria todos os diretórios de dados se não existirem."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def sanitize_title(title: str) -> str:
    """Converte título em nome de arquivo seguro."""
    import re
    title = title.lower().strip()
    title = re.sub(r"[^a-z0-9\s-]", "", title)
    title = re.sub(r"\s+", "-", title)
    return title
