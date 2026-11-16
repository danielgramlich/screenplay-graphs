#!/usr/bin/env python3
"""Etapa 25: Expandir corpus com MovieSum (HuggingFace).

1. Baixa MovieSum dataset (1800 roteiros com imdb_id)
2. Converte de XML para texto puro (formato slugline)
3. Deduplicata com corpus existente
4. Salva novos roteiros em data/raw/
"""

import sys
import re
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import RAW, METADATA, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def xml_to_plaintext(script_xml):
    """Converte roteiro XML do MovieSum para texto puro com sluglines.

    MovieSum usa:
      <scene> ... </scene>
      <stage_direction>INT. CASA — DIA</stage_direction>
      <scene_description>...</scene_description>
      <dialogue>
        <character>JOHN</character>
        <line>Hello world</line>
        <parenthetical>(smiling)</parenthetical>
      </dialogue>
    """
    # Remover tag <script> externa
    text = script_xml.strip()
    text = re.sub(r"</?script>", "", text).strip()

    lines = []

    # Processar cena por cena
    scenes = re.findall(r"<scene>(.*?)</scene>", text, re.DOTALL)

    for scene in scenes:
        # Stage direction (slugline)
        sluglines = re.findall(r"<stage_direction>(.*?)</stage_direction>", scene)
        for sl in sluglines:
            sl_clean = sl.strip()
            if sl_clean:
                lines.append("")
                lines.append(sl_clean)
                lines.append("")

        # Scene description (ação/direção)
        descriptions = re.findall(r"<scene_description>(.*?)</scene_description>", scene, re.DOTALL)
        for desc in descriptions:
            desc_clean = desc.strip()
            if desc_clean:
                # Limpar tags internas residuais
                desc_clean = re.sub(r"<.*?>", "", desc_clean)
                lines.append("\t" + desc_clean)

        # Diálogos
        dialogues = re.findall(r"<dialogue>(.*?)</dialogue>", scene, re.DOTALL)
        for dial in dialogues:
            # Character name
            char_match = re.search(r"<character>(.*?)</character>", dial)
            if char_match:
                char_name = char_match.group(1).strip().upper()
                lines.append("")
                lines.append("\t\t\t\t" + char_name)

            # Parenthetical
            paren_match = re.search(r"<parenthetical>(.*?)</parenthetical>", dial)
            if paren_match:
                lines.append("\t\t\t" + paren_match.group(1).strip())

            # Lines of dialogue
            dial_lines = re.findall(r"<line>(.*?)</line>", dial)
            for dl in dial_lines:
                dl_clean = dl.strip()
                if dl_clean:
                    lines.append("\t\t\t" + dl_clean)

    result = "\n".join(lines)

    # Limpar tags residuais
    result = re.sub(r"<.*?>", "", result)
    # Limpar linhas em branco excessivas
    result = re.sub(r"\n{4,}", "\n\n\n", result)

    return result


def sanitize_filename(name):
    """Converte nome do filme em filename seguro."""
    # Remover ano do final (ex: 'The Matrix_1999' -> 'the-matrix')
    name = re.sub(r"_\d{4}$", "", name)
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s-]", "", name)
    name = re.sub(r"\s+", "-", name)
    return name


def main():
    ensure_dirs()

    # Filmes existentes
    existing = {p.stem for p in RAW.glob("*.txt")}
    logging.info("Corpus existente: %d filmes", len(existing))

    # Baixar MovieSum
    logging.info("Baixando MovieSum do HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("rohitsaxena/MovieSum")

    # Juntar todos os splits
    all_records = []
    for split in ds:
        for record in ds[split]:
            all_records.append(record)

    logging.info("MovieSum total: %d filmes", len(all_records))

    # Processar
    added = 0
    skipped_duplicate = 0
    skipped_short = 0
    skipped_error = 0
    imdb_map = {}  # filename -> imdb_id para metadata

    for i, record in enumerate(all_records):
        movie_name = record["movie_name"]
        imdb_id = record.get("imdb_id", "")
        script_xml = record.get("script", "")

        filename = sanitize_filename(movie_name)
        if not filename:
            skipped_error += 1
            continue

        # Deduplicar
        if filename in existing:
            skipped_duplicate += 1
            continue

        # Converter
        try:
            plaintext = xml_to_plaintext(script_xml)
        except Exception as e:
            skipped_error += 1
            continue

        if len(plaintext) < 1000:
            skipped_short += 1
            continue

        # Salvar
        out_path = RAW / f"{filename}.txt"
        if out_path.exists():
            # Nome já existe (colisão diferente)
            skipped_duplicate += 1
            continue

        out_path.write_text(plaintext, encoding="utf-8")
        existing.add(filename)
        imdb_map[filename] = imdb_id
        added += 1

        if (i + 1) % 200 == 0:
            logging.info("Progresso: %d/%d processados, %d adicionados", i + 1, len(all_records), added)

    # Salvar mapa de imdb_ids para metadata posterior
    import json
    map_path = METADATA / "moviesum_imdb_map.json"
    with open(map_path, "w") as f:
        json.dump(imdb_map, f, indent=2)

    logging.info("=" * 60)
    logging.info("  EXPANSÃO DO CORPUS")
    logging.info("=" * 60)
    logging.info("  MovieSum processados: %d", len(all_records))
    logging.info("  Adicionados: %d", added)
    logging.info("  Duplicados ignorados: %d", skipped_duplicate)
    logging.info("  Muito curtos: %d", skipped_short)
    logging.info("  Erros: %d", skipped_error)
    logging.info("  Corpus total agora: %d", len(list(RAW.glob("*.txt"))))
    logging.info("  Mapa IMDb salvo: %s", map_path)


if __name__ == "__main__":
    main()
