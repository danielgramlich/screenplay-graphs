"""
Amostragem estratificada de 100 filmes para os papers ENIAC 2026 e IBERAMIA 2026.

Critérios:
- Subset: 574 filmes com metadados completos (scored_dataset_v2.csv)
- Estratificação por main_genre (primeiro gênero da lista IMDb)
- Filtro: year >= 1990 (consistência de parsing)
- Exclusão: Documentary, Short (perfis textuais distintos de ficção narrativa)
- Seed fixa: 42 (reprodutibilidade)

Saída: data/metadata/corpus_100_v1.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "features" / "scored_dataset_v2.csv"
OUT = ROOT / "data" / "metadata" / "corpus_100_v1.csv"

SEED = 42

# Cotas por gênero principal
QUOTAS = {
    "Action": 18,
    "Comedy": 18,
    "Drama": 16,
    "Crime": 12,
    "Biography": 8,
    "Horror": 9,
    "Adventure": 6,
    "Animation": 5,
    # 10 vagas para "Outros"
}
OTHERS_QUOTA = 8  # 2 vagas redirecionadas para Drama/Horror abaixo
EXCLUDE_GENRES = {"Documentary", "Short", "Adult", "News"}
MIN_YEAR = 1990


def main():
    df = pd.read_csv(SRC)
    print(f"[1/4] Total inicial: {len(df)}")

    # Gênero principal = primeiro gênero da lista
    df["main_genre"] = df["genre"].str.split(",").str[0].str.strip()

    # Filtros (descarta sem gênero ou sem ano)
    df = df.dropna(subset=["main_genre", "year"])
    df = df[df["year"] >= MIN_YEAR]
    df = df[~df["main_genre"].isin(EXCLUDE_GENRES)]
    print(f"[2/4] Após filtro (year>={MIN_YEAR}, sem {EXCLUDE_GENRES}): {len(df)}")

    # Amostragem por estrato
    sampled = []
    for genre, quota in QUOTAS.items():
        pool = df[df["main_genre"] == genre]
        if len(pool) < quota:
            print(f"  ⚠ {genre}: pool={len(pool)} < quota={quota}, pegando todos")
            sampled.append(pool)
        else:
            sampled.append(pool.sample(n=quota, random_state=SEED))

    # Outros: gêneros não cobertos pelas cotas explícitas
    covered = set(QUOTAS.keys()) | EXCLUDE_GENRES
    others_pool = df[~df["main_genre"].isin(covered)]
    print(f"[3/4] Pool 'Outros' ({sorted(others_pool['main_genre'].unique())}): {len(others_pool)}")
    if len(others_pool) >= OTHERS_QUOTA:
        sampled.append(others_pool.sample(n=OTHERS_QUOTA, random_state=SEED))
    else:
        sampled.append(others_pool)

    out = pd.concat(sampled, ignore_index=True)

    # Selecionar e ordenar colunas
    cols = ["title", "year", "main_genre", "genre", "imdb_actual", "box_office_actual"]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].sort_values(["main_genre", "year", "title"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[4/4] Salvo: {OUT} ({len(out)} filmes)")
    print()
    print("Distribuição final:")
    print(out["main_genre"].value_counts().to_string())
    print()
    print("Range de anos:", out["year"].min(), "—", out["year"].max())
    print("IMDb: min/mediana/max =",
          f"{out['imdb_actual'].min():.1f}",
          f"{out['imdb_actual'].median():.1f}",
          f"{out['imdb_actual'].max():.1f}")


if __name__ == "__main__":
    main()
