# Representações em grafo de roteiros cinematográficos

Código da dissertação de mestrado *"Representações em grafo de roteiros
cinematográficos: comparabilidade entre construções e especialização por
gênero"* (Daniel Maximo Gramlich, PPGEE/UFES, 2026) e do artigo aceito no
LVIII Simpósio Brasileiro de Pesquisa Operacional (SBPO 2026).

## Estrutura

- `src/` — biblioteca: *parser* determinístico de roteiros (segmentação de
  cenas, identificação de personagens, normalização por distância de
  Levenshtein), construção das redes, classificação de sentimento e coleta
  de metadados.
- `pipeline/` — etapas numeradas das análises, da coleta ao *hold-out*
  temporal. Os números das etapas correspondem aos registrados na
  dissertação.
- `scripts_corpus/` — amostragem estratificada do subconjunto de 100
  roteiros do estudo comparativo (semente fixa).
- `tests/` — testes do *parser*.

## Reprodução

O corpus (1.655 roteiros em inglês) **não é redistribuído** por restrições
de direitos autorais. Os roteiros foram obtidos dos repositórios públicos
[IMSDb](https://imsdb.com) e [SimplyScripts](https://www.simplyscripts.com);
os metadados vêm de [OMDb](https://www.omdbapi.com) e
[TMDB](https://www.themoviedb.org).

As etapas de coleta exigem chaves de API nas variáveis de ambiente
`OMDB_API_KEY` e `TMDB_API_KEY`.

Cada execução do *parser* registra o *hash* MD5 do arquivo de entrada, a
configuração e as versões de *software*; com o mesmo insumo, a saída é
sempre a mesma.

## Licença

MIT (ver `LICENSE`). O código é disponibilizado para fins de verificação e
reprodução acadêmica.
