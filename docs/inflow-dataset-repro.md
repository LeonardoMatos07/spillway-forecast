# Conjunto de treino da afluência: reprodutibilidade (Etapa 1)

Conferido em 30/10/2026. Nenhum modelo foi treinado; nenhum dado de teste foi usado para escolher nada.

## Como reproduzir

1. Exportar do warehouse (somente leitura): `reservoir_daily` (colunas `reservoir_id, day, natural_inflow_m3s, storage_pct`) e `reservoir` (`id` como `reservoir_id`, `basin, subsystem, useful_volume_hm3, latitude, longitude`) em Parquet.
2. Chuva: `python -m energy_pipeline.rain` no `spillway-data` (IMERG, NASA POWER, CC BY 4.0), exportada em Parquet (`reservoir_id, day, precip_mm, run`).
3. `uv run spillway inflow-dataset --reservoir-daily rd.parquet --register reg.parquet --rain rain.parquet --out datasets/`

A saída fica em `datasets/<dataset_id>/` com `features.parquet` e `manifest.json`. O `dataset_id` é o hash (16 caracteres) da configuração, do hash de cada entrada e do hash do arquivo de saída: mesmas entradas, mesmo id. Sementes não se aplicam (não há sorteio). Versões travadas em `uv.lock`.

## Regras contra vazamento (cada uma tem teste em `tests/test_inflow_dataset.py`)

| Regra | Teste |
|---|---|
| Features no dia de origem usam só dias até a origem | alterar vazão e chuva depois de uma data não muda nenhuma feature anterior |
| Chuva com atraso de 46 dias (atraso real do Zarr em 30/10/2026) | alterar a chuva após origem - 46 não muda a feature; a janela de 7 dias bate com a soma esperada |
| Alvo = vazão natural observada em t+h, h = 1..14 | comparação direta com a série |
| Purge: janela de 14 dias termina dentro do próprio período | último origin de treino, validação e teste |
| Média de longo prazo só dos anos de treino | triplicar a vazão depois do treino não muda `q_mean_train` |
| Lacunas até 3 dias interpoladas e marcadas; nunca viram alvo; lacunas longas retiram as janelas | testes de lacuna curta e longa |
| Hash estável e sensível à entrada | dois builds iguais, e uma alteração de 1 m3s em um dia muda o id |

## Divisão

Treino 2002-2017, validação 2018-2020, teste 2021-2025, reserva 2026 em diante (nunca usada para decidir). O teste úmido do Sudeste não existe nos anos recentes: limitação documentada.

## Execução de verificação (dados reais, bacias Grande e Iguaçu)

Resultado: dataset 138c1cae45a337dc, 95.657 linhas (treino 22.839, validação 24.886, teste 41.676, reserva 6.256), 7,5 MB. Banco local (2015-01 a 2026-09-30, 23 reservatórios) e chuva IMERG real (98.325 linhas, 23 pontos, 2015 a 14/09/2026). Como o banco local ainda não tem 2000-2014, os números abaixo só provam que o construtor roda em dados reais; o conjunto de produção exige a carga histórica (`--from 2000-01`) no `spillway-data`.

## Limites conhecidos

- A chuva do Zarr termina em 14/09/2026 (atraso de cerca de 46 dias, apesar do `last_available.json`). Por isso a feature de chuva usa defasagem de 46 dias; o ganho esperado é pequeno. Decisão do dono: seguir só com vazões ou obter fonte de menor atraso (GPM IMERG Late no GES DISC exige conta Earthdata).
- Chuva é média de 3 x 3 pixels em torno da barragem, não da bacia de contribuição.
- Montante (topologia) não entrou: só nos pilotos, à mão, em etapa seguinte.
