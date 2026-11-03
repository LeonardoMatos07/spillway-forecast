# Previsão de afluência: rede recorrente no piloto (Etapa 3)

Conferido em 01/11/2026. Previsão estatística; não é a previsão oficial do ONS. Resultado honesto: **a rede não passa no critério de publicação em nenhum horizonte**; a afluência fica "em avaliação" e não vai ao ar.

## Protocolo

- Mesmo conjunto da Etapa 2 (`26b4096746a8573b`): Grande (15) e Iguaçu (8), 23 reservatórios; treino 2002-2017, validação 2018-2020 (parada antecipada e escolha da célula), teste 2021-2025 só previsto no fim; reserva 2026 não lida.
- Um modelo global (`inflow_deep.py`), CPU apenas (PyTorch CPU, 4 threads, extra opcional `deep`; nada instalado por padrão). Entrada: últimos 90 dias de log da vazão normalizada por reservatório (média e desvio só do treino), armazenamento, chuva IMERG defasada 46 dias (atraso real), seno e cosseno do dia do ano, mais estáticos (volume útil, bacia). Saída: variação em log de 1 a 14 dias à frente, P10/P50/P90, perda pinball, sem cruzamento de quantis.
- LSTM contra GRU: uma semente, pela perda de validação. Ganhou a GRU (0,1018 contra 0,1032). Camada única de 48 unidades, dropout 0,2, Adam 2e-3, 30 mil amostras por época, até 20 épocas, paciência 4 (as redes pararam em 6 a 13 épocas), clip de gradiente 1,0.
- Conjunto de 5 sementes fixas (11, 22, 33, 44, 55), quantis médios. Cerca de 2 a 4 minutos por rede na CPU.
- Ablação: a mesma rede sem o canal de chuva (`net_no_rain`).
- Nenhuma busca de hiperparâmetros usou o teste. Os únicos valores escolhidos em dados foram a célula (validação) e a época de parada (validação).
- Critério estrito inalterado (`inflow_eval.gate`), contra a melhor entre TODOS os modelos da Etapa 2 (quatro referências, ridge e boosting), escolhida na validação (ridge em todos os horizontes).

## Resultado no teste (2021-2025), mediana dos 23 reservatórios

| h | Modelo | NSE | KGE | Viés % | Pico % | Cobertura |
|---|---|---|---|---|---|---|
| 1 | ridge | 0,985 | 0,986 | -0,2 | -1,4 | 0,80 |
| 1 | rede sem chuva | 0,978 | 0,954 | -2,2 | -4,6 | 0,86 |
| 3 | ridge | 0,843 | 0,892 | -1,3 | -10,9 | 0,80 |
| 3 | rede sem chuva | 0,845 | 0,809 | -6,7 | -18,2 | 0,82 |
| 7 | ridge | 0,658 | 0,726 | -3,1 | -26,5 | 0,80 |
| 7 | boosting | 0,662 | 0,685 | -7,5 | -31,5 | 0,78 |
| 7 | rede | 0,667 | 0,675 | -8,6 | -31,2 | 0,79 |
| 7 | rede sem chuva | 0,678 | 0,681 | -8,4 | -31,2 | 0,79 |
| 14 | ridge | 0,511 | 0,576 | -3,9 | -38,8 | 0,77 |
| 14 | rede | 0,554 | 0,587 | -8,2 | -40,3 | 0,75 |
| 14 | rede sem chuva | 0,545 | 0,617 | -6,3 | -38,9 | 0,76 |

Tabelas completas em `reports/inflow_stage3/` (`aggregate_*.csv`, `gate_*.csv`).

## Critério de publicar

| Candidato | Horizontes aprovados |
|---|---|
| Rede com chuva | nenhum |
| Rede sem chuva | nenhum |

Motivos: a rede só passa na condição de NSE/KGE a partir de h=3 a 5; em todos os horizontes o viés (-6% a -9%) e o erro nos picos (-31% a -40%) são piores que os do ridge e da persistência; nos horizontes curtos ela perde para o ridge em mais de 30% dos reservatórios. A cobertura P10-P90 é aceitável (0,75 a 0,86).

## Leitura

1. O ganho da rede sobre ridge e boosting é pequeno e dentro do ruído entre sementes (h=7: NSE 0,67 contra 0,66; h=14: 0,55 contra 0,51). Nos horizontes 1 a 3 ela é pior.
2. A chuva com atraso de 46 dias continua sem ajudar: a rede sem chuva é igual ou melhor. Qualquer ganho real depende de chuva com atraso de poucos dias (fonte que exige conta, decisão do dono) ou de previsão de chuva, que está fora de escopo.
3. O limite é a informação, não a arquitetura: sem chuva recente nem futura, os picos de cheia não são previsíveis a partir da própria vazão.
4. Não foi feito: janela de 365 dias, mais bacias, teste espacial, ajuste de hiperparâmetros além da célula, GPU. Nenhuma instância paga e nenhuma conta foram usadas.

## Decisão

Por regra da spec, como nenhum horizonte passou, o trabalho para aqui como relato técnico: não há etapas 4 a 7 (todos os reservatórios, publicação, página, operação) para esta previsão. A afluência deve constar como "em avaliação" no hub. Ponto de decisão 2 do dono: aceitar o encerramento, ou liberar uma fonte de chuva de baixo atraso e repetir a avaliação; mudar o critério depois destes resultados não é permitido pela spec.

## Reprodução

`uv sync --extra deep`; depois `python -m spillway_forecast.inflow_run --dataset datasets/<id> --out out2` (gera `forecasts.parquet`) e `python -m spillway_forecast.inflow_deep_run --dataset datasets/<id> --reservoir-daily rd.parquet --register reg.parquet --rain rain.parquet --baseline-forecasts out2/forecasts.parquet --out out3`. Versões em `uv.lock`.
