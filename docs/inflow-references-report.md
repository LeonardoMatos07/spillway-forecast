# Previsão de afluência: referências e arcabouço de avaliação (Etapa 2)

Conferido em 01/11/2026. Previsão estatística; não é a previsão oficial do ONS. Nenhuma rede neural foi treinada. Nenhum dado de teste escolheu algo: referências e modelos foram ajustados em 2002-2017, os poucos hiperparâmetros foram escolhidos em 2018-2020, e 2021-2025 só foi pontuado. A reserva (2026 em diante) não foi lida.

## O que foi medido

Piloto: bacias Grande (15 reservatórios) e Iguaçu (8), 23 reservatórios, vazão natural diária desde 2000 (ONS) e chuva IMERG com atraso real de 46 dias. Conjunto `26b4096746a8573b` (construtor da Etapa 1). Linhas: treino 134.090, validação 24.886, teste 41.676.

Modelos (ordem da spec): persistência, climatologia por dia do ano, recessão exponencial por reservatório, média móvel de 7 dias; regressão ridge por horizonte; gradient boosting com quantis por horizonte (sklearn, sem dependência nova). As referências são pontuais: a faixa P10/P90 delas vem dos resíduos (em log) do treino, por reservatório e horizonte. Ridge e boosting preveem a variação em log em relação à última vazão. Reprodução: `uv run spillway inflow-evaluate --dataset datasets/<id> --out <dir>`; semente 20260101; resultados em `reports/inflow_stage2/`.

## Resultado no teste (2021-2025), mediana entre os 23 reservatórios

| h | Modelo | NSE | KGE | Viés % | Erro nos picos % | Cobertura P10-P90 |
|---|---|---|---|---|---|---|
| 1 | persistência | 0,965 | 0,983 | 0,06 | -0,9 | 0,80 |
| 1 | ridge | 0,985 | 0,986 | -0,16 | -1,4 | 0,80 |
| 1 | boosting | 0,973 | 0,948 | -2,1 | -4,6 | 0,76 |
| 3 | persistência | 0,807 | 0,904 | 0,15 | -6,0 | 0,80 |
| 3 | ridge | 0,843 | 0,892 | -1,3 | -10,9 | 0,80 |
| 3 | boosting | 0,825 | 0,837 | -5,1 | -16,0 | 0,77 |
| 7 | persistência | 0,566 | 0,784 | 0,30 | -15,0 | 0,80 |
| 7 | recessão | 0,573 | 0,686 | -17,9 | -27,9 | 0,80 |
| 7 | ridge | 0,658 | 0,726 | -3,1 | -26,5 | 0,80 |
| 7 | boosting | 0,662 | 0,685 | -7,5 | -31,5 | 0,78 |
| 14 | persistência | 0,283 | 0,644 | 0,55 | -24,2 | 0,79 |
| 14 | média móvel | 0,303 | 0,642 | 0,67 | -26,8 | 0,78 |
| 14 | ridge | 0,511 | 0,576 | -3,9 | -38,8 | 0,77 |
| 14 | boosting | 0,499 | 0,611 | -5,0 | -38,9 | 0,76 |
| 1 a 14 | climatologia | 0,39 | 0,55 | +13 | -42 | 0,79 |

Todas as linhas e a dispersão (quartis) entre reservatórios estão em `reports/inflow_stage2/aggregate_test.csv` e `aggregate_valid.csv`.

Leituras:

1. Ridge e boosting ganham da melhor referência em NSE em todos os horizontes (ganho grande de h=7 em diante: NSE de 14 dias de 0,28 para 0,51). Em KGE a persistência ainda vence a partir de h=3, porque os modelos encolhem a variabilidade.
2. Os dois modelos subestimam picos (-27% a -39% em h=7 a 14) e têm viés negativo (-3% a -8%). A persistência tem viés quase zero por construção.
3. A faixa P10-P90 está calibrada (cobertura mediana de 0,76 a 0,80; alvo 0,80, aceitável 0,70 a 0,90) em todos os modelos.
4. Chuva com atraso de 46 dias não ajuda: o boosting sem as três variáveis de chuva é igual ao com chuva (NSE de 14 dias: 0,514 contra 0,499; de 7 dias: 0,659 contra 0,662). O ganho vem das vazões e do calendário.
5. Por bacia, h=7 (NSE mediano): Grande ridge 0,72 contra persistência 0,66; Iguaçu ridge 0,51 contra persistência 0,42. O Iguaçu perde viés (ridge -9,6%, boosting -18%) e é o mais difícil.
6. Por ano hidrológico (alvo, h=7, ridge): 2021 (seco) 0,70, 2022 0,53, 2023 0,66, 2024 0,60, 2025 0,49. O ganho sobre a persistência existe em todos os anos; o ano 2026 do arquivo só tem os primeiros meses e não é interpretável. Limite herdado: nos anos de teste não há Sudeste realmente úmido.

## Critério de publicar, aplicado de forma estrita

Por horizonte, contra a melhor referência (escolhida na VALIDAÇÃO, não no teste): vencer em NSE ou KGE (mediana), não piorar viés absoluto nem erro nos picos (mediana), cobertura entre 70% e 90%, e não ser pior em NSE em mais de 30% dos reservatórios. Implementado em `inflow_eval.gate`, com teste para cada condição.

| Candidato | Horizontes que passam |
|---|---|
| Ridge | apenas h=7 (a referência escolhida ali é a recessão, que tem viés de -18%) |
| Boosting | nenhum |
| Boosting sem chuva | nenhum |

A falha é sempre a mesma: venceram em NSE e cobertura, mas não em viés e picos contra a persistência (viés de 0,1 a 0,7% é difícil de bater). Isto é o critério como foi escrito; ele não foi alterado.

## Ponto de decisão 1

1. Os resultados acima são o piso que a rede (Etapa 3) precisa superar. A rede só vale a pena se corrigir picos e viés; o ganho possível está em h=3 a 14.
2. O critério estrito permanece como está e vale para a publicação. A condição "viés e picos não piores que a melhor referência" é muito dura contra a persistência (viés de 0,1% a 0,7%), mas ajustá-la depois de ver os resultados não é permitido pela spec; qualquer mudança é decisão do dono para rodadas futuras.

   **Análise exploratória pós-hoc, não vale para publicação:** se a regra fosse uma tolerância absoluta (viés mediano até 5% em módulo, erro de pico até 30% em módulo, cobertura 70% a 90%, sem as demais condições), o ridge cumpriria h=1 a 8 e o boosting h=1 e 2 no teste. Isto foi calculado depois de ver os resultados e serve só de informação.
3. Chuva: com atraso de 46 dias ela não entra. Seguir só com vazões, ou obter fonte de menor atraso (GPM IMERG Late exige conta Earthdata, que é decisão do dono).
4. Topologia e montante continuam fora (só nos pilotos, à mão).

## Limites

- Medianas de 23 reservatórios de duas bacias; sem teste espacial (bacia inteira fora do treino).
- Sem intervalo de confiança do bootstrap (não implementado nesta etapa).
- Alvo é a vazão natural como publicada hoje pelo ONS; revisões posteriores não foram simuladas.
- O banco/lake local não foi usado (Docker fora do ar): vazões e cadastro vieram da exportação em arquivos, chuva do bucket aberto da NASA POWER (CC BY 4.0).
