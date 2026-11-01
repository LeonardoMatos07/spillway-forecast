# Previsão da bandeira: critério de publicação (fixado antes de ver resultados)

Etapa 4. Este critério foi escrito e gravado em código (`flag_eval.GATE` e `flag_eval.evaluate_gate`)
**antes** de qualquer avaliação com dados reais. Não será ajustado depois de ver os números. Se
for preciso mudar, o commit tem de explicar o motivo e o `MODEL_CARD_FLAG.md` tem de registrar a
mudança. Quem decide a regra é o dono; o código só a aplica.

## O que se avalia

- **Alvo**: o PLD gatilho oficial do mês (regressão com incerteza) e o GSF do mês. A cor não é
  rótulo: a probabilidade de cada cor vem das faixas daquele mês (motor `flag_rule`) aplicadas a
  amostras do gatilho e do GSF previstos. Os limites PLD mín e máx do mês são regulatórios e
  conhecidos com antecedência.
- **Prazos**: 21, 14, 7 e 3 dias antes do anúncio da ANEEL (`announced_on` da semente). Cada
  prazo é avaliado e publicado (ou não) sozinho.
- **Insumos** (só o que existia no corte): PLD semanal do SE (último e média de 4 semanas), EAR do
  SE e média dos 4 subsistemas, ENA % da MLT do SE (30 dias), variação da EAR em 30 dias, cor do
  mês anterior, mês do ano. Dados do dia D só entram até D-1.
- **Origem móvel, janela crescente**: para prever o mês M, treina-se só com meses cujo gatilho já
  era legível no corte (início do mês + 10 dias, aproximação da data do boletim), com mínimo de 12
  meses. Nenhum ajuste de hiperparâmetro (alfa do ridge, grade de quantis e profundidade do
  boosting são fixos no código).
- **Candidatos, na ordem da spec**: (1) ridge com resíduos leave-one-out, (2) regressão quantílica
  linear, (3) boosting pequeno de quantis, **só testado** quando um candidato linear tem Brier
  médio abaixo da melhor referência (olhada descritiva, sem selecionar por ela).
- **Referências que o modelo tem de vencer**: (1) mesma cor do mês anterior; (2) distribuição
  histórica da cor para o mês do ano (série ANEEL de 2015 em diante, suavização +0,5); (3) PLD
  atual como gatilho (último PLD semanal do SE, GSF pela mediana do modelo).
  Acrescentei uma quarta, mais dura que a spec: (4) frequência geral das cores nos mesmos meses de treino do
  modelo (suavização +0,5). Sem ela, um modelo sem informação alguma vencia a climatologia por mês do
  ano só por entregar a distribuição geral (visto num teste com dados embaralhados).

## Critério (todas as condições, por prazo e por candidato)

Contra a **melhor referência daquele prazo** (escolhida nos próprios meses avaliados, a escolha
mais dura para o candidato):

1. pelo menos **24 meses** avaliados no prazo;
2. Brier médio (soma sobre as 4 cores, de 0 a 2) pelo menos **5% menor**;
3. bootstrap pareado em blocos de 3 meses (2.000 reamostragens, semente fixa) da diferença mensal
   de Brier: o quantil `1 - 0,10/k` da diferença fica **abaixo de zero**, com `k` o número de
   candidatos testados naquele prazo (correção de Bonferroni; nível unilateral total de 10%);
4. acerto da cor mais provável **não pior** que o da melhor referência;
5. erro de calibração esperado (ECE, 5 faixas, pooled um-contra-todos) de **no máximo 0,15**;
6. o motor reproduz **100%** dos boletins usados (princípio 5 da spec: se divergir, para).

**Prazos que não passam não são publicados.** Se nenhum passar, a página `/bandeira` abre só com
calculadora (simulador), histórico e placar, sem probabilidades previstas.

## Poder do teste com a amostra que temos

Fatos, da semente (`flag_rule_month`) e da série da ANEEL:

| Item | Valor |
|---|---|
| Boletins lidos | 37 (nº 52 a 102); 14 meses do intervalo estão pendentes de leitura |
| Por regra | 19 na versão B e 18 na versão C |
| Cores nos 37 meses | Verde 29, Amarela 7, Vermelha 1: 1, **Vermelha 2: 0** |
| Máximo de meses avaliados por prazo (37 menos 12 de treino mínimo) | 25 (17 Verde, 7 Amarela, 1 Vermelha 1, 0 Vermelha 2) |
| Brier da melhor previsão constante (frequências da amostra) | 0,458 |

Consequências, com os números:

- O piso de 24 meses deixa folga de **1 mês**: qualquer mês sem PLD ou EAR no corte derruba o
  prazo.
- Menor diferença verdadeira de Brier detectável (teste pareado unilateral, nível 10%, poder
  80%, aproximação normal): `(z0,90 + z0,80) × sd / √n = 2,12 × sd / √25`. Com desvio-padrão da
  diferença mensal de Brier de **0,35** (premissa; o valor medido sai da avaliação real e é
  gravado em `sd_paired_diff` e `mdd`): **0,149**, ou seja, 32% do Brier de 0,458. Com 0,50:
  0,212 (46%). O piso de ganho de 5% está bem abaixo do que o teste consegue distinguir de
  sorte, então na prática o que decide é o item 3.
- Com 1 mês de Vermelha 1 e nenhum de Vermelha 2, **a probabilidade dessas cores não pode ser
  calibrada nem validada**. O Brier depende quase só de Verde contra Amarela (24 de 25 meses).
- Com 18 meses na versão C, qualquer modelo que dependa de comportamento pós-abr/2024 treina com
  poucos pontos.

**Expectativa honesta, escrita antes de rodar**: é improvável que algum prazo passe. Se nenhum
passar, a decisão já está tomada: `/bandeira` abre só com calculadora, histórico e placar, e a
previsão fica "em avaliação" até haver mais meses (cada mês novo acrescenta um ponto; os 14
boletins pendentes e os de antes de ago/2022 ajudam bastante).

## Como rodar com o banco

Depois que o Docker voltar e o Postgres local estiver no ar:

```
uv run spillway flag-evaluate --dsn "postgresql://warehouse:warehouse@localhost:5433/warehouse" --out reports/flag
```

Só faz SELECT em `pld_weekly`, `subsystem_storage_daily` e `tariff_flag_monthly`. Escreve
`reports/flag/results.json` (métricas, resultado do critério e prazos publicáveis por candidato),
`scored.csv` (probabilidades e escores por mês, prazo e modelo) e `summary_<prazo>.csv`.
Sem banco, `uv run spillway flag-backtest --out docs/flag-rule-reproduction.md` refaz o relatório
de reprodução da regra.
