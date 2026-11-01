# Model card: previsão da bandeira tarifária

Status: **código pronto, avaliação real pendente** (precisa do banco; ver o comando em
`docs/flag-publication-criterion.md`). Nenhum resultado real consta aqui até a avaliação rodar.
Este arquivo será atualizado com os números de `reports/flag/results.json`, sem mexer no critério.

## O que é

Probabilidade de cada cor (Verde, Amarela, Vermelha 1, Vermelha 2) da bandeira do **próximo mês**,
quatro prazos (21, 14, 7 e 3 dias antes do anúncio da ANEEL). Não é a decisão da ANEEL nem
aconselhamento financeiro. A previsão aparece sempre ao lado da bandeira oficial.

## Como é calculada

1. Modelo do PLD gatilho do mês (regressão com incerteza) e do GSF do mês, a partir de PLD
   semanal, EAR, ENA, cor do mês anterior e mês do ano.
2. A regra oficial (REH 3.306/2024 e boletim InfoBandeira da CCEE) transforma PLD mín, PLD máx e
   GSF em faixas; a cor é a faixa que contém o gatilho (`flag_rule.py`). O motor reproduz as
   faixas e as cores dos 37 boletins lidos (`docs/flag-rule-reproduction.md`).
3. A probabilidade de cada cor é a fração de cenários (gatilho x GSF) que cai em cada faixa.

## Limites conhecidos

- Não temos as rodadas do DECOMP nem a previsão de geração hidrelétrica do ONS: o gatilho e o GSF
  são aproximados com EAR, ENA e PLD já publicados.
- As constantes da regra foram ajustadas nas faixas dos boletins, não lidas do Anexo I da REH
  3.306/2024. Nenhum mês da amostra tem GFOM maior que zero.
- Amostra: 37 meses com gatilho oficial (Verde 29, Amarela 7, Vermelha 1: 1, Vermelha 2: 0) e no
  máximo 25 avaliados por prazo. Vermelha 2 nunca apareceu: não pode ser validada.
- Gatilho e GSF são tratados como independentes dado os insumos (simplificação).
- Fonte dos fatos dos boletins: CCEE, InfoBandeira Tarifária (só números, sem PDFs). É preciso
  parecer jurídico antes de monetizar.

## Critério de publicação

Fixado antes dos resultados em `docs/flag-publication-criterion.md` e em `flag_eval.GATE`.
Prazos que não passarem não são publicados.

## Resultados

Pendentes (preencher a partir de `reports/flag/results.json`).
