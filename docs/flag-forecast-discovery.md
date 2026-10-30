# Previsão da bandeira tarifária: relatório da Etapa 0 (descoberta)

Data: 2026-10-30. Só leitura e pesquisa; nenhum código de produção, conta ou segredo. Links consultados em 2026-10-30.
Regra de parada: a amostra de 5 boletins (mais 1 extra) foi reproduzida (seção 4). Itens que seguram a Etapa 1 até o dono decidir: seção 8.

## 1. Regra oficial

| Item | Achado | Fonte |
|---|---|---|
| Norma vigente | REH ANEEL 3.306/2024 (Anexo I = ábaco de acionamento) e REN ANEEL 1.084/2024: desde 01/04/2024 novas faixas e adicionais para despacho térmico por segurança energética fora da ordem de mérito (GFOM, decidido pelo CMSE) | [InfoBandeira 92, dez/2025](https://www.ccee.org.br/documents/80415/30980462/92%20-%20InfoBandeira%20Tarif%C3%A1ria_2025_12.pdf/77de920b-59b3-447a-010b-a8b7764bc261) |
| Metodologia-base | NT 025/2019-SGT-SRG-SRM/ANEEL (fórmulas); NT 035/2019 (garantia física "flat" no GSF); NT 021/2021 (Figura 5, ábaco); AP 08/2019 (vigência jun/2019); AP 61/2017 e NT 040/2018 (dez/2018) | mesmo boletim |
| Texto da ANEEL | **bloqueado**: `www2.aneel.gov.br/cedoc/reh20243306.html` e `ren20241084.html` devolveram HTTP 403 (não contornado). O dono precisa baixar à mão REH 3.306/2024 (Anexo I) e REN 1.084/2024 | |
| Quem faz o quê | ONS envia a GH (DECOMP, revisão 0 do PMO); CCEE calcula e publica o InfoBandeira (memória de cálculo); ANEEL divulga a cor (nos exemplos, dias 28 a 31 do mês anterior) | boletins |

Fórmulas (boletim, Tabela 1 e equações):

- `GSF_band = round(GH_band / GF_band, 2)` (2 casas, a pedido da ANEEL).
- `PLD_lim_pat = min(PLDmax, max(PLDmin, Lim_pat / (1 - GSF_band)))`. É DIVISÃO por `(1 - GSF)`; o ábaco dá `Lim_pat` constante por patamar.
- Faixas: Verde `PLDmin..L1`; Amarela `L1+0,01..L2`; Vermelha 1 `L2+0,01..L3`; Vermelha 2 `L3+0,01..PLDmax`. A cor é a faixa que contém o PLD gatilho.
- PLD gatilho: média mensal dos PLDs por submercado (CMO do DECOMP da 1ª semana operativa + expectativa da mesma rodada para as demais semanas, limitado a PLDmin/PLDmax), ponderada pela carga média mensal de cada submercado (`dadger.rv0`), com as horas por patamar de cada semana.
- `GH_band` = GH do ONS menos (ANDE + consumo interno de Itaipu), aplicadas perdas (bruta->conexão 1,335%, conexão->centro de gravidade 2,151% em 2025), mais PCH/CGH do MRE (participação 64,973%). `GF_band` = garantia física sazonalizada flat (InfoMercado dados abertos) menos perdas e fator de disponibilidade.
- GFOM (desde abr/2024): conforme o despacho térmico total (MWm), a cor sobe e o adicional muda (Tabela 2 do boletim de dez/2025):

| Cor original | Despacho térmico total (MWm), faixas sucessivas | Cor composta |
|---|---|---|
| Verde | <= 6.962 / <= 11.362 / <= 14.262 / <= 16.562 | Verde (0) / Amarela (18,85) / Vermelha 1 (44,63) / Vermelha 2 (78,77) |
| Amarela | <= 9.262 / <= 12.162 / <= 16.562 | Amarela / Vermelha 1 / Vermelha 2 |
| Vermelha 1 | <= 11.462 / <= 14.262 | Vermelha 1 / Vermelha 2 |
| Vermelha 2 | <= 11.762 | Vermelha 2 |

Limites de MWm lidos só do boletim de dez/2025; confirmar na REN 1.084.

Versões da regra que os boletins permitem afirmar:

| Versão | Vigência | Evidência |
|---|---|---|
| A | nov/2017 (extraordinária), NT 040/2018 (dez/2018), AP 08/2019 (jun/2019) | só o texto do boletim; constantes não verificadas |
| B (Lim 31,93 / 69,06 / 92,47) | ao menos out/2022 e ago/2023 | ajuste da amostra (seção 4) |
| C (Lim 27,48 / 68,99 / 95,05; com GFOM) | desde abr/2024; set/2024 a ago/2026 consistentes | idem |

A data exata da troca B->C e o histórico anterior a 2022 dependem do Anexo I da REH 3.306 e de boletins antigos (dono).

## 2. Boletins InfoBandeira (CCEE)

| Item | Achado |
|---|---|
| Quantos / desde quando | Vistos: nº 52 (ago/2022) a nº 101 (set/2026), numeração mensal contínua; nº 1 inferido ~mai/2018 (não visto). Nº 102 (out/2026) não localizado. |
| Onde | Acervo CCEE (`ccee.org.br/documents/80415/...`); página [Bandeira Tarifária](https://www.ccee.org.br/web/guest/dados-e-analises/dados-bandeira-tarifaria) com "Baixar informativo" e "Consultar histórico". Nome `NN - InfoBandeira Tarifária_AAAA_MM.pdf` |
| Formato | PDF com camada de texto (extraível por pdftotext), Tabelas 1 a 17 por mês. Sem planilha. |
| Dados abertos melhores? | [dadosabertos.ccee.org.br](https://dadosabertos.ccee.org.br/) exibe PLD_FINAL_HISTORICO (semanal) e PLD_SOMBRA (horário); **não achei** conjunto de bandeira/faixas (catálogo completo `/dataset` não verificado). O boletim cita InfoMercado `GARANTIA_FISICA_SAZO_SUBMERCADO` e `MRE_MENSAL` (insumos do GF). |
| robots.txt | [ccee.org.br/robots.txt](https://www.ccee.org.br/robots.txt): `User-Agent: *` com `Disallow:` vazio (permite). |
| Bloqueio ativo | `curl` direto aos PDFs retornou página "Acesso bloqueado" (WAF). **Não contornei.** Download automático na prática não funciona. |
| Termos | [Termos de Uso CCEE](https://www.ccee.org.br/termos-de-uso) (atualizados 13/10/2025): reprodução de conteúdo público só para fins **sem fins lucrativos**, íntegra, citando fonte, data e hora de extração; "usar apenas as ferramentas disponibilizadas no site"; proibido usar a marca CCEE em plataformas de terceiros. Rodapé dos PDFs: reprodução proibida "sem a identificação da fonte". Nenhuma licença aberta (CC-BY etc.) declarada. |

Conclusão prática: não há licença de reuso comercial. Fatos numéricos (faixas, gatilho, GSF) com atribuição são o uso mais defensável; hospedar PDFs ou uso comercial exige decisão do dono (seção 8). Não é parecer jurídico.

## 3. Disponibilidade dos insumos e prazos propostos

A = dia do anúncio da ANEEL (28 a 31 do mês anterior: 30/09/2022, 28/07/2023, 30/08/2024, 28/03/2025, 28/11/2025, 31/07/2026 nos boletins lidos). Atrasos de publicação abaixo marcados "a confirmar" não foram verificados em fonte oficial.

| Insumo | No banco | Disponível para o mês M a partir de | Observação |
|---|---|---|---|
| Cor/adicional passados | `tariff_flag_monthly` 2015-01..2026-10 | no anúncio | |
| EAR e ENA por subsistema | `subsystem_storage_daily` 2015-01-01..2026-09-30, 4 subsistemas x 4.291 dias, sem lacunas | ~D-1 (a confirmar) | bom para prazos longos |
| PLD oficial (horário, diário, semanal, mensal) | `pld_*` só **2025-01-01..2026-10-31** | semanal na semana anterior (a confirmar) | histórico curto |
| PLD sombra | **ausente** (só `variant=official`) | dados abertos CCEE | ingerir se quiserem |
| PLD da 1ª semana de M | `pld_weekly` | ~A-1 a A (mesma rodada que gera o gatilho) | quase o rótulo; vazamento se usado cedo |
| GH prevista (ONS) | **não** | ~A-2 a A (a confirmar) | proxy: ENA, geração hidro |
| GF flat | **não** | InfoMercado, mensal | quase constante (52.900 a 55.400 MWm nos boletins) |
| PLDmin/PLDmax do ano | derivável | meses antes (58,60 e 751,73 em 2025; 57,31 e 758,27 em 2026) | |
| Despacho térmico (GFOM) | `thermal_dispatch_monthly` só 2026-07..09 | decisão CMSE | fora do escopo |
| Geração/carga | `subsystem_monthly_generation` 2026-07..09; `subsystem_hourly_balance` 2026-07..10 | | insuficiente |

Prazos propostos: **21, 14, 7 e 3 dias antes de A**. Em 21 e 14: EAR/ENA, PLD recente, sazonalidade. Em 7 e 3: PLD semanal mais recente do mês corrente. Evitar prazo de 1 dia (gatilho praticamente conhecido).

Gatilho vs PLD publicado (3 pontos, indicativo):

| Mês | PLD gatilho (boletim) | PLD SE 1ª semana (`pld_weekly`) | PLD SE médio do mês (`pld_monthly`) |
|---|---|---|---|
| abr/2025 | 208,03 | 255,40 | 202,18 |
| dez/2025 | 304,02 | 333,04 | 265,89 |
| ago/2026 | 128,67 | 117,90 | 128,12 |

O gatilho não coincide com nenhum dos dois: bom proxy, não rótulo.

## 4. Amostra reproduzível (código descartável fora do repositório)

Entradas lidas do boletim: PLDmin, PLDmax, GH, GF e PLD gatilho. Calculado: GSF, faixas e cor. Arredondamento meia-unidade para cima, 2 casas.

| Boletim | PLDmin / PLDmax | GSF | Gatilho | Limites superiores V / A / V1 calculados | = boletim | Cor (calculada = boletim = `tariff_flag_monthly`) |
|---|---|---|---|---|---|---|
| nº 54 out/2022 | 55,70 / 646,58 | 0,83 | 55,70 | 187,81 / 406,23 / 543,93 | sim | Verde |
| nº 64 ago/2023 | 69,04 / 684,73 | 0,74 | 69,04 | 122,80 / 265,61 / 355,65 | sim | Verde |
| nº 77 set/2024 | 61,07 / 716,80 | 0,65 | 254,18 | 78,51 / 197,12 / 271,58 | sim | Vermelha 1 |
| nº 84 abr/2025 (extra) | 58,60 / 751,73 | 0,94 | 208,03 | 458,00 / 751,73 / 751,73 | sim | Verde |
| nº 92 dez/2025 | 58,60 / 751,73 | 0,84 | 304,02 | 171,75 / 431,20 / 594,08 | sim | Amarela |
| nº 100 ago/2026 | 57,31 / 758,27 | 0,72 | 128,67 | 98,14 / 246,40 / 339,47 | sim | Amarela |

Resultado: 6 de 6 reproduzidos (faixas ao centavo e cor); regra de parada não acionada. Ressalvas:

- As constantes `Lim` foram **ajustadas** nas faixas dos próprios boletins, não lidas do Anexo I. Versão C: 27,4798 / 68,992 / 95,0522 (intervalos viáveis 27,4795-27,4800; 68,9915-68,9925; 95,0520-95,0525), consistentes em 4 GSFs (0,65; 0,72; 0,84; 0,94), o que confirma a forma `Lim/(1-GSF)`. Versão B: 31,9275 / 69,059 / 92,4683, com só 2 boletins. Falta conferir com o Anexo I da REH 3.306/2024.
- Ago/2026: a Tabela 15 diz PLDmax 758,27 e a Tabela 16 termina a Vermelha 2 em 785,27 (inconsistência/provável erro de digitação do boletim). O motor deve tratar divergência assim como alerta.
- Set/2024: a cor foi **revisada em 04/09/2024** (Despacho ANEEL 2.668/2024); flag_rule_month precisa registrar revisão.
- Não testado: meses com GFOM > 0 (nenhum dos 6).

## 5. Tamanho do histórico (banco local, somente leitura)

| Recorte | Meses | Verde | Amarela | Vermelha 1 | Vermelha 2 | Escassez Hídrica |
|---|---|---|---|---|---|---|
| Série toda (2015-01..2026-10) | 142 | 68 | 25 | 26 | 15 | 8 |
| Desde jun/2019 (AP 08/2019) | 89 | 48 | 17 | 9 | 7 | 8 |
| Desde abr/2024 (versão C) | 31 | 14 | 9 | 5 | 3 | 0 |
| Com PLD no banco (2025-01..2026-10) | 22 | 9 | 7 | 4 | 2 | 0 |

Insumos completos hoje: EAR/ENA em todos os meses; **PLD só em 22 meses**; geração, carga e térmico só de 2026-07 em diante. O rótulo (PLD gatilho oficial) só existe nos boletins: um valor por mês, 31 na versão C. Poucos meses por cor (Vermelha 2: 3 a 7) confirma modelar o gatilho, não a cor. Para 60+ meses de PLD o dono precisa autorizar backfill (PLD_FINAL_HISTORICO semanal e horário no portal CCEE; a verificar).

## 6. PLD oficial, sombra e gatilho

- **Oficial**: preço de liquidação publicado pela CCEE por submercado; no banco, série desde 2025-01 (`pld_hourly.variant='official'`). Fonte: [dados abertos CCEE](https://dadosabertos.ccee.org.br/) (PLD_FINAL_HISTORICO, semanal).
- **Sombra**: série horária da CCEE (conjunto PLD_SOMBRA no mesmo portal), usada para comparação com o oficial; definição detalhada não verificada nesta consulta. Ausente do banco.
- **Gatilho**: média mensal ponderada por carga dos CMOs do DECOMP rev0 (1ª semana + expectativa das demais), limitados a PLDmin/PLDmax; é o único número comparado com as faixas (InfoBandeira, seção "iii").

## 7. Riscos

| Risco | Impacto | Mitigação |
|---|---|---|
| Constantes do ábaco só ajustadas | motor errado em GSF extremos | conferir Anexo I; rule_version por mês |
| Mudança de metodologia (abr/2024, GFOM) e nova possível | histórico misto | versionar; treinar na versão C ou com indicador de versão |
| GH e PLD da 1ª semana saem em A-2..A | prazo curto vira leitura | prazos 21/14/7/3 |
| PLD curto (22 meses), GH/GF ausentes | modelo fraco | backfill; critério de publicar já cobre |
| Termos CCEE (sem fins lucrativos; só ferramentas do site; WAF) | sem automação; produto comercial | decisão do dono; ingestão manual mensal de 1 PDF |
| ANEEL 403 | norma não verificada no texto | dono baixa |
| Erro de digitação em boletim; cor revisada depois | validação cruzada falsa | tabela de exceções e status |
| Placar com n pequeno (31 meses) | pouca significância | mostrar incerteza |

## 8. Decisões do dono (seguram a Etapa 1)

1. **Licença**: o produto (Plano Pro previsto) é comercial? Os termos da CCEE só liberam reprodução sem fins lucrativos. Consultar a CCEE, ou guardar só fatos numéricos com atribuição (fonte, data e hora de extração) sem hospedar PDFs.
2. **Download manual**: REH 3.306/2024 (Anexo I) e REN 1.084/2024 (ANEEL, 403); boletins nº 1 a 102 (CCEE bloqueia robôs) ou via oficial autorizada.
3. **Backfill de PLD** (semanal 2019+, horário 2021+) e PLD sombra: autorizar ingestão no spillway-data.
4. Confirmar prazos 21/14/7/3 e ingestão manual mensal dos boletins.
