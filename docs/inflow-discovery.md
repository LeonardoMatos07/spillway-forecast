# Previsão de afluência: relatório da Etapa 0 (descoberta)

Conferido em 30/10/2026. Nenhum código de produção, nenhum treino, nenhuma conta, nenhum segredo. Banco local usado só para leitura; arquivos do ONS baixados só para medir (cerca de 36 MB de parquet, descartados).

## Resumo

| Tema | Resultado |
|---|---|
| Cobertura | 158 reservatórios. 153 com série diária de vazão natural contínua e utilizável por 10+ anos (151 com todos os anos desde 1990). Falhas: 1,15% dos dias |
| Histórico | ONS tem 1990 em diante para os 151 antigos. Estender é barato: +175 MB (desde 2000) ou +295 MB (desde 1990) no banco, arquivos de 1 a 1,7 MB por ano |
| Ano inicial proposto | **2000-01** para `HYDRO_START` (a chuva IMERG final só é confiável desde 2001; um ano de aquecimento para janelas de 365 dias) |
| Latência ONS | Cerca de 1 dia (D-1 completo; uma só medição, ver seção 3) |
| Topologia | Cadastro NÃO traz montante/jusante. Traz ordem de cascata (`num_ordemcs`) e vazão incremental; grafo explícito só por ONS (diagrama) ou ANA (BHO) |
| ENA/vazão | Razão constante em 137 de 149 reservatórios (CV < 5%); instável em 12; soma dos reservatórios nem sempre bate com o subsistema |
| Chuva recomendada | NASA POWER `imerg` diário (bucket aberto), CC BY 4.0, 0,1 grau, sem conta |
| Bacias piloto | Grande (SE) e Iguaçu (S) |
| Treino/val/teste | Treino 2002-2017, validação 2018-2020, teste 2021-2025 (seco 2021; úmido 2023 no Sul) |

## 1. Cobertura das séries (`natural_inflow_m3s`)

Banco local (`reservoir_daily`, porta 5433): 671.769 linhas, 158 reservatórios, 2015-01-01 a 2026-09-30 (o ONS já publica até 2026-10-29: o banco local está cerca de 1 mês atrás). `subsystem_storage_daily`: 17.164 linhas no mesmo período. O histórico local começa em 2015 para 156 reservatórios (os outros dois entram em 2023 e 2024).

| Medida | Valor |
|---|---|
| Reservatórios com vazão natural em algum dia | 158 de 158 |
| Falhas totais (dias sem vazão natural dentro do próprio período do reservatório) | 1,15% |
| Reservatórios com mais de 5% de falhas | 3 |
| Reservatórios com mais de 1% de falhas | 5 |
| Maior trecho contínuo, 2015+: mediana | 11,7 anos (praticamente toda a janela) |
| Com trecho contínuo de 10+ anos (2015+) | 151 |
| Vazão natural igual a zero | 9.421 linhas (verificar se é zero real ou preenchimento) |
| Vazão negativa | 0 |

Com os arquivos do ONS (anos 1990 a 2014, contagem de dias com vazão natural por reservatório e ano; ano utilizável = 350 dias ou mais):

| Série de 2025 para trás sem ano ruim | Reservatórios |
|---|---|
| Desde 1990 (36 anos) | 151 |
| Desde 2000 | 152 |
| Desde 2005 | 153 |
| Com 10+ anos utilizáveis | 153 |
| Sem série utilizável em 2025 | 1 |
| Início tardio (2023 ou 2024) | 4 (têm menos de 3 anos: ficam só no treino global por atributos estáticos, ou fora) |

Limite do método: a conta mede dias com valor, não a qualidade. Não detectei, por exemplo, trechos repetidos ou preenchidos pelo ONS.

## 2. Custo de estender o histórico

Arquivos por ano em `ons-aws-prod-opendata.s3.amazonaws.com/dataset/` (mesmo caminho que o pipeline usa).

| Conjunto | Tamanho/ano | Desde |
|---|---|---|
| `dados_hidrologicos_di` (vazões, cotas) | 0,7 MB (1970) a 1,7 MB (2026) | 1970 (134 reservatórios em 1970; 162 em 1990; 165 em 2000) |
| `ena_reservatorio_di` | cerca de 1 MB | 2000 |
| `ear_reservatorio_di` | cerca de 0,8 MB | 2000 |
| `ena_subsistema_di` / `ear_subsistema_di` | menos de 0,1 MB | 2000 |

| Cenário | Linhas novas em `reservoir_daily` | Espaço no banco (estimado a 211 B/linha, índices incluídos: hoje 142 MB para 672 mil linhas) | Download |
|---|---|---|---|
| Desde 2000 | cerca de 0,84 milhão | cerca de +175 MB | cerca de 50 MB |
| Desde 1990 (só vazões; ENA/EAR não existem antes de 2000) | cerca de 1,4 milhão | cerca de +295 MB | cerca de 70 MB |

Tempo: dominado pela rede e por `to_sql`; não medi a carga completa (não é permitido escrever). Estimativa: minutos, no mesmo ritmo do backfill de 2015 já feito. Impacto: o banco passa de 285 MB para cerca de 460 MB; tabelas ainda pequenas. Sem mudança de schema.

**Proposta:** `HYDRO_START = 2000-01`. Motivos: cobre as crises hídricas de 2001, 2014-2015 e 2021 dentro de treino ou teste; EAR e ENA existem desde 2000; chuva final desde 2001. Série de 1990 a 1999 só entraria nas referências sem chuva e sem ENA; ganho pequeno, custo extra: **não proposto**. A decisão de ampliar é do usuário (ver Decisões).

## 3. Latência de publicação do ONS

| Observação | Resultado |
|---|---|
| Portal `dados.ons.org.br/dataset/dados-hidrologicos-res` | "Atualização diária" às 9:15, 14:15 e 17:00; sem atraso declarado |
| Arquivo `DADOS_HIDROLOGICOS_RES_2026.parquet` em 30/10/2026 | Último dia completo: 29/10 (177 linhas, 173 com vazão natural, 177 com afluente). Dia 30/10 vinha com apenas 6 reservatórios, só com vazão natural |
| Conclusão | Atraso médio cerca de 1 dia; dia de origem = D-1. Pior caso **não medido**: foi uma só fotografia |
| Pendência | A etapa 1 deve registrar a data em que cada dia aparece pela primeira vez (`first_seen_at`) para medir média e pior caso por vários dias |

Também no mesmo arquivo: de 23 a 27/10, 172 ou 173 de 175 reservatórios têm vazão natural (cerca de 1,5% de falha corrente). A página do ONS avisa que os dados passam por consistência recorrente e podem ser atualizados depois (risco de revisão, seção 10).

## 4. Topologia de bacia

| Fonte | Montante/jusante? | Licença | Observação |
|---|---|---|---|
| Cadastro ONS `RESERVATORIOS.parquet` (colunas: bacia, rio, posto, REE, cotas, volumes, produtibilidade, lat/lon) | **Não** | CC-BY (portal ONS) | Só bacia e rio |
| `dados_hidrologicos_di` | **Parcial**: `num_ordemcs` (ordem de cascata, de 1 a 46 na amostra) e `val_vazaoincremental` | CC-BY | Em 28/10/2026, Grande: Camargos 104,57 m3/s natural, Itutinga 104,58 com incremental 0,01: natural a jusante = natural a montante + incremental. Dá para inferir arestas e validar, mas não é lista oficial |
| ANA, Base Hidrográfica Ottocodificada (BHO 6) | Sim (código Otto Pfafstetter; trechos a montante e a jusante), GeoPackage sem login | Página diz só "acesso livre", **sem licença explícita** | Exige casar cada barragem (lat/lon) a um trecho. Também dá a área de contribuição para a média de chuva |
| ONS, diagrama esquemático das usinas | Sim (PDF) | Não verificado | Não consultei; pode servir de conferência manual |

Conclusão: **montante fica fora da v1 geral**. Para os pilotos, montar à mão a cascata (Grande: 15 reservatórios, ordens 32 a 46; Iguaçu: 8) a partir de `num_ordemcs` e vazão incremental, com conferência do dono contra o diagrama do ONS. Para todos os reservatórios: BHO, **só depois de resposta sobre licença** (parar e perguntar).

Fontes: [ONS dados hidrológicos](https://dados.ons.org.br/dataset/dados-hidrologicos-res) e [ANA BHO 6](https://metadados.snirh.gov.br/geonetwork/srv/api/records/32e309da-a8c1-443f-90ac-0cd79ce6a33d), consultadas em 30/10/2026.

## 5. Relação ENA / vazão natural

Medida: `ena_gross_mw_avg / natural_inflow_m3s`, dias com vazão acima de 1 m3/s, 623.892 pares, 152 reservatórios (149 com mais de 1.000 dias).

| Resultado | Valor |
|---|---|
| CV (desvio/média) mediano da razão por reservatório | 0,0007% (constante) |
| Reservatórios com CV abaixo de 5% | 137 de 149 |
| CV acima de 20% | 4: AMUSPM (0,54), RLFONT (0,40), RLNPEC (0,36), RLPPAS (0,35). Prefixo RL sugere o complexo de Lajes/Paraíba do Sul, com bombeamento; AMUSPM não investigado |
| CV entre 5% e 20% | 8, ex.: PBILHP, TIBONI, TIBARI, IGSEGR (razão de 0,08 a 0,12 de CV, correlação ainda 0,995+) |
| Correlação ENA x vazão, mediana | 0,99999; mínima 0,43 (RLNPEC) |
| Pilotos: Grande e Paranapanema | CV 0,0000 (razão fixa); Iguaçu: CV médio 0,8%, máximo 6,5% |

Leitura: ENA por reservatório é essencialmente vazão natural x produtibilidade fixa; derivar ENA da vazão prevista é viável com uma constante por reservatório (mediana da razão recente), **exceto** nos cerca de 12 reservatórios acima. Subsistema: a soma da ENA dos reservatórios dividida pela ENA do subsistema tem média 1,01 (N), 1,02 (SE), 0,98 (S) mas **1,18 (NE)**, com faixa diária larga (N 0,38 a 1,43; NE 0,77 a 1,99). Ou seja, o agregado do subsistema não é soma simples; precisa de calibração própria (ou de série própria) na etapa 1. ENA só entra se esta razão passar no critério da etapa 3.

## 6. Chuva: fontes candidatas

| Fonte | Licença | Conta? | robots.txt | Resolução | Atraso | Histórico | Veredito |
|---|---|---|---|---|---|---|---|
| **NASA POWER `imerg`** (bucket `nasa-power`, `imerg/temporal/power_imerg_daily_temporal_utc.zarr`) | CC BY 4.0 (`LICENSE.txt` do bucket); citar POWER e IMERG | Não (HTTPS anônimo, S3 sem assinatura) | O bucket S3 não tem `robots.txt` (404); o `robots.txt` do site proíbe `/api/`, que não usamos | 0,1 grau (cerca de 10 km), diário em UTC | Último dia no bucket: 28/10/2026 (D-2). Dias recentes são IMERG "Late" (14 h) e depois substituídos pela "Final" (cerca de 3,5 meses) | Zarr desde 1998-01-01; POWER documenta Final desde 2001-01 | **Recomendada** |
| GPM IMERG Final no AWS (GES DISC) | CC BY 4.0 | **Sim**: credenciais S3 do GES DISC (Earthdata) | n/a | 0,1 grau | até 4 meses | 1998+ | Fora; o dono precisaria criar conta |
| CHIRPS v2 (UCSB CHC) | Domínio público (declarado na página) | Não declarado | `data.chc.ucsb.edu/robots.txt`: **`Disallow: /`** | 0,05 grau, 50S a 50N | não descrito | 1981+ | **Bloqueada**: o dono teria de baixar à mão ou pedir autorização. Aviso na página: v2 termina após dez/2026, v3 já existe |
| ERA5 (AWS `era5-pds`, deprecado; rehospedado pelo NCAR) | Termos Copernicus (não li o texto completo) | Não, no bucket | n/a | cerca de 31 km | mensal, sem latência declarada | 1940+ | Descartada: grossa e lenta para uso operacional |
| INMET | Sem licença declarada | n/a | n/a | estações | n/a | 2000+ | Fora (aguarda resposta) |
| INPE/CPTEC MERGE | Não verificada | n/a | n/a | 0,1 grau | n/a | n/a | Não avaliada: o caminho testado devolveu 404; o dono pode indicar a URL oficial |

**Descoberta do `imerg` (conjunto ainda não inspecionado antes):** variáveis `IMERG_PRECTOT` (mm/dia), `IMERG_PRECTOT_COUNT` e `IMERG_PRECLIQUID_PROB`; grade 1800 x 3600 (0,1 grau); 11.688 passos de tempo; blocos Zarr de 1461 dias x 50 x 100 pixels (5 graus de latitude x 10 de longitude). Dois blocos de teste pesam 22 e 53 KB (provavelmente polares: os de terra tropical serão maiores). Estimativa: o Brasil inteiro cabe em cerca de 32 blocos espaciais x 8 de tempo = 256 objetos, de dezenas a poucas centenas de MB; uma bacia piloto, poucos MB. Zarr é lido por bloco, sem baixar o globo.

**Extração da média por área:** v1 = média de uma janela (por exemplo 3 x 3 pixels) ao redor da barragem, simples, mas só aproximada; v2 = média ponderada sobre a área de contribuição incremental (polígonos Otto da ANA, depende da licença da seção 4). Observações: dia UTC contra dia civil brasileiro (3 h de diferença); mistura Late/Final: treinar com Final e servir com Late muda a distribuição. Mitigação: avaliar o modelo também com Late nos últimos meses; **não medi** o tamanho dessa diferença.

Fontes (consultadas em 30/10/2026): [AWS Registry NASA POWER](https://registry.opendata.aws/nasa-power/), [fontes de dados do POWER](https://power.larc.nasa.gov/docs/methodology/data/sources/), [GPM IMERG diário no AWS](https://registry.opendata.aws/nasa-gpm3imergdf/), [CHIRPS](https://www.chc.ucsb.edu/data/chirps), [ERA5 no AWS](https://registry.opendata.aws/ecmwf-era5/), `last_available.json` e `.zattrs` do bucket `nasa-power`.

## 7. Bacia piloto

| Bacia | Reservatórios | Motivo |
|---|---|---|
| **Grande** (SE) | 15, cascata clara (ordens 32 a 46), Camargos, Furnas, Marimbondo, Água Vermelha; 31 mil hm3 úteis | Cobertura de 100%, ENA proporcional (CV 0), muita regularização, cascata curta para testar montante, SE é o maior subsistema |
| **Iguaçu** (S) | 8 | Resposta rápida à chuva (teste duro para o horizonte curto), contraste seco/úmido na janela de teste (Sul: 2021 seco, 2023 e 2024 úmidos), ENA com CV 0,8% (máx. 6,5%) |

Alternativa: Paranapanema (12 reservatórios, SE, razão fixa). Tocantins/São Francisco têm os maiores volumes, mas poucos reservatórios e menos sinal para um modelo global.

## 8. Anos de validação e teste

ENA média (% da MLT) por subsistema, banco local e arquivos do ONS:

| Ano | SE | S | NE | N |
|---|---|---|---|---|
| 2011 | 118 | 155 | 85 | 108 |
| 2014 | 74 | 137 | 51 | 92 |
| 2020 | 78 | 51 | 71 | 95 |
| 2021 | 72 | 66 | 52 | 94 |
| 2022 | 84 | 117 | 85 | 118 |
| 2023 | 93 | 163 | 59 | 79 |
| 2024 | 68 | 140 | 52 | 60 |
| 2025 | 76 | 89 | 48 | 73 |
| 2026 (até set.) | 93 | 127 | 93 | 69 |

Divisão proposta (anos hidrológicos de out a set; janela de 365 dias: primeira origem em 2002):

| Papel | Anos | Observação |
|---|---|---|
| Treino | 2002 a 2017 | inclui o SE úmido de 2011 e a seca de 2014-2015 |
| Validação (parada antecipada) | 2018 a 2020 | inclui o Sul muito seco de 2020 |
| Teste | 2021 a 2025 | seco: 2021 (SE 72, S 66) e 2024 (SE 68); úmido: 2022-2023 e 2024 no Sul |
| Reserva "futura" | 2026 até a data da carga | nunca usada para decidir nada |

Limite honesto: nos últimos anos **não há SE realmente úmido** (máximo 93% em 2023). Por isso o piloto Iguaçu leva o teste úmido; para o Grande, ou aceita-se teste sem ano úmido, ou usa-se validação cruzada por blocos incluindo 2011 (decisão). Treino de produção usa todos os anos.

## 9. Estimativas (sem treino executado)

Hipóteses: janela N = 365 dias; cerca de 12 entradas por dia (vazões, EAR, chuva, calendário); LSTM de uma camada com 64 unidades, cabeça de quantis 14 x 3; origem diária. Custo por amostra: 4 x 64 x (64+12) = 19,5 mil multiplicações por passo x 365 = 7,1 milhões na ida, cerca de 21 milhões com a volta (cerca de 43 MFLOP).

| Cenário | Amostras por época | Trabalho por época | CPU (20 a 50 GFLOP/s efetivos em LSTM) | GPU |
|---|---|---|---|---|
| Piloto Grande (15 reservatórios x cerca de 9.000 dias) | 135 mil | cerca de 6 TFLOP | 2 a 5 min | desnecessária |
| Piloto, 30 épocas x 5 sementes | | | **5 a 12 h**; com amostragem a cada 3 dias, 2 a 4 h | |
| Todos (158 reservatórios) | 1,4 milhão | cerca de 60 TFLOP | 20 a 50 min por época; 30 épocas x 5 sementes = **dezenas de horas a dias: inviável em CPU** | GPU de classe T4: cerca de 5 a 10 vezes mais rápida, 5 a 10 h por conjunto |
| Árvores com quantis por horizonte (14 x 3 modelos) | | | minutos a dezenas de minutos | |

Custo na nuvem: não contratei nada. Ordem de grandeza: uma GPU de classe T4 sob demanda custa na faixa de poucos reais por hora (preço de tabela **não consultado**; conferir antes) e 5 a 10 h por conjunto de sementes dá algumas dezenas de reais. Qualquer instância paga exige aprovação do dono.

## 10. Riscos

| Risco | Efeito | Mitigação |
|---|---|---|
| Vazão natural é série naturalizada e revisada pelo ONS depois da publicação | Alvo muda; treino e avaliação enganosos | Guardar fotografias com `first_seen_at`; avaliar contra o valor visto na data; aviso na tela |
| Mudança de metodologia da vazão natural entre décadas | Quebra estrutural; a série de 1990 pode não ser comparável | Não usar antes de 2000; teste de estabilidade; ano de início é decisão do dono |
| Chuva: mistura Late/Final do IMERG | Treino diferente do serviço | Medir a diferença; treinar com ruído ou só com Late recente |
| Chuva: bias do satélite sobre bacias de montanha; janela pontual no lugar de área | Sinal fraco, sobretudo no Iguaçu | Área de contribuição (BHO) na v2 |
| Sem chuva futura | Modelo só tem bom desempenho nos primeiros dias | Já previsto na spec; avisos fixos |
| Sem topologia explícita | Montante só no piloto, manual | Seção 4 |
| ENA do subsistema não é soma simples | Alvo derivado do subsistema pode errar | Calibração própria ou fora da v1 |
| Reservatórios novos (4 com menos de 3 anos) e 12 com razão ENA instável | Poucos dados; extras | Treino global com estáticos; excluir da métrica ENA |
| Banco local 1 mês atrás | Divergência com o ONS | Rodar o refresh antes da etapa 1 |
| Zeros (9.421 linhas) | Log da vazão quebra; zero real ou preenchimento | Investigar; usar log1p ou piso |

## 11. Decisões do usuário

1. Ampliar `HYDRO_START` de 2015-01 para 2000-01 (cerca de +175 MB, +0,84 milhão de linhas). Opcional e não proposto: 1990.
2. Fonte de chuva: NASA POWER `imerg` (CC BY 4.0, sem conta). Confirmar. CHIRPS só se você baixar à mão ou obtiver autorização do CHC.
3. Topologia: aceitar montante manual só nos pilotos; para todos, decidir sobre a BHO da ANA (licença não declarada na página; perguntar à ANA) ou o diagrama do ONS.
4. Bacias piloto: Grande e Iguaçu.
5. Divisão de anos (seção 8), em especial como obter um teste úmido para o Sudeste.
6. GPU/nuvem: nada será contratado sem aprovação.
7. Rodar `refresh` do banco local antes da etapa 1.

## Reprodução

Consultas em `reservoir_daily`, `reservoir` e `subsystem_storage_daily` (somente leitura); arquivos `dados_hidrologicos_di/DADOS_HIDROLOGICOS_RES_{ano}.parquet` (1970, 1980, 1990 a 2014, 2026), `ena_subsistema_di/ENA_DIARIO_SUBSISTEMA_{2000..2014}.parquet`, `reservatorio/RESERVATORIOS.parquet` no bucket do ONS; `GET` e `HEAD` anônimos no bucket `nasa-power` (`.zmetadata`, `.zattrs`, `last_available.json`, dois blocos). Previsão estatística; nada neste relatório é previsão oficial do ONS.
