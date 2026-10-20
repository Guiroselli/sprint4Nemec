# Análise das reuniões reais (Proton Flow)

- Transcrições únicas analisadas: **26** (de 30 registros; duplicatas removidas)
- Tamanho: mediana de **5133** palavras (mín. 1150, máx. 17871)
- Sem produto identificado (poucos termos de domínio): **1**

## Distribuição

**Produto dominante**: rh: 13, protheus: 8, crm: 3, supply: 1

**Sentimento**: positivo: 15, neutro: 9, negativo: 2

**Risco de churn**: Baixo: 24, Médio: 2

**Prioridade**: Baixa: 19, Média: 4, Alta: 3

## Comparação com a análise antiga

Concordância no produto principal: **11%** em 9 reuniões com os dois resultados. A análise antiga indicou `fluig` em 8 das 12 reuniões que ela analisou com transcrição; ela é um ponto de comparação, não um gabarito, então a divergência não prova erro de nenhum dos dois lados.

## Reuniões de maior risco

     id cliente  produto  risco_churn                                         sinais_churn
1057714  TFDRE6 protheus         52.5        cancelamento/rescisão (x2); concorrência (x1)
1027294  CXHNWJ       rh         50.0 cancelamento/rescisão (x1); troca de fornecedor (x1)
 939460  TFDRE6 protheus         17.5                                    insatisfação (x1)
 715850  TDC50S protheus         17.5                                    insatisfação (x1)
1075117  TEZAPV       rh         17.5                                    insatisfação (x1)

## Como validar de verdade

Preencha `reports/revisao_humana.csv` (produto correto, sentimento, churn e oportunidade reais) e compare com as colunas previstas. Conferência manual feita neste trabalho: as duas reuniões de maior risco (ids 1057714 e 1027294, ambas nível Médio) têm falas reais de saída ("trocar de fornecedor", "decisão é encerrar o projeto").