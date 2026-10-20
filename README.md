# Sprint 4 — Data Science and Statistical Computing

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Guiroselli/sprint4Nemec/blob/main/notebooks/Sprint4_Colab.ipynb)

Solução que recebe uma **nova transcrição** de atendimento e gera automaticamente informações para a área comercial:

```
Transcrição → Processamento → Modelo → Indicadores → Recomendação
```

Saída: produto identificado · sentimento · risco de churn · oportunidade comercial · principais termos · recomendação de ação.

## Aviso importante sobre os dados

O repositório não continha as transcrições reais nem os modelos das sprints anteriores. Por isso, os dados de treino
(`data/transcricoes_treino.csv`) são **sintéticos**, gerados por `src/data_gen.py`. Eles servem para demonstrar o
pipeline de ponta a ponta, mas **as métricas absolutas são otimistas**. Para usar dados reais, basta substituir o CSV
mantendo as colunas `id, transcricao, produto, sentimento, churn, oportunidade` e rodar `python -m src.train`.
Os produtos (`src/config.py`) são exemplos de um contexto bancário/seguros; ajuste para o seu domínio.

## Como executar

```bash
pip install -r requirements.txt
python -m src.data_gen      # (re)gera os dados sintéticos  — pule se usar dados reais
python -m src.manuais       # grava as transcrições escritas à mão (validação)
python -m src.train         # compara modelos, seleciona e salva models/modelos.joblib
python -m src.validate      # valida em transcrições novas -> reports/validacao.md
streamlit run app.py        # protótipo funcional
python -m src.pipeline arquivo.txt   # uso por linha de comando (ou via stdin)
```

## Estrutura

| Caminho | Papel |
|---|---|
| `src/processing.py` | Processamento: limpeza, remoção de falantes/acentos/stopwords, termos |
| `src/train.py` | Modelo: compara 4 classificadores por tarefa e seleciona o melhor |
| `src/scoring.py` | Indicadores (scores 0-100) e regras de recomendação |
| `src/pipeline.py` | Orquestra as etapas: `analisar(transcricao) -> dict` |
| `src/validate.py` | Validação com novas transcrições |
| `app.py` | Protótipo Streamlit (transcrição única e lote CSV) |
| `reports/` | Comparação de modelos, justificativa e validação |

## Mapeamento com a avaliação

| Entrega | Onde está |
|---|---|
| Pipeline completo de processamento (15) | `src/processing.py`, `src/pipeline.py` |
| Modelo final selecionado e justificativa (15) | `src/train.py`, `reports/modelo_selecionado.md`, `reports/comparacao_modelos.csv` |
| Sistema de scoring/indicadores (20) | `src/scoring.py` (seção abaixo) |
| Protótipo funcional (20) | `app.py`, `python -m src.pipeline` |
| Validação com novas transcrições (15) | `src/validate.py`, `reports/validacao.md` |
| Recomendações de negócio (10) | seção abaixo |
| Limitações e próximos passos (5) | seção abaixo |

## Modelo

TF-IDF (1-2 gramas) + classificador, um por tarefa. Foram comparados Regressão Logística, Naive Bayes, LinearSVC
calibrado e Random Forest (F1 macro, 5-fold estratificado + hold-out de 20%). Os vencedores foram Regressão Logística
(produto, sentimento) e LinearSVC calibrado (churn, oportunidade); o detalhe e a tabela completa estão em
`reports/modelo_selecionado.md`. Na tarefa de sentimento todos os modelos empataram porque o único erro restante
vem dos 4% de rótulos ruidosos que o gerador injeta de propósito.

## Scoring

Cada score vai de 0 a 100 e combina **70% probabilidade do modelo + 30% sinais léxicos** com pesos explícitos
(`SINAIS_CHURN`, `SINAIS_OPORT` em `src/scoring.py`). Os sinais tornam o score explicável ("subiu porque o cliente
citou *cancelar* e *concorrência*") e protegem contra vocabulário não visto no treino. Negações ("não quero
cancelar") limitam o score de churn a 30.

| Indicador | Faixas | Limiar de alerta |
|---|---|---|
| Risco de churn | Baixo < 35 ≤ Médio < 65 ≤ Alto | ≥ 35 (prioriza não perder churn) |
| Oportunidade | idem | ≥ 50 |
| Sentimento | -100 a +100 (P(positivo) − P(negativo)); 40% do peso vem de um léxico simples | — |

A **prioridade** (Crítica/Alta/Média/Baixa) e a ação sugerida saem de regras sobre os três indicadores:
churn alto → retenção urgente; oportunidade alta → encaminhar ao comercial com oferta por produto; oportunidade com
risco → resolver a insatisfação antes de vender; etc.

## Validação

Resultados em `reports/validacao.md` (acurácia / F1 macro):

| Conjunto | n | Produto | Sentimento | Churn | Oportunidade |
|---|---|---|---|---|---|
| Sintéticas, seed diferente | 150 | 0,94 / 0,94 | 0,98 / 0,98 | 1,00 / 1,00 | 1,00 / 1,00 |
| Manuais (regras calibradas olhando estes casos) | 12 | 1,00 / 1,00 | 0,92 / 0,91 | 1,00 / 1,00 | 1,00 / 1,00 |
| **Cegas** (escritas à mão, sem nenhum ajuste depois) | 8 | 1,00 / 1,00 | **0,75 / 0,74** | 1,00 / 1,00 | 1,00 / 1,00 |

Como ler: apenas o conjunto **cego** estima a generalização, e com 8 exemplos a incerteza é grande. Um achado
relevante: com apenas o modelo treinado em dados sintéticos, os acertos nas transcrições manuais eram bem piores
(sentimento 67%, churn 75%, oportunidade 83%). A melhora veio dos sinais léxicos e do tratamento de negação,
não de um modelo melhor — o que reforça a necessidade de treinar com transcrições reais.

## Recomendações de negócio

1. **Retenção em até 24h** para clientes com churn ≥ 65, com oferta personalizada e tratamento da causa-raiz
   (cobrança indevida, tarifas, reincidência de chamados costumam disparar o sinal).
2. **Não vender para quem está insatisfeito**: quando há oportunidade *e* risco, resolver o problema primeiro.
3. **Fila de leads** para o comercial a partir de oportunidade ≥ 50, já com o produto e a oferta sugerida.
4. **Monitorar termos** que mais aparecem entre transcrições de churn (cancelar, concorrência, tarifa, portabilidade)
   e levá-los a produto e operações.
5. **Transformar elogios em ação**: pedir NPS/indicação aos clientes de sentimento positivo.

## Limitações e próximos passos

**Limitações**
- Dados de treino sintéticos e padronizados; métricas absolutas não representam o desempenho em produção.
- Validação com poucos casos escritos à mão (12 + 8): alta variância; sem intervalo de confiança.
- Sentimento é a tarefa mais frágil fora da distribuição de treino (75% no teste cego).
- Sinais léxicos escritos manualmente: cobrem só parte das formas de dizer a mesma coisa (ironia, gírias, siglas).
- Rótulo de churn = *intenção declarada na ligação*, não churn efetivamente observado depois.
- Uma transcrição = um produto; chamadas com vários produtos ou vários assuntos perdem informação.
- Pesos 70/30 e limiares 35/50 foram definidos por julgamento, não otimizados com custo de negócio.

**Próximos passos**
1. Treinar com as transcrições reais e rotulagem revisada (medir concordância entre anotadores).
2. Ampliar o conjunto cego (idealmente 100+ chamadas reais) e reportar intervalos de confiança.
3. Rotular churn com o desfecho real (cliente cancelou em 30/60 dias) e calibrar os scores com isso.
4. Testar embeddings/transformers em português (ex.: BERTimbau) e comparar com o baseline TF-IDF.
5. Otimizar limiares pelo custo de negócio (custo de perder um cliente × custo de uma oferta desnecessária).
6. Colocar em produção: API (FastAPI), monitoramento de drift e feedback do comercial para retreinar.
