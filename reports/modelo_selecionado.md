# Modelo final selecionado e justificativa

Para cada tarefa (produto, sentimento, risco de churn, oportunidade comercial) foram comparados 4 classificadores sobre a mesma representação TF-IDF (unigramas+bigramas). Critério: **F1 macro em validação cruzada estratificada (5 folds)**, com conferência em hold-out de 20%. Em empate técnico (< 0,005) prevalece o modelo mais simples e interpretável.

## Produto: LogisticRegression

- F1 macro CV: **0.9504** (±0.0158) · hold-out F1: **0.904** · acurácia hold-out: **0.9062**

| modelo | f1_macro_cv | desvio_cv | f1_macro_holdout | acuracia_holdout |
|---|---|---|---|---|
| LogisticRegression | 0.9504 | 0.0158 | 0.904 | 0.9062 |
| NaiveBayes | 0.9028 | 0.0294 | 0.8723 | 0.875 |
| LinearSVC (calibrado) | 0.9535 | 0.0126 | 0.9165 | 0.9187 |
| RandomForest | 0.9473 | 0.0057 | 0.9116 | 0.9125 |

## Sentimento: LogisticRegression

- F1 macro CV: **0.9609** (±0.0138) · hold-out F1: **0.9748** · acurácia hold-out: **0.975**

| modelo | f1_macro_cv | desvio_cv | f1_macro_holdout | acuracia_holdout |
|---|---|---|---|---|
| LogisticRegression | 0.9609 | 0.0138 | 0.9748 | 0.975 |
| NaiveBayes | 0.9609 | 0.0138 | 0.9748 | 0.975 |
| LinearSVC (calibrado) | 0.9609 | 0.0138 | 0.9748 | 0.975 |
| RandomForest | 0.9609 | 0.0138 | 0.9748 | 0.975 |

## Churn: LinearSVC (calibrado)

- F1 macro CV: **1.0** (±0.0) · hold-out F1: **1.0** · acurácia hold-out: **1.0**

| modelo | f1_macro_cv | desvio_cv | f1_macro_holdout | acuracia_holdout |
|---|---|---|---|---|
| LogisticRegression | 0.9909 | 0.0086 | 1.0 | 1.0 |
| NaiveBayes | 0.8003 | 0.0237 | 0.83 | 0.8625 |
| LinearSVC (calibrado) | 1.0 | 0.0 | 1.0 | 1.0 |
| RandomForest | 0.9385 | 0.0202 | 0.9648 | 0.975 |

## Oportunidade: LogisticRegression

- F1 macro CV: **0.9959** (±0.005) · hold-out F1: **1.0** · acurácia hold-out: **1.0**

| modelo | f1_macro_cv | desvio_cv | f1_macro_holdout | acuracia_holdout |
|---|---|---|---|---|
| LogisticRegression | 0.9959 | 0.005 | 1.0 | 1.0 |
| NaiveBayes | 0.8787 | 0.0294 | 0.8691 | 0.8875 |
| LinearSVC (calibrado) | 1.0 | 0.0 | 1.0 | 1.0 |
| RandomForest | 0.9795 | 0.0093 | 0.9836 | 0.9875 |

## Por que esta abordagem

- **Poucos dados e texto curto**: modelos lineares sobre TF-IDF costumam igualar ou superar modelos mais complexos, treinam em segundos e dispensam GPU.
- **Interpretabilidade**: os coeficientes mostram quais termos puxam a decisão, o que ajuda a área comercial a confiar na recomendação.
- **Probabilidades calibradas** alimentam diretamente o sistema de scoring (0-100).
- **Cautela**: os dados atuais são sintéticos; os números absolutos são otimistas e devem ser reavaliados com as transcrições reais (ver `README.md`).