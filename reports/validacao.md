# Validação com novas transcrições

## Manuais (escritas à mão)

_As regras do scoring foram calibradas olhando estes casos: resultado NÃO é independente._  (n = 12)

| Tarefa | Acurácia | F1 macro |
|---|---|---|
| produto | 1.0 | 1.0 |
| sentimento | 0.917 | 0.905 |
| churn | 1.0 | 1.0 |
| oportunidade | 1.0 | 1.0 |

Transcrições com ao menos um erro: **1 de 12**.

## Cegas (escritas à mão, nunca usadas para ajuste)

_Melhor estimativa de generalização disponível; amostra pequena, então a variância é alta._  (n = 8)

| Tarefa | Acurácia | F1 macro |
|---|---|---|
| produto | 1.0 | 1.0 |
| sentimento | 0.75 | 0.739 |
| churn | 1.0 | 1.0 |
| oportunidade | 1.0 | 1.0 |

Transcrições com ao menos um erro: **2 de 8**.

## Sintéticas (seed diferente do treino)

_Mesma distribuição do treino, portanto resultado otimista; serve como teste de regressão._  (n = 150)

| Tarefa | Acurácia | F1 macro |
|---|---|---|
| produto | 0.94 | 0.936 |
| sentimento | 0.98 | 0.979 |
| churn | 1.0 | 1.0 |
| oportunidade | 1.0 | 1.0 |

Transcrições com ao menos um erro: **12 de 150**.
