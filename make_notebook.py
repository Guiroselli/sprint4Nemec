"""Gera notebooks/Sprint4_Colab.ipynb (rodar: python make_notebook.py)."""
import json
from pathlib import Path


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(True)}


cells = [
    md("""
# Sprint 4 — Data Science and Statistical Computing
**Transcrição → Processamento → Modelo → Indicadores → Recomendação**

Este notebook roda de ponta a ponta no Google Colab (Ambiente de execução > Executar tudo).
Atenção: os dados de treino são **sintéticos** (o repositório não tinha as transcrições reais);
para usar dados reais, substitua `data/transcricoes_treino.csv` (colunas `id, transcricao, produto, sentimento, churn, oportunidade`).
"""),
    md("## 1. Preparação (clona o repositório e instala dependências)"),
    code("""
import os
if not os.path.exists("sprint4Nemec"):
    !git clone -q https://github.com/Guiroselli/sprint4Nemec.git
%cd sprint4Nemec
!pip -q install scikit-learn pandas numpy joblib
"""),
    md("## 2. Dados e processamento"),
    code("""
import pandas as pd
from src import config
from src.processing import limpar, texto_processado, termos_frequentes

df = pd.read_csv(config.TREINO_CSV)
print(df.shape)
display(df.head(3))
print(df.produto.value_counts(), "\\n")
print(df.sentimento.value_counts())

ex = df.transcricao[0]
print("\\n--- ORIGINAL ---\\n", ex)
print("\\n--- PROCESSADO ---\\n", texto_processado(ex))
"""),
    md("""
## 3. Modelo: comparação e seleção
Quatro classificadores (Regressão Logística, Naive Bayes, LinearSVC calibrado, Random Forest) sobre TF-IDF,
para cada tarefa (produto, sentimento, churn, oportunidade). Critério: F1 macro em validação cruzada de 5 folds.
"""),
    code("""
!python -m src.train
comp = pd.read_csv("reports/comparacao_modelos.csv")
display(comp)
"""),
    code("""
print(open("reports/modelo_selecionado.md", encoding="utf-8").read())
"""),
    md("## 4. Indicadores e recomendação para uma nova transcrição"),
    code("""
from src.pipeline import analisar, formatar

transcricao = \"\"\"
Atendente: Bom dia, em que posso ajudar?
Cliente: Estou muito irritado, o cartão foi cobrado duas vezes na fatura e ninguém resolve. Já liguei três vezes.
Atendente: Peço desculpas, vou abrir um protocolo.
Cliente: Se não estornarem, eu cancelo e vou para a concorrência.
\"\"\"
print(formatar(analisar(transcricao)))
"""),
    md("### Teste com sua própria transcrição\nEdite o texto da caixa e rode a célula."),
    code("""
#@title Cole a transcrição aqui { display-mode: "form" }
texto = "Atendente: Ol\\u00e1!\\nCliente: Quero investir mais, o que voc\\u00eas t\\u00eam com melhor rentabilidade? Tenho interesse em previd\\u00eancia." #@param {type:"string"}
print(formatar(analisar(texto.replace("\\\\n", "\\n"))))
"""),
    md("""
## 5. Validação com novas transcrições
- **Sintéticas**: seed diferente do treino (otimista, teste de regressão).
- **Manuais**: escritas à mão; as regras foram calibradas olhando estes casos (não independente).
- **Cegas**: escritas à mão, nunca usadas para ajuste → melhor estimativa de generalização (n pequeno).
"""),
    code("""
!python -m src.validate
print(open("reports/validacao.md", encoding="utf-8").read())
"""),
    code("""
d = pd.read_csv("reports/validacao_detalhe_cegas.csv")
display(d[["produto", "pred_produto", "sentimento", "pred_sentimento", "churn", "pred_churn",
           "oportunidade", "pred_oportunidade", "risco_churn", "score_oportunidade", "prioridade"]])
"""),
    md("## 6. Painel em lote (todas as transcrições novas)"),
    code("""
novas = pd.read_csv(config.NOVAS_CEGAS_CSV)
res = pd.DataFrame([analisar(t) for t in novas.transcricao])
painel = pd.concat([novas[["id"]], res[["produto", "sentimento", "risco_churn", "oportunidade", "prioridade", "recomendacao"]]], axis=1)
display(painel)
print(painel.prioridade.value_counts())
painel.prioridade.value_counts().plot(kind="bar", title="Transcrições por prioridade");
"""),
    md("""
## 7. Recomendações de negócio, limitações e próximos passos
Veja o `README.md` do repositório para o detalhamento completo:
https://github.com/Guiroselli/sprint4Nemec#readme

**Resumo:** retenção em até 24h para churn ≥ 65; não vender para cliente insatisfeito (resolver primeiro);
fila de leads para oportunidade ≥ 50; monitorar termos de churn.
**Limitações:** dados sintéticos, validação cega com apenas 8 casos, sentimento é a tarefa mais frágil (75% no cego).
**Próximos passos:** treinar com transcrições reais, ampliar o conjunto cego, rotular churn com desfecho real,
testar embeddings (ex.: BERTimbau) e otimizar limiares por custo de negócio.
"""),
]

nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
Path("notebooks").mkdir(exist_ok=True)
Path("notebooks/Sprint4_Colab.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok")
