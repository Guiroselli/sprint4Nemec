"""Validação do pipeline completo em transcrições novas (nunca vistas no treino).

Uso: python -m src.validate
Saídas: reports/validacao.md, reports/validacao_detalhe.csv
"""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from . import config
from .pipeline import analisar
from .scoring import LIMIAR_CHURN, LIMIAR_OPORT

# como converter a saída do pipeline em classe prevista
PRED = {
    "produto": lambda r: r["produto"],
    "sentimento": lambda r: r["sentimento"],
    "churn": lambda r: int(r["risco_churn"] >= LIMIAR_CHURN),
    "oportunidade": lambda r: int(r["oportunidade"] >= LIMIAR_OPORT),
}


def avaliar(df: pd.DataFrame, nome: str):
    res = [analisar(t) for t in df["transcricao"]]
    linhas = {}
    for tarefa, col in config.TAREFAS.items():
        pred = [PRED[tarefa](r) for r in res]
        linhas[tarefa] = {
            "acuracia": round(accuracy_score(df[col], pred), 3),
            "f1_macro": round(f1_score(df[col], pred, average="macro", zero_division=0), 3)}
    detalhe = df.copy()
    for tarefa in config.TAREFAS:
        detalhe[f"pred_{tarefa}"] = [PRED[tarefa](r) for r in res]
    detalhe["risco_churn"] = [r["risco_churn"] for r in res]
    detalhe["score_oportunidade"] = [r["oportunidade"] for r in res]
    detalhe["prioridade"] = [r["prioridade"] for r in res]
    return linhas, detalhe


def main():
    partes = ["# Validação com novas transcrições", ""]
    for nome, path, aviso in [
        ("Manuais (escritas à mão)", config.NOVAS_MANUAIS_CSV,
         "As regras do scoring foram calibradas olhando estes casos: resultado NÃO é independente."),
        ("Cegas (escritas à mão, nunca usadas para ajuste)", config.NOVAS_CEGAS_CSV,
         "Melhor estimativa de generalização disponível; amostra pequena, então a variância é alta."),
        ("Sintéticas (seed diferente do treino)", config.NOVAS_SINTETICAS_CSV,
         "Mesma distribuição do treino, portanto resultado otimista; serve como teste de regressão."),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        linhas, detalhe = avaliar(df, nome)
        slug = path.stem.replace("novas_", "")
        detalhe.to_csv(config.REPORTS_DIR / f"validacao_detalhe_{slug}.csv", index=False)
        partes += [f"## {nome}", "", f"_{aviso}_  (n = {len(df)})", "",
                   "| Tarefa | Acurácia | F1 macro |", "|---|---|---|"]
        partes += [f"| {t} | {m['acuracia']} | {m['f1_macro']} |" for t, m in linhas.items()]
        erros = detalhe[(detalhe.produto != detalhe.pred_produto) | (detalhe.sentimento != detalhe.pred_sentimento)
                        | (detalhe.churn != detalhe.pred_churn) | (detalhe.oportunidade != detalhe.pred_oportunidade)]
        partes += ["", f"Transcrições com ao menos um erro: **{len(erros)} de {len(df)}**.", ""]
        print(f"\n== {nome} ==")
        for t, m in linhas.items():
            print(f"  {t:13s} acc={m['acuracia']}  f1={m['f1_macro']}")
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "validacao.md").write_text("\n".join(partes), encoding="utf-8")


if __name__ == "__main__":
    main()
