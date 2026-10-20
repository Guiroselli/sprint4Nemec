"""Etapa de Modelo: compara candidatos por tarefa, seleciona o melhor e salva.

Uso:  python -m src.train
Saídas: models/modelos.joblib, reports/comparacao_modelos.csv, reports/modelo_selecionado.md
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import config
from .processing import texto_processado

# ordem = preferência em caso de empate (do mais simples/interpretável ao mais complexo)
CANDIDATOS = {
    "LogisticRegression": lambda: LogisticRegression(C=5, max_iter=2000, class_weight="balanced"),
    "NaiveBayes": lambda: MultinomialNB(alpha=0.3),
    "LinearSVC (calibrado)": lambda: CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, random_state=config.SEED,
                                                   class_weight="balanced", n_jobs=-1),
}


def montar(nome: str) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", CANDIDATOS[nome]()),
    ])


def carregar_treino() -> pd.DataFrame:
    if not config.TREINO_CSV.exists():
        raise SystemExit("Dados de treino não encontrados. Rode antes: python -m src.data_gen")
    df = pd.read_csv(config.TREINO_CSV)
    df["texto"] = df["transcricao"].map(texto_processado)
    return df


def main():
    df = carregar_treino()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    linhas, escolhidos = [], {}

    for tarefa, col in config.TAREFAS.items():
        y = df[col]
        X_tr, X_te, y_tr, y_te = train_test_split(
            df["texto"], y, test_size=0.2, stratify=y, random_state=config.SEED)
        melhor = None
        for nome in CANDIDATOS:
            f1_cv = cross_val_score(montar(nome), X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=1)
            m = montar(nome).fit(X_tr, y_tr)
            pred = m.predict(X_te)
            linha = {"tarefa": tarefa, "modelo": nome,
                     "f1_macro_cv": round(f1_cv.mean(), 4), "desvio_cv": round(f1_cv.std(), 4),
                     "f1_macro_holdout": round(f1_score(y_te, pred, average="macro"), 4),
                     "acuracia_holdout": round(accuracy_score(y_te, pred), 4)}
            linhas.append(linha)
            # seleção: maior F1 CV; empate (diferença < 0,005) -> modelo mais simples (vem antes)
            if melhor is None or linha["f1_macro_cv"] > melhor["f1_macro_cv"] + 0.005:
                melhor = linha
        escolhidos[tarefa] = melhor
        print(f"[{tarefa}] melhor: {melhor['modelo']} (F1 CV={melhor['f1_macro_cv']})")

    # reajusta os escolhidos com todos os dados de treino
    modelos = {t: montar(e["modelo"]).fit(df["texto"], df[config.TAREFAS[t]])
               for t, e in escolhidos.items()}

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump({"modelos": modelos, "escolhidos": escolhidos}, config.MODELO_PATH)
    comp = pd.DataFrame(linhas)
    comp.to_csv(config.REPORTS_DIR / "comparacao_modelos.csv", index=False)
    escrever_justificativa(comp, escolhidos)
    print("Modelos salvos em", config.MODELO_PATH)


def escrever_justificativa(comp: pd.DataFrame, escolhidos: dict):
    md = ["# Modelo final selecionado e justificativa", "",
          "Para cada tarefa (produto, sentimento, risco de churn, oportunidade comercial) foram "
          "comparados 4 classificadores sobre a mesma representação TF-IDF (unigramas+bigramas). "
          "Critério: **F1 macro em validação cruzada estratificada (5 folds)**, com conferência em "
          "hold-out de 20%. Em empate técnico (< 0,005) prevalece o modelo mais simples e interpretável.", ""]
    for tarefa, e in escolhidos.items():
        md += [f"## {tarefa.capitalize()}: {e['modelo']}", "",
               f"- F1 macro CV: **{e['f1_macro_cv']}** (±{e['desvio_cv']}) · hold-out F1: "
               f"**{e['f1_macro_holdout']}** · acurácia hold-out: **{e['acuracia_holdout']}**", "",
               _md_tabela(comp[comp.tarefa == tarefa].drop(columns="tarefa")), ""]
    md += ["## Por que esta abordagem", "",
           "- **Poucos dados e texto curto**: modelos lineares sobre TF-IDF costumam igualar ou superar "
           "modelos mais complexos, treinam em segundos e dispensam GPU.",
           "- **Interpretabilidade**: os coeficientes mostram quais termos puxam a decisão, "
           "o que ajuda a área comercial a confiar na recomendação.",
           "- **Probabilidades calibradas** alimentam diretamente o sistema de scoring (0-100).",
           "- **Cautela**: os dados atuais são sintéticos; os números absolutos são otimistas e devem "
           "ser reavaliados com as transcrições reais (ver `README.md`)."]
    (config.REPORTS_DIR / "modelo_selecionado.md").write_text("\n".join(md), encoding="utf-8")


def _md_tabela(df: pd.DataFrame) -> str:
    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    corpo = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([cab, sep, *corpo])


if __name__ == "__main__":
    main()
