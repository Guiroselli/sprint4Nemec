"""Pipeline completo: Transcrição -> Processamento -> Modelo -> Indicadores -> Recomendação."""
import json
from functools import lru_cache

import joblib
import numpy as np

from . import config
from .processing import termos_frequentes, texto_processado
from .scoring import (mistura_sentimento, nivel, recomendar, score_churn,
                      score_oportunidade, score_sentimento)


@lru_cache(maxsize=1)
def _carregar():
    if not config.MODELO_PATH.exists():
        raise FileNotFoundError("Modelos não treinados. Rode: python -m src.train")
    return joblib.load(config.MODELO_PATH)["modelos"]


def _proba(modelo, texto: str) -> dict:
    p = modelo.predict_proba([texto])[0]
    return {str(c): float(v) for c, v in zip(modelo.classes_, p)}


def termos_chave(txt: str, modelo_prod, n: int = 8) -> list[str]:
    """Termos com maior peso TF-IDF no documento (mistura frequência e raridade no corpus)."""
    tfidf = modelo_prod.named_steps["tfidf"]
    vet = tfidf.transform([texto_processado(txt)])
    vocab = np.array(tfidf.get_feature_names_out())
    idx = vet.toarray()[0].argsort()[::-1][:n]
    return [str(vocab[i]) for i in idx if vet[0, i] > 0]


def analisar(transcricao: str) -> dict:
    modelos = _carregar()
    texto = texto_processado(transcricao)

    p_prod = _proba(modelos["produto"], texto)
    produto = max(p_prod, key=p_prod.get)
    p_sent = mistura_sentimento(_proba(modelos["sentimento"], texto), transcricao)
    sentimento = max(p_sent, key=p_sent.get)
    p_churn = _proba(modelos["churn"], texto).get("1", 0.0)
    p_op = _proba(modelos["oportunidade"], texto).get("1", 0.0)

    s_churn, sin_churn = score_churn(p_churn, transcricao)
    s_op, sin_op = score_oportunidade(p_op, transcricao)
    prioridade, acao, oferta = recomendar(produto, sentimento, s_churn, s_op)

    return {
        "produto": produto, "confianca_produto": round(p_prod[produto], 3),
        "sentimento": sentimento, "score_sentimento": score_sentimento(p_sent),
        "risco_churn": s_churn, "nivel_churn": nivel(s_churn), "sinais_churn": sin_churn,
        "oportunidade": s_op, "nivel_oportunidade": nivel(s_op), "sinais_oportunidade": sin_op,
        "principais_termos": termos_chave(transcricao, modelos["produto"]),
        "termos_frequentes": [t for t, _ in termos_frequentes(transcricao, 5)],
        "prioridade": prioridade, "recomendacao": acao, "oferta_sugerida": oferta,
    }


def formatar(res: dict) -> str:
    return "\n".join([
        f"Produto identificado : {res['produto']} (confiança {res['confianca_produto']:.0%})",
        f"Sentimento           : {res['sentimento']} (score {res['score_sentimento']:+.0f})",
        f"Risco de churn       : {res['risco_churn']:.0f}/100 [{res['nivel_churn']}]"
        + (f"  sinais: {', '.join(res['sinais_churn'])}" if res["sinais_churn"] else ""),
        f"Oportunidade comercial: {res['oportunidade']:.0f}/100 [{res['nivel_oportunidade']}]"
        + (f"  sinais: {', '.join(res['sinais_oportunidade'])}" if res["sinais_oportunidade"] else ""),
        f"Principais termos    : {', '.join(res['principais_termos'])}",
        f"Prioridade           : {res['prioridade']}",
        f"Recomendação         : {res['recomendacao']}"
        + (f"\nOferta sugerida      : {res['oferta_sugerida']}" if res["oferta_sugerida"] else ""),
    ])


if __name__ == "__main__":
    import sys
    txt = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else sys.stdin.read()
    print(formatar(analisar(txt)))
