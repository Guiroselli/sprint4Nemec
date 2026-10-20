"""Roda o pipeline nas reuniões reais do Proton Flow e compara com a análise antiga.

Uso: python -m src.validate_totvs
Saídas: reports/analise_reunioes_reais.csv, reports/validacao_reunioes_reais.md, reports/revisao_humana.csv

IMPORTANTE: não existe gabarito humano para estas reuniões. A "análise antiga" (LLM local de 3B parâmetros +
regras) é só um ponto de comparação. Para medir acurácia de verdade, preencha reports/revisao_humana.csv.
"""
import pandas as pd

from . import config
from .totvs_pipeline import analisar_reuniao, carregar_reunioes

PRODUTO_LEGADO = {"fluig": "fluig", "protheus": "protheus", "crm": "crm", "rh": "rh",
                  "analytics": "analytics", "supply": "supply"}


def main():
    reunioes = carregar_reunioes()
    vistos, linhas = set(), []
    for r in reunioes:
        if r["hash"] in vistos:  # mesma transcrição repetida no dataset original
            continue
        vistos.add(r["hash"])
        a = analisar_reuniao(r["transcricao"], "")
        linhas.append({
            "id": r["id"], "data": r["data"], "cliente": r["segmento"], "palavras": a["n_palavras"],
            "produto": a["produto_key"], "parcela_produto": a["confianca_produto"],
            "produto_legado": r["legado"]["produto_top"],
            "sentimento": a["sentimento"], "risco_churn": a["risco_churn"], "nivel_churn": a["nivel_churn"],
            "oportunidade": a["oportunidade"], "urgencia": a["urgencia"], "urgencia_legado": r["legado"]["urgencia"],
            "prioridade": a["prioridade"], "dores": ";".join(a["dores"]),
            "termos": ", ".join(a["principais_termos"][:6]), "sinais_churn": "; ".join(a["sinais_churn"]),
            "sinais_oportunidade": "; ".join(a["sinais_oportunidade"]),
        })
    df = pd.DataFrame(linhas)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    df.to_csv(config.REPORTS_DIR / "analise_reunioes_reais.csv", index=False)

    # gabarito para revisão humana (preencher e reavaliar)
    rev = df[["id", "cliente", "produto", "sentimento", "nivel_churn", "oportunidade"]].copy()
    rev["produto_correto"] = rev["sentimento_correto"] = rev["churn_real (0/1)"] = rev["oportunidade_real (0/1)"] = ""
    rev.to_csv(config.REPORTS_DIR / "revisao_humana.csv", index=False)

    com_leg = df[df.produto_legado.notna() & df.produto.notna()]
    conc = (com_leg.produto == com_leg.produto_legado).mean() if len(com_leg) else float("nan")
    md = [
        "# Análise das reuniões reais (Proton Flow)", "",
        f"- Transcrições únicas analisadas: **{len(df)}** (de {len(reunioes)} registros; duplicatas removidas)",
        f"- Tamanho: mediana de **{int(df.palavras.median())}** palavras (mín. {df.palavras.min()}, máx. {df.palavras.max()})",
        f"- Sem produto identificado (poucos termos de domínio): **{df.produto.isna().sum()}**", "",
        "## Distribuição", "",
        "**Produto dominante**: " + ", ".join(f"{k}: {v}" for k, v in df.produto.value_counts().items()), "",
        "**Sentimento**: " + ", ".join(f"{k}: {v}" for k, v in df.sentimento.value_counts().items()), "",
        "**Risco de churn**: " + ", ".join(f"{k}: {v}" for k, v in df.nivel_churn.value_counts().items()), "",
        "**Prioridade**: " + ", ".join(f"{k}: {v}" for k, v in df.prioridade.value_counts().items()), "",
        "## Comparação com a análise antiga", "",
        f"Concordância no produto principal: **{conc:.0%}** em {len(com_leg)} reuniões com os dois resultados. "
        "A análise antiga indicou `fluig` em 8 das 12 reuniões que ela analisou com transcrição; ela é um ponto de "
        "comparação, não um gabarito, então a divergência não prova erro de nenhum dos dois lados.", "",
        "## Reuniões de maior risco", "",
        df.sort_values("risco_churn", ascending=False).head(5)[["id", "cliente", "produto", "risco_churn", "sinais_churn"]]
        .to_string(index=False), "",
        "## Como validar de verdade", "",
        "Preencha `reports/revisao_humana.csv` (produto correto, sentimento, churn e oportunidade reais) "
        "e compare com as colunas previstas. Conferência manual feita neste trabalho: as duas reuniões de maior risco "
        "(ids 1057714 e 1027294, ambas nível Médio) têm falas reais de saída (\"trocar de fornecedor\", \"decisão é encerrar o projeto\").",
    ]
    (config.REPORTS_DIR / "validacao_reunioes_reais.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
