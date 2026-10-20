"""Protótipo funcional: cole uma transcrição e receba indicadores + recomendação.

Executar:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src import config
from src.pipeline import analisar

st.set_page_config(page_title="Análise de Transcrições", page_icon="📞", layout="wide")
st.title("📞 Análise automática de transcrições")
st.caption("Transcrição → Processamento → Modelo → Indicadores → Recomendação")

COR = {"Alto": "🔴", "Médio": "🟠", "Baixo": "🟢"}
PRIO = {"Crítica": "🔴", "Alta": "🟠", "Média": "🟡", "Baixa": "🟢"}


@st.cache_data
def exemplos() -> dict[str, str]:
    out = {}
    for arq in (config.NOVAS_CEGAS_CSV, config.NOVAS_MANUAIS_CSV):
        if arq.exists():
            df = pd.read_csv(arq)
            for _, r in df.iterrows():
                out[f"{arq.stem.replace('novas_', '')} #{r['id']} · {r['produto']} · {r['sentimento']}"] = r["transcricao"]
    return out


def mostrar(res: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Produto identificado", res["produto"], f"confiança {res['confianca_produto']:.0%}", delta_color="off")
    c2.metric("Sentimento", res["sentimento"].capitalize(), f"score {res['score_sentimento']:+.0f}", delta_color="off")
    c3.metric(f"{COR[res['nivel_churn']]} Risco de churn", f"{res['risco_churn']:.0f}/100", res["nivel_churn"], delta_color="off")
    c4.metric(f"{COR[res['nivel_oportunidade']]} Oportunidade", f"{res['oportunidade']:.0f}/100",
              res["nivel_oportunidade"], delta_color="off")

    st.subheader(f"{PRIO[res['prioridade']]} Prioridade {res['prioridade']}")
    st.info(res["recomendacao"] + (f"\n\n**Oferta sugerida:** {res['oferta_sugerida']}" if res["oferta_sugerida"] else ""))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Principais termos encontrados**")
        st.write(", ".join(res["principais_termos"]) or "—")
    with col_b:
        st.markdown("**Sinais que explicam os scores**")
        st.write(f"Churn: {', '.join(res['sinais_churn']) or 'nenhum'}")
        st.write(f"Oportunidade: {', '.join(res['sinais_oportunidade']) or 'nenhum'}")


aba1, aba2, aba3 = st.tabs(["Transcrição única", "Lote (CSV)", "Reuniões reais (TOTVS)"])

with aba1:
    ex = exemplos()
    escolha = st.selectbox("Carregar exemplo (opcional)", ["—"] + list(ex))
    texto = st.text_area("Transcrição", value=ex.get(escolha, ""), height=220,
                         placeholder="Atendente: ...\nCliente: ...")
    if st.button("Analisar", type="primary", disabled=not texto.strip()):
        mostrar(analisar(texto))

with aba2:
    arq = st.file_uploader("CSV com coluna `transcricao`", type="csv")
    if arq:
        df = pd.read_csv(arq)
        if "transcricao" not in df.columns:
            st.error("O CSV precisa ter a coluna `transcricao`.")
        else:
            res = pd.DataFrame([analisar(t) for t in df["transcricao"]])
            tab = pd.concat([df.drop(columns="transcricao"),
                             res[["produto", "sentimento", "risco_churn", "oportunidade", "prioridade", "recomendacao"]].add_prefix("prev_")
                             ], axis=1)
            st.dataframe(tab, use_container_width=True)
            st.download_button("Baixar resultado", tab.to_csv(index=False).encode("utf-8"), "analise.csv", "text/csv")

with aba3:
    from src.totvs_pipeline import REAL_JSON, analisar_reuniao, carregar_reunioes
    if not REAL_JSON.exists():
        st.warning("Importe os dados reais: python -m src.importar_totvs <dataset_limpo.json>")
    else:
        reunioes = {f"{r['id']} · cliente {r['segmento']} · {r['data']}": r for r in carregar_reunioes()}
        r = reunioes[st.selectbox("Reunião", list(reunioes))]
        res = analisar_reuniao(r["transcricao"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Produto", (res["produto"] or "sem sinal").replace("TOTVS ", ""), f"{res['confianca_produto']:.0%} dos termos", delta_color="off")
        c2.metric("Sentimento", res["sentimento"].capitalize(), f"score {res['score_sentimento']:+.0f}", delta_color="off")
        c3.metric(f"{COR[res['nivel_churn']]} Risco de churn", f"{res['risco_churn']:.0f}/100", res["nivel_churn"], delta_color="off")
        c4.metric(f"{COR[res['nivel_oportunidade']]} Oportunidade", f"{res['oportunidade']:.0f}/100", res["nivel_oportunidade"], delta_color="off")
        st.subheader(f"{PRIO[res['prioridade']]} Prioridade {res['prioridade']}")
        info = res["recomendacao"]
        if res["produto_recomendado"]:
            info += f"\n\n**Próximo passo ({res['produto_recomendado']['nome']}):** {res['produto_recomendado']['proximo_passo']}"
        st.info(info)
        st.write("**Principais termos:**", ", ".join(res["principais_termos"]))
        st.write("**Sinais de churn:**", "; ".join(res["sinais_churn"]) or "nenhum")
        st.write("**Dores detectadas:**", ", ".join(res["dores"]) or "nenhuma")
        with st.expander("Ver transcrição"):
            st.text(r["transcricao"][:6000])
