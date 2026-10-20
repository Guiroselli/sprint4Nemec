"""Pipeline para as reuniões reais do Proton Flow (transcrições longas, anonimizadas).

Transcrição -> Processamento -> Modelo (catálogo TOTVS + léxicos) -> Indicadores -> Recomendação

Diferenças para o pipeline sintético (src/pipeline.py):
- as reuniões têm de 7 mil a 120 mil caracteres e não há rótulos humanos; por isso o "modelo" de produto
  é o catálogo TOTVS (palavras fortes/genéricas por produto) e os indicadores usam sinais léxicos explicáveis;
- os termos principais vêm de TF-IDF ajustado no corpus real.
"""
import json
import math
import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .processing import STOPWORDS, limpar, remover_acentos
from .scoring import RECOMENDACOES, nivel
from .totvs_catalog import CATALOGO_TOTVS, compute_recommendations

REAL_JSON = config.DATA_DIR / "real" / "reunioes_totvs.json"

# ruído de transcrição oral e marcadores de anonimização
STOP_ORAL = set("""
locutor pessoa empresa local ta ne ai pra gente cara tipo assim entao ficar fica aqui ali la isso entendi beleza
sim nao ok certo ne so vou vai vamos ver vamos acho acha pode podemos tem tinha temos ter seria sendo todo toda
tudo nada alguma algum voces voce nos eles agora ainda depois antes ja entao tambem muito pouco mais menos
coisa coisas bom dia tarde noite obrigado obrigada ola
""".split())

# vocabulário de domínio que complementa as palavras do catálogo (acentos removidos)
EXTRAS_PRODUTO = {
    "fluig": ["fluig", "workflow", "aprovacao", "assinatura", "fluxo de aprovacao", "ecm"],
    "crm": ["crm", "oportunidade", "proposta", "renovacao", "follow up", "pos venda", "pipeline"],
    "protheus": ["protheus", "fiscal", "tributacao", "tributacao", "nota fiscal", "contabil", "faturamento",
                 "financeiro", "contas a pagar", "contas a receber", "fechamento", "nfe", "cte", "erp"],
    "analytics": ["dashboard", "kpi", "indicadores", "bi", "analytics", "relatorio gerencial"],
    "rh": ["folha", "ponto", "ferias", "atestado", "esocial", "admissao", "meu rh", "rm", "beneficios", "escala"],
    "supply": ["estoque", "armazem", "wms", "logistica", "expedicao", "frete", "inventario"],
}

REGRAS_DORES = {  # categoria -> (rótulo, regex sobre texto sem acentos)
    "aprovacao_pendente": ("Aprovação pendente", r"aguardando aprovacao|aprovacao pendente|falta aprovacao|depende (de|da) aprovacao|alcada"),
    "atraso_prazo": ("Atraso / prazo em risco", r"atras(o|ad[oa]s?)|fora do prazo|prazo (estourou|vencid\w*|curto)|nao (entregou|entregaram)|postergar"),
    "gargalo_operacional": ("Gargalo operacional", r"gargalo|retrabalho|planilha|travad[oa]|muito lent[oa]|nao consegu\w+|manualmente|erro de"),
    "insatisfacao_cliente": ("Insatisfação do cliente", r"insatisf\w+|reclam\w+|frustrad\w+|decepcion\w+|pessim[oa]|nao (estou|estamos) satisfeit\w+"),
    "documento_faltando": ("Documento não localizado", r"documento (faltando|pendente|nao localizado)|falta (o )?documento|extravi\w+"),
    "falta_metricas": ("Falta de métricas/visão", r"sem (indicador\w*|visibilidade|relatorio\w*)|nao temos (indicador\w*|metrica\w*|visao)"),
    "problemas_equipe": ("Sobrecarga/falta de pessoal", r"sobrecarga|falta de (gente|pessoal|equipe)|equipe reduzida|turnover|afastament\w+"),
    "gestao_estoque": ("Estoque/logística", r"ruptura|estoque (zerado|baixo)|inventario|divergencia de estoque"),
}

SINAIS_RISCO = {  # sinal -> (regex, peso). Exigem contexto: "cancelamento de nota" é rotina fiscal, não churn.
    "cancelamento/rescisão": (r"(cancel\w+|rescin\w+|encerrar) (o |a |do |da |de )?(contrato|servico|licenca|assinatura|projeto|parceria)|rescis\w+ contratual|(vou|vamos|quero|queremos) cancelar tudo", 0.5),
    "troca de fornecedor": (r"trocar de (sistema|fornecedor|empresa)|migrar para (outr\w+|a concorrencia)|mudar de (sistema|fornecedor)|trocar (o|a) (totvs|sistema)", 0.5),
    "concorrência": (r"concorrente\w*|(para|na|da|a) concorrencia(?! de)", 0.3),
    "insatisfação": (r"insatisf\w+|frustrad\w+|decepcion\w+|pessim\w+|nao (estou|estamos) satisfeit\w+|reclamacao", 0.35),
    "perda de cliente": (r"perder (o|a) (cliente|conta)|risco de perder", 0.5),
    "projeto parado/atrasado": (r"projeto (parado|atrasad\w+)|prazo (estourou|vencid\w+)|fora do prazo|atraso (no|na|do|da) (projeto|entrega|implantacao)", 0.2),
    "problema crítico": (r"producao parada|sistema (fora|parado|fora do ar)|paralisac\w+", 0.3),
    "multa/jurídico": (r"multa (contratual|de rescisao|por rescisao)|notificacao extrajudicial|processo judicial", 0.25),
}
NEGACAO = re.compile(r"nao (quero|vou|pretendo|vamos) (cancelar|trocar|sair|encerrar)|sem (risco|problema) de")

SINAIS_OPORT = {  # "implantação" fica de fora: é projeto em andamento, não venda nova
    "pedido/interesse": (r"temos interesse|tenho interesse|gostar\w+ de (conhecer|contratar|ver)|queremos (contratar|ampliar|adquirir)", 0.4),
    "orçamento/proposta": (r"orcamento|proposta comercial|nova proposta|cotacao", 0.35),
    "expansão/novo módulo": (r"novo modulo|novos modulos|ampliar o (escopo|contrato)|expansao|proxima fase|segunda fase|nova filial|novas unidades", 0.35),
    "treinamento/consultoria": (r"pacote de horas|horas de consultoria|contratar (treinamento|consultoria)|treinamento adicional", 0.25),
}

LEX_POS = r"(obrigad[oa]|otimo|otima|excelente|perfeito|legal|show|parabens|satisfeit\w+|bacana|funcionou|resolveu|resolvido|gostei|tranquilo|feliz)"
LEX_NEG = r"(dificuldade\w*|reclam\w+|atras\w+|preocupad\w+|complicad\w+|travou|parou|falha\w*|insatisf\w+|frustrad\w+|pessim\w+|nao funciona|nao consegu\w+)"

COMPROMISSO = re.compile(
    r"(vou|vamos|fica combinado|ficou de|fico de|preciso|precisamos|vou te|vou enviar|pode enviar|me envia|enviar|"
    r"agendar|marcar|alinhar|verificar|validar|subir|abrir (um )?chamado)[^.?!\n]{8,140}")

CRITICAS = ["producao parada", "risco de perder o cliente", "paralisacao", "sistema fora"]
ALTAS = ["bloqueado", "bloqueada", "bloqueio", "prazo vencido", "fora do prazo", "atraso critico",
         "cliente insatisfeito", "reclamacao grave", "critico"]


# ----------------------------- processamento -----------------------------
def preparar(txt: str) -> str:
    """Remove rótulos de locutor e marcadores de anonimização, preservando as frases."""
    txt = re.sub(r"(?m)^\s*\[LOCUTOR \d+\]:\s*", "", txt)
    txt = re.sub(r"\[LOCUTOR \d+\]:?", " ", txt)
    txt = re.sub(r"\[(PESSOA|EMPRESA|LOCAL|[A-ZÇÃ_ ]{3,20})\]", " ", txt)
    return re.sub(r"[ \t]+", " ", txt).strip()


def normalizar(txt: str) -> str:
    return re.sub(r"\s+", " ", remover_acentos(preparar(txt).lower()))


def _frases(txt: str) -> list[str]:
    return [f.strip() for f in re.split(r"(?<=[.?!])\s+|\n", preparar(txt)) if len(f.strip()) > 25]


def _tokens(txt: str) -> list[str]:
    return [t for t in limpar(preparar(txt)).split()
            if t not in STOPWORDS and t not in STOP_ORAL and len(t) > 2]


# ----------------------------- corpus / termos -----------------------------
@lru_cache(maxsize=1)
def _tfidf():
    reunioes = carregar_reunioes()
    docs = list({r["hash"]: " ".join(_tokens(r["transcricao"])) for r in reunioes}.values())
    vet = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.7, sublinear_tf=True)
    vet.fit(docs)
    return vet


def carregar_reunioes() -> list[dict]:
    if not REAL_JSON.exists():
        raise SystemExit("Importe os dados reais: python -m src.importar_totvs <dataset_limpo.json>")
    return json.load(open(REAL_JSON, encoding="utf-8"))


def termos_chave(txt: str, n: int = 10) -> list[str]:
    vet = _tfidf()
    m = vet.transform([" ".join(_tokens(txt))])
    vocab = vet.get_feature_names_out()
    idx = m.toarray()[0].argsort()[::-1][:n]
    return [str(vocab[i]) for i in idx if m[0, idx[0]] > 0 and m[0, i] > 0]


# ----------------------------- produto -----------------------------
def _contar(rx: str, t: str) -> int:
    return len(re.findall(rf"\b(?:{rx})\b", t))


def pontuar_produtos(t: str) -> dict[str, dict]:
    """Pontua cada produto do catálogo pela densidade de termos (por 1.000 palavras)."""
    n_pal = max(1, len(t.split()))
    out = {}
    for key, prod in CATALOGO_TOTVS.items():
        fortes = {remover_acentos(p.lower()) for p in prod["palavras_fortes"]} | set(EXTRAS_PRODUTO[key])
        hits = {p: _contar(re.escape(p), t) for p in fortes}
        hits = {p: c for p, c in hits.items() if c}
        bruto = sum(1 + math.log(c) for c in hits.values())  # amortece termo repetido
        out[key] = {"pontos": bruto, "densidade": round(1000 * bruto / n_pal, 3),
                    "termos": sorted(hits, key=hits.get, reverse=True)[:6]}
    return out


# ----------------------------- dores / tarefas -----------------------------
def detectar_dores(txt: str) -> list[dict]:
    dores = []
    for cat, (label, rx) in REGRAS_DORES.items():
        achados = [f for f in _frases(txt) if re.search(rx, normalizar(f))]
        for f in achados[:3]:
            dores.append({"categoria": cat, "label": label, "trecho": f[:220]})
        if achados:
            dores[-1]["total"] = len(achados)
    return dores


def extrair_tarefas(txt: str, n: int = 6) -> list[dict]:
    tarefas = []
    for f in _frases(txt):
        m = COMPROMISSO.search(f)
        if m and len(tarefas) < n:
            tarefas.append({"tarefa": m.group(0).strip(), "responsavel": "Não identificado", "prazo": "Não mencionado"})
    return tarefas


# ----------------------------- indicadores -----------------------------
def _score_sinais(t: str, sinais: dict, neg=None) -> tuple[float, list[str]]:
    total, achados = 0.0, []
    for nome, (rx, w) in sinais.items():
        c = _contar(rx, t)
        if c:
            total += w * (1 - 0.5 ** c)  # retorno decrescente: 1 menção ≈ w/2; 3+ ≈ w
            achados.append(f"{nome} (x{c})")
    if neg and neg.search(t):
        total *= 0.5
    return round(100 * min(1.0, total), 1), achados


def sentimento(t: str) -> tuple[str, float]:
    pos, neg = _contar(LEX_POS, t), _contar(LEX_NEG, t)
    score = round(100 * (pos - neg) / (pos + neg + 10), 1)  # +10 amortece textos longos e neutros
    return ("positivo" if score >= 15 else "negativo" if score <= -15 else "neutro"), score


def urgencia(t: str) -> str:
    if any(c in t for c in CRITICAS):
        return "Crítica"
    return "Alta" if any(a in t for a in ALTAS) else "Média" if "urgente" in t or "prioridade" in t else "Baixa"


# ----------------------------- análise -----------------------------
def analisar_reuniao(transcricao: str, segmento: str = "") -> dict:
    t = normalizar(transcricao)
    n_pal = len(t.split())

    pontos = pontuar_produtos(t)
    total = sum(p["pontos"] for p in pontos.values())
    ranking = sorted(pontos, key=lambda k: pontos[k]["pontos"], reverse=True)
    top = ranking[0]
    share = pontos[top]["pontos"] / total if total else 0.0
    sem_sinal = total < 3  # menos de ~3 termos distintos: não afirma produto

    dores = detectar_dores(transcricao)
    tarefas = extrair_tarefas(transcricao)
    urg = urgencia(t)
    tema = ", ".join(pontos[top]["termos"][:3])
    recs = compute_recommendations(dores, tarefas, tema=tema, urgencia=urg, segmento=segmento or "",
                                   full_transcript=transcricao)
    fit = {r["product_key"]: r for r in recs}

    sent, s_sent = sentimento(t)
    s_risco, sin_risco = _score_sinais(t, SINAIS_RISCO, NEGACAO)
    s_op_bruto, sin_op = _score_sinais(t, SINAIS_OPORT)
    fit_top = fit.get(top, {}).get("fit_score", 0.0)
    # oportunidade: interesse declarado + aderência do produto dominante ao que foi discutido
    s_op = round(0.8 * s_op_bruto + 0.2 * 100 * (0 if sem_sinal else share), 1)

    prod = CATALOGO_TOTVS[top]
    if s_risco >= 65:
        prioridade, acao = "Crítica", RECOMENDACOES["retencao_urgente"]
    elif s_op >= 50 and s_risco >= 35:
        prioridade, acao = "Alta", RECOMENDACOES["oportunidade_com_risco"]
    elif s_op >= 50:
        prioridade, acao = "Alta", RECOMENDACOES["oportunidade"]
    elif s_risco >= 35 or sent == "negativo":
        prioridade, acao = "Média", RECOMENDACOES["retencao_preventiva"]
    elif s_op >= 35:
        prioridade, acao = "Média", RECOMENDACOES["oportunidade"]
    else:
        prioridade, acao = "Baixa", RECOMENDACOES["acompanhar"]

    return {
        "produto": None if sem_sinal else prod["product_name"], "produto_key": None if sem_sinal else top,
        "confianca_produto": round(share, 3), "ranking_produtos": [(k, round(pontos[k]["pontos"], 1)) for k in ranking],
        "termos_do_produto": pontos[top]["termos"],
        "sentimento": sent, "score_sentimento": s_sent,
        "risco_churn": s_risco, "nivel_churn": nivel(s_risco), "sinais_churn": sin_risco,
        "oportunidade": s_op, "nivel_oportunidade": nivel(s_op), "sinais_oportunidade": sin_op,
        "urgencia": urg,
        "dores": sorted({d["categoria"] for d in dores}),
        "tarefas_sugeridas": [t_["tarefa"] for t_ in tarefas],
        "principais_termos": termos_chave(transcricao),
        "n_palavras": n_pal, "prioridade": prioridade, "recomendacao": acao,
        "produto_recomendado": None if sem_sinal else {
            "nome": prod["product_name"], "fit_score": fit_top,
            "por_que": fit.get(top, {}).get("why_recommended", f"Termos do módulo em destaque: {', '.join(pontos[top]['termos'][:3])}."),
            "proximo_passo": prod["proximo_passo_implantacao"], "beneficios": prod["beneficios_esperados"][:2]},
    }


def formatar(r: dict) -> str:
    linhas = [
        f"Produto identificado  : {r['produto'] or 'sem sinal suficiente'} (parcela {r['confianca_produto']:.0%}; termos: {', '.join(r['termos_do_produto'][:4])})",
        f"Sentimento            : {r['sentimento']} (score {r['score_sentimento']:+.0f})",
        f"Risco de churn        : {r['risco_churn']:.0f}/100 [{r['nivel_churn']}] {', '.join(r['sinais_churn'])}",
        f"Oportunidade comercial: {r['oportunidade']:.0f}/100 [{r['nivel_oportunidade']}] {', '.join(r['sinais_oportunidade'])}",
        f"Urgência              : {r['urgencia']} · dores: {', '.join(r['dores']) or '—'}",
        f"Principais termos     : {', '.join(r['principais_termos'])}",
        f"Prioridade            : {r['prioridade']}",
        f"Recomendação          : {r['recomendacao']}",
    ]
    if r["produto_recomendado"]:
        linhas.append(f"Próximo passo         : {r['produto_recomendado']['proximo_passo']}")
    return "\n".join(linhas)


if __name__ == "__main__":
    import sys
    txt = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else sys.stdin.read()
    print(formatar(analisar_reuniao(txt)))
