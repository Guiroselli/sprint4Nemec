"""Etapas de Indicadores e Recomendação.

Os scores (0-100) combinam a probabilidade do modelo (70%) com sinais léxicos
interpretáveis (30%). Os sinais léxicos tornam o score robusto a frases que o
modelo não viu no treino e explicam *por quê* o score subiu.
"""
import re

from .processing import limpar

PESO_MODELO, PESO_REGRA = 0.7, 0.3

# nome do sinal -> (regex sobre texto normalizado, peso). A soma dos pesos é limitada a 1.
SINAIS_CHURN = {
    "cancelar": (r"cancel\w*", 0.6), "encerrar": (r"encerr\w*", 0.6),
    "portabilidade": (r"portabilidade", 0.6), "quer sair": (r"(vou|quero|vamos) sair|sair se", 0.6),
    "migrar/mudar de empresa": (r"(migrar|migracao|mudar para outr\w*)", 0.5),
    "concorrência": (r"(concorrenci\w*|concorrente\w*)", 0.4),
    "cotou/foi para outra empresa": (r"(cotei|fui) (em|na|no|para) outr\w*|outr[ao] (banco|empresa|instituicao|operadora|seguradora|lugar)", 0.4),
    "não pretende continuar": (r"nao pretendo continuar", 0.6),
    "reclamação recorrente": (r"(terceira|quarta|varias) vez\w*|ja liguei", 0.2),
}
NEGACAO_CHURN = re.compile(r"nao (quero|vou|pretendo) (cancelar|sair|encerrar)|prefiro continuar")

SINAIS_OPORT = {
    "interesse declarado": (r"(tenho interesse|queria conhecer|gostaria de conhecer|adorei)", 0.4),
    "contratar": (r"contrat\w+", 0.4), "cotação/proposta": (r"(cotacao|proposta)", 0.5),
    "aumentar limite/cobertura": (r"(aumentar|ampliar) (o )?(limite|cobertura)|incluir", 0.6),
    "quer investir": (r"(quero|queria|gostaria de) investir|investir mais|melhor rentabilidade", 0.6),
    "mais benefícios": (r"(mais beneficios|cashback|programa de pontos|previdencia|diversificar)", 0.4),
}

# léxico simples de sentimento (complementa o modelo em vocabulário não visto no treino)
LEX_NEG = r"(chateado|absurdo|irritad\w*|decepcionad\w*|abusiv\w*|pessim\w*|revoltad\w*|insatisfeit\w*|travand\w*|reclam\w*|cobrad\w* (indevid\w*|duas vezes)|nao (abre|funciona|resolve\w*)|mais cara|dificil|descaso|ninguem (resolve|avisou|me avis\w*))"
LEX_POS = r"(agradecer|otimo|otima|excelente|adorei|adoro|parabens|satisfeit\w*|feliz|contente|recomendo|rapid\w*|perfeito|gostei|tudo otimo|super)"


RECOMENDACOES = {
    "retencao_urgente": "Acionar o time de retenção em até 24h: ligar com oferta personalizada "
                        "(isenção de tarifa, revisão de taxa ou bônus) e tratar a causa-raiz da reclamação.",
    "retencao_preventiva": "Cliente com sinais de insatisfação: fazer follow-up em 7 dias, confirmar que o problema "
                           "foi resolvido e oferecer um benefício de fidelização.",
    "oportunidade": "Encaminhar o lead ao time comercial com a oferta indicada abaixo "
                    "e agendar contato em até 48h.",
    "oportunidade_com_risco": "Cliente tem interesse comercial mas também risco de saída: resolver primeiro a "
                              "insatisfação e só depois apresentar a oferta.",
    "acompanhar": "Sem sinal relevante de risco ou venda: registrar o atendimento e manter o relacionamento regular.",
    "elogio": "Cliente satisfeito: pedir indicação/avaliação (NPS) e registrar como caso de sucesso.",
}

OFERTA_POR_PRODUTO = {
    "Conta Digital": "pacote de serviços / cartão vinculado à conta",
    "Cartão de Crédito": "aumento de limite ou cartão com mais benefícios",
    "Empréstimo": "refinanciamento ou novo crédito com taxa menor",
    "Seguro": "seguro complementar (residencial/vida) ou ampliação de cobertura",
    "Investimentos": "produto de maior rentabilidade ou previdência",
}


LIMIAR_CHURN, LIMIAR_OPORT = 35, 50  # limiares de alerta (ver README)


def sinais(txt: str, dicionario: dict) -> list[tuple[str, float]]:
    t = limpar(txt)
    return [(nome, w) for nome, (rx, w) in dicionario.items() if re.search(rf"\b(?:{rx})\b", t)]


def nivel(score: float) -> str:
    return "Alto" if score >= 65 else "Médio" if score >= 35 else "Baixo"


def _combinar(p_modelo: float, achados: list[tuple[str, float]]) -> float:
    regra = min(1.0, sum(w for _, w in achados))
    return round(100 * (PESO_MODELO * p_modelo + PESO_REGRA * regra), 1)


def score_churn(p_modelo: float, txt: str) -> tuple[float, list[str]]:
    achados = sinais(txt, SINAIS_CHURN)
    if NEGACAO_CHURN.search(limpar(txt)):  # "não quero cancelar" não é intenção de saída
        achados = [(n, w) for n, w in achados if n not in ("cancelar", "encerrar", "quer sair")]
        return min(30.0, _combinar(p_modelo, achados)), [n for n, _ in achados]
    return _combinar(p_modelo, achados), [n for n, _ in achados]


def score_oportunidade(p_modelo: float, txt: str) -> tuple[float, list[str]]:
    achados = sinais(txt, SINAIS_OPORT)
    return _combinar(p_modelo, achados), [n for n, _ in achados]


def probas_lexico(txt: str) -> dict[str, float]:
    t = limpar(txt)
    neg = len(re.findall(rf"\b(?:{LEX_NEG})\b", t))
    pos = len(re.findall(rf"\b(?:{LEX_POS})\b", t))
    raw = {"negativo": neg, "positivo": pos, "neutro": 0.7}
    tot = sum(raw.values())
    return {k: v / tot for k, v in raw.items()}


def mistura_sentimento(p_modelo: dict[str, float], txt: str, peso_lexico: float = 0.4) -> dict[str, float]:
    lex = probas_lexico(txt)
    return {k: (1 - peso_lexico) * p_modelo.get(k, 0) + peso_lexico * lex[k] for k in lex}


def score_sentimento(probas: dict[str, float]) -> float:
    """-100 (muito negativo) a +100 (muito positivo)."""
    return round(100 * (probas.get("positivo", 0) - probas.get("negativo", 0)), 1)


def recomendar(produto: str, sentimento: str, s_churn: float, s_oport: float) -> tuple[str, str, str]:
    """Retorna (prioridade, ação, oferta sugerida)."""
    oferta = OFERTA_POR_PRODUTO.get(produto, "")
    if s_churn >= 65:
        return "Crítica", RECOMENDACOES["retencao_urgente"], ""
    if s_oport >= 65 and s_churn >= 35:
        return "Alta", RECOMENDACOES["oportunidade_com_risco"], oferta
    if s_oport >= 65:
        return "Alta", RECOMENDACOES["oportunidade"], oferta
    if s_churn >= 35 or sentimento == "negativo":
        return "Média", RECOMENDACOES["retencao_preventiva"], ""
    if s_oport >= 35:
        return "Média", RECOMENDACOES["oportunidade"], oferta
    if sentimento == "positivo":
        return "Baixa", RECOMENDACOES["elogio"], ""
    return "Baixa", RECOMENDACOES["acompanhar"], ""
