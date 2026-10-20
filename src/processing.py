"""Etapa de Processamento: limpeza, normalização e extração de termos."""
import re
import unicodedata
from collections import Counter

STOPWORDS = set("""
a o as os um uma uns umas de do da dos das em no na nos nas por para com sobre ao aos
e ou mas que se como mais menos muito muita muitos muitas ja sim eu voce vocês voces me te
meu minha meus minhas seu sua seus suas ele ela eles elas nos lhe isso isto esse essa esses essas
este esta estes estas aquele aquela ser estar estou esta estamos sao foi era tem tenho ter tinha
vou vai vamos pode posso podem poder quero queria gostaria preciso favor bom dia boa tarde noite
ola oi tudo bem obrigado obrigada ai la aqui so apenas tambem entao pois porque quando onde qual
quais algum alguma coisa vez vezes hoje agora momento instante claro certo perfeito atendente
cliente ligando contato ajudar ajuda pra pro numa num dele dela deles delas ate sido ainda
mesmo mesma depois antes durante essa isso fazer faz feito
""".split())


def remover_acentos(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn")


def remover_falantes(txt: str) -> str:
    """Remove os rótulos 'Atendente:' / 'Cliente:' do início das linhas."""
    return re.sub(r"(?im)^\s*(atendente|cliente|agente|consultor)\s*:\s*", "", txt)


def limpar(txt: str) -> str:
    """Minúsculas, sem rótulos de falante, sem acentos, sem pontuação/dígitos soltos."""
    txt = remover_falantes(str(txt)).lower()
    txt = remover_acentos(txt)
    txt = re.sub(r"https?://\S+|\S+@\S+", " ", txt)
    txt = re.sub(r"\d+", " ", txt)
    txt = re.sub(r"[^a-z\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def tokens(txt: str) -> list[str]:
    return [t for t in limpar(txt).split() if t not in STOPWORDS and len(t) > 2]


def texto_processado(txt: str) -> str:
    """Texto limpo, sem stopwords, usado como entrada dos modelos (TF-IDF)."""
    return " ".join(tokens(txt))


def termos_frequentes(txt: str, n: int = 8) -> list[tuple[str, int]]:
    return Counter(tokens(txt)).most_common(n)
