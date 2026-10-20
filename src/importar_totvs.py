"""Importa as reuniões reais do projeto Proton Flow (TOTUVsVers-oFinal).

Uso: python -m src.importar_totvs caminho/para/backend/dataset_limpo.json
Gera data/real/reunioes_totvs.json com as transcrições anonimizadas + a análise antiga (só para comparação).
"""
import hashlib
import json
import sys

from . import config

DESTINO = config.DATA_DIR / "real" / "reunioes_totvs.json"
MIN_CHARS = 300  # abaixo disso a reunião não tem transcrição útil


def converter(e: dict) -> dict:
    rec = (e.get("recomendacoes_totvs") or [])
    ia = e.get("RESUMO_IA") or {}
    return {
        "id": str(e["ID_MEETING"]),
        "data": str(e.get("DT_MEETING", ""))[:10],
        "segmento": e.get("NOME_SEGMENTO"),
        "duracao": e.get("DURACAO_MEETING"),
        "transcricao": e["ANON_TRANSCRICAO"],
        "hash": hashlib.md5(e["ANON_TRANSCRICAO"].encode("utf-8")).hexdigest()[:10],
        # análise antiga (LLM local + regras): usada apenas para comparação, NÃO é verdade absoluta
        "legado": {
            "tema": ia.get("tema"),
            "urgencia": e.get("NIVEL_URGENCIA"),
            "produto_top": rec[0]["product_key"] if rec else None,
            "dores": sorted({d.get("categoria") for d in ia.get("dores", []) if isinstance(d, dict)}),
        },
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    bruto = json.load(open(sys.argv[1], encoding="utf-8"))
    reunioes = [converter(e) for e in bruto if len(e.get("ANON_TRANSCRICAO") or "") >= MIN_CHARS]
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    json.dump(reunioes, open(DESTINO, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    unicas = len({r["hash"] for r in reunioes})
    print(f"{len(reunioes)} reuniões com transcrição ({unicas} textos únicos) -> {DESTINO}")


if __name__ == "__main__":
    main()
