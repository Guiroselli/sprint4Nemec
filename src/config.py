"""Configuração central do projeto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TREINO_CSV = DATA_DIR / "transcricoes_treino.csv"
NOVAS_SINTETICAS_CSV = DATA_DIR / "novas_sinteticas.csv"
NOVAS_MANUAIS_CSV = DATA_DIR / "novas_manuais.csv"
NOVAS_CEGAS_CSV = DATA_DIR / "novas_cegas.csv"
MODELO_PATH = MODELS_DIR / "modelos.joblib"

SEED = 42

PRODUTOS = ["Conta Digital", "Cartão de Crédito", "Empréstimo", "Seguro", "Investimentos"]
SENTIMENTOS = ["negativo", "neutro", "positivo"]

# alvos supervisionados -> coluna do CSV
TAREFAS = {
    "produto": "produto",
    "sentimento": "sentimento",
    "churn": "churn",
    "oportunidade": "oportunidade",
}
