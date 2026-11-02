"""Configurações globais do projeto (caminhos e sementes)."""
from pathlib import Path

# Mesma semente da Fase 1 -> mesmo split treino/teste, comparação justa.
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Número de folds da validação cruzada usada na função fitness.
CV_FOLDS = 5

# No dataset do sklearn: 0 = maligno, 1 = benigno.
# A classe que mais importa clinicamente é a MALIGNA (evitar falso negativo).
MALIGNANT = 0
BENIGN = 1
CLASS_NAMES = {MALIGNANT: "maligno", BENIGN: "benigno"}

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
