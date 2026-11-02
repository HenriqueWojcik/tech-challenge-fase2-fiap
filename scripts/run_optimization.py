"""Roda todos os experimentos do AG e gera os resultados em results/.

Uso:
    python scripts/run_optimization.py            # completo (~20-30 min)
    python scripts/run_optimization.py --quick    # versão rápida para testar (~1 min)
    python scripts/run_optimization.py --model logistic_regression
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tc2.experiments import run_all  # noqa: E402
from tc2.logging_utils import setup_logging  # noqa: E402
from tc2.models import MODEL_NAMES  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="população e gerações reduzidas")
    parser.add_argument("--model", choices=MODEL_NAMES, help="roda só um modelo")
    args = parser.parse_args()

    log_file = setup_logging("otimizacao")
    print(f"Log em: {log_file}")
    run_all(quick=args.quick, models=(args.model,) if args.model else MODEL_NAMES)
    print("Resultados em results/ (resumo_experimentos.csv, gráficos .png, modelos .joblib)")
