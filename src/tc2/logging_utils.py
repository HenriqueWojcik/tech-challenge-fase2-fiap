"""Logging e monitoramento (requisito 2 do enunciado: "monitoramento e logging para tracking").

Tudo vai para dois lugares:
- console (acompanhar ao vivo);
- logs/<nome>_<data>.log (histórico permanente, útil para auditoria e para o relatório).
As métricas numéricas de cada geração também são salvas em results/*.json e *.csv pelo experiments.py.
"""
import logging
import sys
from datetime import datetime

from .config import LOGS_DIR


def setup_logging(name: str = "run", level: int = logging.INFO) -> str:
    log_file = LOGS_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    root = logging.getLogger("tc2")
    root.setLevel(level)
    root.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    return str(log_file)
