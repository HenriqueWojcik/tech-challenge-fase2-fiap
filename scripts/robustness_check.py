"""Checagem de robustez: original x otimizado em validação cruzada REPETIDA (10 x 5 folds).

Por quê: o teste tem só 42 casos malignos, então 1 falso negativo a mais ou a menos muda o recall
em 2,4 pontos. Para separar "melhora real" de "sorte do split", comparamos os dois conjuntos de
hiperparâmetros em 50 treinos/testes diferentes, com folds DIFERENTES dos usados pela fitness
(outra semente), sobre as 569 pacientes.

Ressalva: os hiperparâmetros foram escolhidos só com o treino, mas aqui o teste também entra nos
folds - é uma checagem complementar, não substitui o teste isolado.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate  # noqa: E402

from tc2.config import RESULTS_DIR  # noqa: E402
from tc2.data import load_data  # noqa: E402
from tc2.experiments import ORIGINAL_INDIVIDUAL  # noqa: E402
from tc2.ga.fitness import SCORERS  # noqa: E402
from tc2.models import build_pipeline  # noqa: E402


def main():
    data = load_data()
    res = json.loads((RESULTS_DIR / "resultados_completos.json").read_text(encoding="utf-8"))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=7)
    rows = []
    for model_name in res:
        best = res[model_name]["melhor"]
        for label, params in [("original_fase1", ORIGINAL_INDIVIDUAL[model_name]),
                              (f"otimizado_{best['experimento']}", best["params"])]:
            s = cross_validate(build_pipeline(model_name, params), data.X, data.y,
                               cv=cv, scoring=SCORERS, n_jobs=-1)
            rows.append({"modelo": model_name, "versao": label,
                         "f2_media": np.mean(s["test_f2"]), "f2_desvio": np.std(s["test_f2"]),
                         "recall_media": np.mean(s["test_recall"]),
                         "precision_media": np.mean(s["test_precision"]),
                         "accuracy_media": np.mean(s["test_accuracy"]),
                         # ~42 malignos por fold de 114 -> falsos negativos médios por fold
                         "fn_medio_por_fold": np.mean((1 - s["test_recall"]) * 42.4)})
            print(rows[-1]["modelo"], label, {k: round(v, 4) for k, v in rows[-1].items()
                                              if isinstance(v, float)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "robustez_cv_repetida.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
