"""FUNÇÃO DE APTIDÃO (fitness) - AG Aula 3; cache inspirado na tabela de distâncias da AG Aula 4.

No TSP, fitness = distância total da rota (menor é melhor).
Aqui, fitness = quão bem o modelo detecta câncer maligno (maior é melhor).

Escolha da métrica: F2-score da classe maligna.
- Recall sozinho seria "trapaceável": um modelo que chama TODO MUNDO de maligno tem recall 1.0.
- F1 dá o mesmo peso a precisão e recall.
- F2 (beta=2) dá ao recall 2x mais importância que à precisão -> falso negativo (câncer não
  detectado) custa mais que falso positivo (exame extra), mas sem ignorar a precisão.

Por que validação cruzada e não o conjunto de teste?
- Se o AG otimizasse no teste, o teste deixaria de ser uma avaliação independente (data leakage).
- A Fase 1 apontou como limitação "um único split de teste". A média em 5 folds é mais estável.
"""
import time
from typing import Any

import numpy as np
from sklearn.metrics import make_scorer, fbeta_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from ..config import CV_FOLDS, MALIGNANT, RANDOM_STATE
from ..models import build_pipeline
from .space import Individual, individual_key

SCORERS = {
    "f2": make_scorer(fbeta_score, beta=2, pos_label=MALIGNANT),
    "recall": make_scorer(recall_score, pos_label=MALIGNANT),
    "precision": make_scorer(precision_score, pos_label=MALIGNANT, zero_division=0),
    "accuracy": "accuracy",
}


class CVFitness:
    """Avalia um indivíduo treinando o modelo com validação cruzada estratificada.

    Mantém um cache: o mesmo conjunto de hiperparâmetros nunca é treinado duas vezes
    (indivíduos repetidos aparecem com frequência por causa do elitismo e do crossover).
    É a mesma troca da AG Aula 4: gastar memória para economizar processamento.
    """

    def __init__(self, model_name: str, X, y, folds: int = CV_FOLDS, n_jobs: int = -1):
        self.model_name = model_name
        self.X, self.y = X, y
        self.cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        self.n_jobs = n_jobs  # -1 = paraleliza os folds em todos os núcleos (escalabilidade)
        self.cache: dict[tuple, dict[str, Any]] = {}
        self.evaluations = 0  # quantos treinos REAIS foram feitos

    def details(self, individual: Individual) -> dict[str, Any]:
        key = individual_key(individual)
        if key not in self.cache:
            start = time.perf_counter()
            scores = cross_validate(
                build_pipeline(self.model_name, individual), self.X, self.y,
                cv=self.cv, scoring=SCORERS, n_jobs=self.n_jobs,
            )
            self.evaluations += 1
            self.cache[key] = {
                "fitness": float(np.mean(scores["test_f2"])),
                "fitness_std": float(np.std(scores["test_f2"])),
                "recall": float(np.mean(scores["test_recall"])),
                "precision": float(np.mean(scores["test_precision"])),
                "accuracy": float(np.mean(scores["test_accuracy"])),
                "seconds": time.perf_counter() - start,
            }
        return self.cache[key]

    def __call__(self, individual: Individual) -> float:
        return self.details(individual)["fitness"]
