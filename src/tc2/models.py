"""Modelos de diagnóstico: os ORIGINAIS da Fase 1 e a fábrica de modelos a partir de genes.

Fase 1 (notebook 03):
- Regressão Logística: Pipeline(StandardScaler, LogisticRegression(max_iter=10000, random_state=42))
- Random Forest:       Pipeline(StandardScaler, RandomForestClassifier(n_estimators=200, random_state=42))

O AG não muda o pipeline - ele só escolhe os hiperparâmetros do passo 'clf'.
"""
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MALIGNANT, RANDOM_STATE

MODEL_NAMES = ("random_forest", "logistic_regression")

# Hiperparâmetros usados na Fase 1 (o "indivíduo" de referência).
ORIGINAL_PARAMS: dict[str, dict[str, Any]] = {
    "random_forest": {"n_estimators": 200},
    "logistic_regression": {"max_iter": 10000},
}


def build_pipeline(model_name: str, params: dict[str, Any] | None = None) -> Pipeline:
    """Monta o mesmo Pipeline da Fase 1 com os hiperparâmetros recebidos."""
    params = dict(params or {})
    if model_name == "random_forest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)
    elif model_name == "logistic_regression":
        if params.get("l1_ratio", 0) > 0:
            # l1_ratio > 0 (L1 ou Elastic Net) exige o solver 'saga'; o padrão 'lbfgs' só faz L2.
            params.setdefault("solver", "saga")
        params.setdefault("max_iter", 10000)
        clf = LogisticRegression(random_state=RANDOM_STATE, **params)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def original_pipeline(model_name: str) -> Pipeline:
    return build_pipeline(model_name, ORIGINAL_PARAMS[model_name])


def evaluate(pipe: Pipeline, X_test, y_test) -> dict[str, float]:
    """Métricas no conjunto de teste, sempre olhando a classe MALIGNA como positiva."""
    y_pred = pipe.predict(X_test)
    proba_malignant = pipe.predict_proba(X_test)[:, list(pipe.classes_).index(MALIGNANT)]
    tn_fp_fn_tp = confusion_matrix(y_test, y_pred, labels=[1 - MALIGNANT, MALIGNANT]).ravel()
    _, _, fn, _ = (int(v) for v in tn_fp_fn_tp)
    return {
        "recall_maligno": recall_score(y_test, y_pred, pos_label=MALIGNANT),
        "precision_maligno": precision_score(y_test, y_pred, pos_label=MALIGNANT),
        "f1_maligno": f1_score(y_test, y_pred, pos_label=MALIGNANT),
        "f2_maligno": fbeta_score(y_test, y_pred, beta=2, pos_label=MALIGNANT),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score((np.asarray(y_test) == MALIGNANT).astype(int), proba_malignant),
        "falsos_negativos": fn,
    }
