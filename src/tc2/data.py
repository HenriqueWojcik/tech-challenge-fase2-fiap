"""Carregamento dos dados - idêntico ao notebook 03 da Fase 1.

Manter exatamente o mesmo split (random_state=42, stratify=y, 80/20) é o que permite
afirmar no relatório que o modelo otimizado foi comparado com o original nas MESMAS pacientes.
"""
from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


def load_data() -> Dataset:
    raw = load_breast_cancer()
    X = pd.DataFrame(raw.data, columns=raw.feature_names)
    y = pd.Series(raw.target, name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return Dataset(X, y, X_train, X_test, y_train, y_test)
