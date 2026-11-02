"""Monta o "caso clínico estruturado" que a LLM vai interpretar.

Ideia central: a LLM NÃO olha os dados brutos nem decide nada. O modelo de ML faz a previsão,
o SHAP (já usado na Fase 1) diz POR QUE, e a LLM só traduz esses números para linguagem
de médico. Isso reduz alucinação (LLMs Aula 1) porque a LLM tem tudo o que precisa no prompt
e é instruída a não usar nada além disso.

O dicionário gerado aqui também é o formato que o Módulo 3 vai reaproveitar (enunciado:
"preparar a base para a futura integração com dados textuais").
"""
import numpy as np
import pandas as pd
import shap

from .config import BENIGN, CLASS_NAMES, MALIGNANT

_BASE_PT = {
    "radius": "raio", "texture": "textura", "perimeter": "perímetro", "area": "área",
    "smoothness": "suavidade", "compactness": "compacidade", "concavity": "concavidade",
    "concave points": "pontos côncavos", "symmetry": "simetria",
    "fractal dimension": "dimensão fractal",
}


def feature_pt(name: str) -> str:
    """'worst concave points' -> 'pontos côncavos (pior valor)'."""
    for prefix, suffix in (("mean ", "média"), ("worst ", "pior valor")):
        if name.startswith(prefix):
            base = name[len(prefix):]
            return f"{_BASE_PT.get(base, base)} ({suffix})"
    if name.endswith(" error"):
        base = name[: -len(" error")]
        return f"{_BASE_PT.get(base, base)} (erro padrão)"
    return name


def confidence_level(p_malignant: float) -> str:
    distance = abs(p_malignant - 0.5)
    if distance >= 0.35:
        return "alta"
    if distance >= 0.15:
        return "moderada"
    return "baixa (zona de incerteza)"


class CaseBuilder:
    """Calcula SHAP para um pipeline (scaler + clf) e monta casos por paciente."""

    def __init__(self, pipe, X_train: pd.DataFrame, X_all: pd.DataFrame, y_all: pd.Series,
                 model_label: str, test_metrics: dict | None = None):
        self.pipe = pipe
        self.scaler = pipe.named_steps["scaler"]
        self.clf = pipe.named_steps["clf"]
        self.model_label = model_label
        self.test_metrics = test_metrics or {}
        self.means_malignant = X_all[y_all == MALIGNANT].mean()
        self.means_benign = X_all[y_all == BENIGN].mean()
        background = self.scaler.transform(X_train)
        if hasattr(self.clf, "estimators_"):          # Random Forest
            self.explainer = shap.TreeExplainer(self.clf)
        else:                                          # Regressão Logística
            self.explainer = shap.LinearExplainer(self.clf, background)

    def _shap_towards_malignant(self, x_scaled: np.ndarray) -> np.ndarray:
        values = self.explainer.shap_values(x_scaled)
        malignant_idx = list(self.clf.classes_).index(MALIGNANT)
        if isinstance(values, list):                   # versões antigas do shap
            return np.asarray(values[malignant_idx])[0]
        values = np.asarray(values)
        if values.ndim == 3:                           # (amostras, features, classes)
            return values[0, :, malignant_idx]
        # LinearExplainer binário explica a classe 1 (benigno) -> inverte o sinal
        return -values[0] if MALIGNANT == 0 else values[0]

    def build(self, patient_row: pd.Series, patient_id, top_k: int = 5) -> dict:
        x = patient_row.to_frame().T
        x_scaled = self.scaler.transform(x)
        p_mal = float(self.pipe.predict_proba(x)[0, list(self.pipe.classes_).index(MALIGNANT)])
        pred = int(self.pipe.predict(x)[0])
        contrib = self._shap_towards_malignant(x_scaled)
        order = np.argsort(-np.abs(contrib))[:top_k]

        fatores = []
        for i in order:
            name = patient_row.index[i]
            fatores.append({
                "caracteristica": feature_pt(name),
                "nome_original": name,
                "valor_paciente": round(float(patient_row.iloc[i]), 4),
                "media_casos_malignos": round(float(self.means_malignant[name]), 4),
                "media_casos_benignos": round(float(self.means_benign[name]), 4),
                "impacto_shap": round(float(contrib[i]), 4),
                "efeito": "aumenta a chance de malignidade" if contrib[i] > 0
                          else "reduz a chance de malignidade",
            })

        return {
            "paciente_id": str(patient_id),
            "modelo": self.model_label,
            "previsao": CLASS_NAMES[pred],
            "probabilidade_maligno": round(p_mal, 4),
            "probabilidade_maligno_pct": round(100 * p_mal, 1),
            "nivel_confianca": confidence_level(p_mal),
            "fatores_principais": fatores,
            "desempenho_do_modelo_no_teste": {k: round(float(v), 4) for k, v in self.test_metrics.items()},
            "origem_dos_dados": "exame citológico por punção aspirativa (FNA) - dataset Breast Cancer Wisconsin",
        }
