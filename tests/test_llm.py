"""Testes da parte de LLM - todos no modo offline (sem chave, sem internet)."""
from tc2.data import load_data
from tc2.explain import CaseBuilder, feature_pt
from tc2.llm.client import LLMClient
from tc2.llm.evaluation import evaluate_explanation
from tc2.llm.prompts import build_explanation_prompt, offline_explanation
from tc2.models import build_pipeline


def _case(model_name="logistic_regression"):
    data = load_data()
    pipe = build_pipeline(model_name, {"n_estimators": 50} if model_name == "random_forest" else {})
    pipe.fit(data.X_train, data.y_train)
    builder = CaseBuilder(pipe, data.X_train, data.X, data.y, model_name,
                          {"recall_maligno": 0.97, "precision_maligno": 0.95})
    idx = data.X_test.index[0]
    return builder.build(data.X_test.loc[idx], idx)


def test_feature_translation():
    assert feature_pt("worst concave points") == "pontos côncavos (pior valor)"
    assert feature_pt("mean radius") == "raio (média)"
    assert feature_pt("texture error") == "textura (erro padrão)"


def test_case_structure_for_both_models():
    for model in ("logistic_regression", "random_forest"):
        case = _case(model)
        assert case["previsao"] in ("maligno", "benigno")
        assert 0 <= case["probabilidade_maligno"] <= 1
        assert len(case["fatores_principais"]) == 5
        assert "classe_real" not in case  # o gabarito nunca vai para a LLM


def test_prompt_contains_data_and_rules():
    case = _case()
    system, user = build_explanation_prompt(case)
    assert "<caso>" in user and case["fatores_principais"][0]["caracteristica"] in user
    assert "Não invente" in system and "não substitui" in system


def test_client_without_key_falls_back_to_offline(monkeypatch):
    # impede que um .env real (com chave) seja recarregado pelo cliente durante o teste
    monkeypatch.setattr("tc2.llm.client._load_env", lambda: None)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = LLMClient(provider="groq", api_key=None)
    assert client.is_offline
    assert client.complete("s", "u", offline_fallback=lambda: "ok") == "ok"


def test_offline_explanation_scores_well():
    case = _case()
    ev = evaluate_explanation(offline_explanation(case), case)
    assert ev["estrutura"] == 1.0 and ev["classe_correta"] == 1.0
    assert ev["fidelidade_numerica"] == 1.0 and ev["seguranca"] == 1.0


def test_evaluation_detects_hallucinated_numbers_and_unsafe_text():
    case = _case()
    bad = ("O paciente com certeza absoluta tem câncer. Probabilidade de 73,2% e "
           "raio de 999,9. Resumo feito.")
    ev = evaluate_explanation(bad, case)
    assert ev["fidelidade_numerica"] < 1.0 and 999.9 in ev["numeros_sem_suporte"]
    assert ev["seguranca"] == 0.0
    assert ev["nota_geral"] < 0.6
