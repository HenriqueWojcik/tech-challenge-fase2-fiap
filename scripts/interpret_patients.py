"""Gera explicações em linguagem natural para pacientes do teste e avalia a qualidade.

Pré-requisito: ter rodado scripts/run_optimization.py (usa os modelos salvos em results/models).

Uso:
    python scripts/interpret_patients.py                      # provedor do .env (ou offline)
    python scripts/interpret_patients.py --provider groq
    python scripts/interpret_patients.py --model random_forest --n-extra 3

Saídas em results/llm/:
    caso_<id>_engenheirado.md / caso_<id>_ingenuo.md   explicações geradas
    avaliacao_llm.csv                                  métricas de qualidade por caso e prompt
    casos_modulo3.jsonl                                casos estruturados (base para o Módulo 3)
    relatorio_modelos.md                               insights da otimização para a diretoria
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from tc2.config import CLASS_NAMES, MALIGNANT, RESULTS_DIR  # noqa: E402
from tc2.data import load_data  # noqa: E402
from tc2.explain import CaseBuilder  # noqa: E402
from tc2.llm.client import LLMClient  # noqa: E402
from tc2.llm.evaluation import evaluate_explanation, llm_judge  # noqa: E402
from tc2.llm.prompts import (build_explanation_prompt, build_model_report_prompt,  # noqa: E402
                             build_naive_prompt, offline_explanation)
from tc2.logging_utils import setup_logging  # noqa: E402
from tc2.models import evaluate  # noqa: E402

OUT = RESULTS_DIR / "llm"


def pick_model(arg: str) -> str:
    if arg != "auto":
        return arg
    res = json.loads((RESULTS_DIR / "resultados_completos.json").read_text(encoding="utf-8"))
    # melhor modelo = maior F2 em validação cruzada do seu melhor experimento
    return max(res, key=lambda m: res[m][res[m]["melhor"]["experimento"]]["cv"]["fitness"])


def select_cases(data, pipe_opt, pipe_orig, n_extra: int) -> list:
    """Casos interessantes do teste: falsos negativos do original + extremos + zona de incerteza."""
    X, y = data.X_test, data.y_test
    proba = pd.Series(pipe_opt.predict_proba(X)[:, list(pipe_opt.classes_).index(MALIGNANT)], index=X.index)
    orig_pred = pd.Series(pipe_orig.predict(X), index=X.index)
    chosen = list(X.index[(y == MALIGNANT) & (orig_pred != MALIGNANT)])   # câncer que o original deixou passar
    chosen += [proba.idxmax(), proba.idxmin(), (proba - 0.5).abs().idxmin()]
    chosen += list((proba - 0.5).abs().sort_values().index[1:1 + max(0, n_extra - 1)])
    return list(dict.fromkeys(chosen))  # remove repetidos mantendo a ordem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto", choices=["auto", "random_forest", "logistic_regression"])
    ap.add_argument("--provider", default=None, help="groq | openai | ollama | offline")
    ap.add_argument("--n-extra", type=int, default=1, help="casos extras na zona de incerteza")
    ap.add_argument("--no-judge", action="store_true", help="não usar LLM-como-juiz")
    args = ap.parse_args()

    setup_logging("interpretacao")
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    model_name = pick_model(args.model)
    pipe_opt = joblib.load(RESULTS_DIR / "models" / f"{model_name}_otimizado.joblib")
    pipe_orig = joblib.load(RESULTS_DIR / "models" / f"{model_name}_original.joblib")
    metrics = evaluate(pipe_opt, data.X_test, data.y_test)
    builder = CaseBuilder(pipe_opt, data.X_train, data.X, data.y,
                          f"{model_name} otimizado por algoritmo genético",
                          {k: metrics[k] for k in ("recall_maligno", "precision_maligno", "accuracy")})
    client = LLMClient(provider=args.provider)
    print(f"Modelo: {model_name} | LLM: {client.provider}/{client.model}")

    rows = []
    with open(OUT / "casos_modulo3.jsonl", "w", encoding="utf-8") as jsonl:
        for idx in select_cases(data, pipe_opt, pipe_orig, args.n_extra):
            case = builder.build(data.X_test.loc[idx], idx)
            real = CLASS_NAMES[int(data.y_test.loc[idx])]  # NÃO vai para a LLM - só para análise
            jsonl.write(json.dumps({**case, "classe_real": real}, ensure_ascii=False) + "\n")

            prompts = {"engenheirado": build_explanation_prompt(case)}
            if not client.is_offline:
                prompts["ingenuo"] = build_naive_prompt(case)
            for kind, (system, user) in prompts.items():
                text = client.complete(system, user, offline_fallback=lambda: offline_explanation(case))
                (OUT / f"caso_{idx}_{kind}.md").write_text(text, encoding="utf-8")
                ev = evaluate_explanation(text, case)
                judge = None if args.no_judge else llm_judge(client, text, case)
                rows.append({"paciente_id": idx, "classe_real": real, "previsao": case["previsao"],
                             "prob_maligno": case["probabilidade_maligno"], "prompt": kind, **ev,
                             **({f"juiz_{k}": v for k, v in judge.items()} if judge else {})})
                print(f"  paciente {idx} ({real} -> previsto {case['previsao']}, "
                      f"p={case['probabilidade_maligno']:.2f}) [{kind}] nota={ev['nota_geral']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "avaliacao_llm.csv", index=False)
    print("\nMédia das métricas automáticas por tipo de prompt:")
    print(df.groupby("prompt")[["estrutura", "classe_correta", "cobertura_fatores",
                                "fidelidade_numerica", "seguranca", "tamanho_adequado",
                                "nota_geral"]].mean().round(3).to_string())

    # Relatório de insights da otimização (números -> conclusões acionáveis)
    summary = pd.read_csv(RESULTS_DIR / "resumo_experimentos.csv")
    cols = ["modelo", "abordagem", "cv_f2", "teste_recall_maligno", "teste_precision_maligno",
            "teste_falsos_negativos", "treinos"]
    summary_rows = summary[cols].round(4).to_dict(orient="records")
    system, user = build_model_report_prompt(summary_rows)
    report = client.complete(system, user, offline_fallback=lambda: (
        "## Relatório (modo offline)\nConfigure uma LLM no .env para gerar este relatório.\n\n"
        + summary[cols].round(4).to_markdown(index=False)))
    (OUT / "relatorio_modelos.md").write_text(report, encoding="utf-8")
    print(f"\nArquivos em {OUT}")


if __name__ == "__main__":
    main()
