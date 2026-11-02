"""Experimentos do AG + comparação com o modelo original (requisito 1 do enunciado).

Enunciado: "Realizar ao menos 3 experimentos com diferentes configurações do algoritmo
genético" e "Comparar o desempenho dos modelos otimizados com os modelos originais".

Cada experimento muda UMA ideia em relação ao experimento base, para que a diferença de
resultado possa ser atribuída a essa ideia (AG Aula 3: convergência x diversidade).

Comparamos também com uma BUSCA ALEATÓRIA com o mesmo orçamento de treinos.
Isso responde à pergunta que todo avaliador faz: "o AG é melhor que sortear hiperparâmetros?"
(ML na Cloud Aula 3 apresenta Hyperparameter Tuning por grid/random search.)
"""
import json
import logging
import random
import time
from dataclasses import asdict, replace

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402

from .config import MALIGNANT, RESULTS_DIR  # noqa: E402
from .data import load_data  # noqa: E402
from .ga.engine import GAConfig, run_ga  # noqa: E402
from .ga.fitness import CVFitness  # noqa: E402
from .ga.space import random_individual  # noqa: E402
from .models import MODEL_NAMES, build_pipeline, evaluate  # noqa: E402

log = logging.getLogger("tc2.experiments")

# Indivíduo que representa EXATAMENTE o modelo da Fase 1 dentro do espaço de busca
# (n_estimators=200 + todos os outros hiperparâmetros no padrão do scikit-learn).
ORIGINAL_INDIVIDUAL = {
    "random_forest": {"n_estimators": 200, "max_depth": None, "min_samples_split": 2,
                      "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini",
                      "class_weight": None},
    "logistic_regression": {"C": 1.0, "l1_ratio": 0.0, "class_weight": None},
}

BASE = GAConfig()
EXPERIMENTS: dict[str, tuple[str, GAConfig]] = {
    "E1_base": ("Configuração de referência: torneio k=3, elitismo 2, mutação 15%", BASE),
    "E2_mutacao_alta": ("Mutação 40% com passo maior: mais exploração, menos convergência",
                        replace(BASE, mutation_rate=0.40, mutation_strength=0.30)),
    "E3_populacao_maior": ("População 28 com menos gerações (mesmo orçamento): mais diversidade inicial",
                           replace(BASE, population_size=28, generations=7)),
    "E4_roleta_sem_elitismo": ("Seleção por roleta e sem elitismo: menor pressão seletiva",
                               replace(BASE, selection="roulette", elitism=0)),
    "E5_semeado_fase1": ("Igual ao E1, mas com o modelo da Fase 1 na população inicial",
                         replace(BASE, seed_original=True)),
}


def quick_mode(cfg: GAConfig) -> GAConfig:
    """Versão reduzida para testar o pipeline em segundos."""
    return replace(cfg, population_size=min(cfg.population_size, 6), generations=3, patience=3)


def random_search(model_name: str, fitness: CVFitness, budget: int, seed: int = 42):
    rng = random.Random(seed)
    best, best_fit = None, float("-inf")
    curve = []
    for _ in range(budget):
        ind = random_individual(model_name, rng)
        f = fitness(ind)
        if f > best_fit:
            best, best_fit = ind, f
        curve.append(best_fit)
    return best, best_fit, curve


def _fmt(params: dict) -> str:
    return ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in params.items())


def run_all(quick: bool = False, models: tuple[str, ...] = MODEL_NAMES) -> dict:
    data = load_data()
    summary_rows, all_results = [], {}
    (RESULTS_DIR / "models").mkdir(exist_ok=True)

    for model_name in models:
        log.info("=" * 70)
        log.info("MODELO: %s", model_name)
        model_results = {}

        # ---------- referência: modelo ORIGINAL da Fase 1 ----------
        orig_fit = CVFitness(model_name, data.X_train, data.y_train)
        orig_ind = ORIGINAL_INDIVIDUAL[model_name]
        orig_cv = orig_fit.details(orig_ind)
        orig_pipe = build_pipeline(model_name, orig_ind).fit(data.X_train, data.y_train)
        orig_test = evaluate(orig_pipe, data.X_test, data.y_test)
        model_results["original_fase1"] = {"params": orig_ind, "cv": orig_cv, "test": orig_test}
        summary_rows.append({"modelo": model_name, "abordagem": "original_fase1",
                             "cv_f2": orig_cv["fitness"], "cv_f2_std": orig_cv["fitness_std"],
                             "treinos": 1, **{f"teste_{k}": v for k, v in orig_test.items()},
                             "params": _fmt(orig_ind)})
        log.info("Original (Fase 1): CV F2=%.4f | teste recall=%.4f FN=%d",
                 orig_cv["fitness"], orig_test["recall_maligno"], orig_test["falsos_negativos"])

        # ---------- experimentos do AG ----------
        budget_ref = None
        for exp_name, (desc, cfg) in EXPERIMENTS.items():
            cfg = quick_mode(cfg) if quick else cfg
            log.info("-" * 70)
            log.info("%s - %s", exp_name, desc)
            fitness = CVFitness(model_name, data.X_train, data.y_train)  # cache novo = comparação justa
            result = run_ga(model_name, fitness, cfg, seed_individuals=[orig_ind])
            budget_ref = budget_ref or fitness.evaluations

            pipe = build_pipeline(model_name, result.best_individual).fit(data.X_train, data.y_train)
            test = evaluate(pipe, data.X_test, data.y_test)
            model_results[exp_name] = {
                "descricao": desc, "config": asdict(cfg), "ga": result.to_dict(),
                "cv": fitness.details(result.best_individual), "test": test,
                "treinos": fitness.evaluations,
            }
            summary_rows.append({"modelo": model_name, "abordagem": exp_name,
                                 "cv_f2": result.best_fitness,
                                 "cv_f2_std": fitness.details(result.best_individual)["fitness_std"],
                                 "treinos": fitness.evaluations,
                                 **{f"teste_{k}": v for k, v in test.items()},
                                 "params": _fmt(result.best_individual)})

        # ---------- busca aleatória com o mesmo orçamento do E1 ----------
        rs_fit = CVFitness(model_name, data.X_train, data.y_train)
        rs_best, rs_score, rs_curve = random_search(model_name, rs_fit, budget_ref)
        rs_pipe = build_pipeline(model_name, rs_best).fit(data.X_train, data.y_train)
        rs_test = evaluate(rs_pipe, data.X_test, data.y_test)
        model_results["busca_aleatoria"] = {"params": rs_best, "cv": rs_fit.details(rs_best),
                                            "test": rs_test, "curve": rs_curve,
                                            "treinos": rs_fit.evaluations}
        summary_rows.append({"modelo": model_name, "abordagem": "busca_aleatoria",
                             "cv_f2": rs_score, "cv_f2_std": rs_fit.details(rs_best)["fitness_std"],
                             "treinos": rs_fit.evaluations,
                             **{f"teste_{k}": v for k, v in rs_test.items()}, "params": _fmt(rs_best)})
        log.info("Busca aleatória (%d treinos): CV F2=%.4f", budget_ref, rs_score)

        # ---------- escolhe o MELHOR pela validação cruzada (nunca pelo teste!) ----------
        ga_exps = [k for k in model_results if k.startswith("E")]
        best_exp = max(ga_exps, key=lambda k: model_results[k]["cv"]["fitness"])
        best_params = model_results[best_exp]["ga"]["best_individual"]
        best_pipe = build_pipeline(model_name, best_params).fit(data.X_train, data.y_train)
        joblib.dump(best_pipe, RESULTS_DIR / "models" / f"{model_name}_otimizado.joblib")
        joblib.dump(orig_pipe, RESULTS_DIR / "models" / f"{model_name}_original.joblib")
        model_results["melhor"] = {"experimento": best_exp, "params": best_params}

        _plot_convergence(model_name, model_results)
        _plot_confusion(model_name, orig_pipe, best_pipe, best_exp, data)
        all_results[model_name] = model_results

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "resumo_experimentos.csv", index=False)
    with open(RESULTS_DIR / "resultados_completos.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    _plot_comparison(summary)
    return all_results


# ----------------------------------------------------------------------------- gráficos
def _plot_convergence(model_name: str, res: dict):
    """Curvas de convergência (AG Aula 5 - visualizar a evolução; AG Aula 3 - métricas)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for exp in [k for k in res if k.startswith("E")]:
        hist = res[exp]["ga"]["history"]
        gens = [h["generation"] for h in hist]
        axes[0].plot(gens, [h["best_fitness"] for h in hist], marker="o", label=f"{exp} (melhor)")
        axes[0].plot(gens, [h["mean_fitness"] for h in hist], linestyle="--", alpha=0.5)
        axes[1].plot(gens, [h["diversity"] for h in hist], marker=".", label=exp)
    axes[0].axhline(res["original_fase1"]["cv"]["fitness"], color="black", linestyle=":",
                    label="original Fase 1")
    axes[0].set(title=f"{model_name} - fitness (F2 em CV) por geração\nlinha cheia = melhor, tracejada = média",
                xlabel="geração", ylabel="F2 (classe maligna)")
    axes[0].legend(fontsize=7)
    axes[1].set(title="Diversidade da população (fração de indivíduos únicos)",
                xlabel="geração", ylabel="diversidade", ylim=(0, 1.05))
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"convergencia_{model_name}.png", dpi=110)
    plt.close()


def _plot_confusion(model_name, orig_pipe, best_pipe, best_exp, data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [MALIGNANT, 1 - MALIGNANT]
    for ax, pipe, title in [(axes[0], orig_pipe, "Original (Fase 1)"),
                            (axes[1], best_pipe, f"Otimizado ({best_exp})")]:
        cm = confusion_matrix(data.y_test, pipe.predict(data.X_test), labels=labels)
        ConfusionMatrixDisplay(cm, display_labels=["maligno", "benigno"]).plot(ax=ax, colorbar=False)
        ax.set_title(title)
    fig.suptitle(f"{model_name} - matriz de confusão no teste (114 pacientes)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"matriz_confusao_{model_name}.png", dpi=110)
    plt.close()


def _plot_comparison(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, model_name in zip(axes, summary["modelo"].unique()):
        sub = summary[summary["modelo"] == model_name]
        x = range(len(sub))
        ax.bar([i - 0.2 for i in x], sub["cv_f2"], width=0.4, label="F2 validação cruzada")
        ax.bar([i + 0.2 for i in x], sub["teste_f2_maligno"], width=0.4, label="F2 teste")
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["abordagem"], rotation=30, ha="right", fontsize=8)
        lo = min(sub["cv_f2"].min(), sub["teste_f2_maligno"].min())
        ax.set_ylim(max(0, lo - 0.03), 1.0)
        ax.set_title(model_name)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparacao_abordagens.png", dpi=110)
    plt.close()
