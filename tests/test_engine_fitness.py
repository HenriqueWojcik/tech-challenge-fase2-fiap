"""Testes do laço evolutivo (com fitness de brinquedo) e da fitness real (com cache)."""
import random

from tc2.data import load_data
from tc2.ga.engine import GAConfig, run_ga
from tc2.ga.fitness import CVFitness
from tc2.ga.space import random_individual


def toy_fitness(ind):
    """Ótimo conhecido: C = 10, l1_ratio = 0.5, class_weight = 'balanced'."""
    import math
    score = -abs(math.log10(ind["C"]) - 1) - abs(ind["l1_ratio"] - 0.5)
    return score + (0.2 if ind["class_weight"] == "balanced" else 0.0)


def test_ga_improves_over_random_initial_population():
    cfg = GAConfig(population_size=20, generations=25, patience=25, random_seed=0)
    result = run_ga("logistic_regression", toy_fitness, cfg)
    initial_best = result.history[0].best_fitness
    assert result.best_fitness > initial_best
    assert result.best_individual["class_weight"] == "balanced"
    assert abs(result.best_individual["l1_ratio"] - 0.5) < 0.15


def test_best_fitness_never_decreases_with_elitism():
    cfg = GAConfig(population_size=10, generations=10, patience=10, elitism=1, random_seed=1)
    hist = run_ga("logistic_regression", toy_fitness, cfg).history
    bests = [h.best_fitness for h in hist]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(bests, bests[1:]))


def test_ga_is_reproducible_with_same_seed():
    cfg = GAConfig(population_size=8, generations=5, random_seed=7)
    a = run_ga("logistic_regression", toy_fitness, cfg)
    b = run_ga("logistic_regression", toy_fitness, cfg)
    assert a.best_individual == b.best_individual


def test_early_stopping_by_patience():
    cfg = GAConfig(population_size=6, generations=50, patience=2, random_seed=0)
    result = run_ga("logistic_regression", lambda ind: 1.0, cfg)  # fitness constante
    assert len(result.history) == 3 and "convergiu" in result.stopped_reason


def test_cv_fitness_is_in_range_and_cached():
    data = load_data()
    fit = CVFitness("logistic_regression", data.X_train, data.y_train, n_jobs=1)
    ind = random_individual("logistic_regression", random.Random(0))
    f1 = fit(ind)
    f2 = fit(dict(ind))
    assert 0.0 <= f1 <= 1.0 and f1 == f2
    assert fit.evaluations == 1  # segunda chamada veio do cache


def test_split_is_the_same_as_phase_1():
    data = load_data()
    assert data.X_train.shape == (455, 30) and data.X_test.shape == (114, 30)
    assert (data.y_test == 0).sum() == 42  # 42 malignos no teste, como na Fase 1
