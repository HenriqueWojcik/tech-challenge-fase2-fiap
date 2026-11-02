"""Testes da codificação e dos operadores genéticos (rodam em milissegundos, sem treinar modelos)."""
import random

import pytest

from tc2.ga.operators import (elite, mutate, roulette_selection, tournament_selection,
                              uniform_crossover)
from tc2.ga.space import SEARCH_SPACES, individual_key, initial_population, random_individual
from tc2.models import MODEL_NAMES, build_pipeline


def _in_domain(model_name, ind):
    for g in SEARCH_SPACES[model_name]:
        v = ind[g.name]
        if g.kind == "cat":
            assert v in g.choices, (g.name, v)
        else:
            assert g.low <= v <= g.high, (g.name, v)
            if g.kind == "int":
                assert isinstance(v, int)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_random_individual_is_valid_and_builds_pipeline(model_name):
    rng = random.Random(0)
    for _ in range(50):
        ind = random_individual(model_name, rng)
        _in_domain(model_name, ind)
    build_pipeline(model_name, ind)  # não pode lançar exceção


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_mutation_keeps_genes_in_domain(model_name):
    rng = random.Random(1)
    ind = random_individual(model_name, rng)
    for _ in range(200):
        ind = mutate(ind, model_name, rng, rate=1.0, strength=0.5)
        _in_domain(model_name, ind)


def test_mutation_rate_zero_changes_nothing():
    rng = random.Random(2)
    ind = random_individual("random_forest", rng)
    assert mutate(ind, "random_forest", rng, rate=0.0) == ind


def test_uniform_crossover_genes_come_from_parents():
    rng = random.Random(3)
    p1, p2 = random_individual("random_forest", rng), random_individual("random_forest", rng)
    for _ in range(30):
        child = uniform_crossover(p1, p2, rng)
        assert set(child) == set(p1)
        assert all(child[g] in (p1[g], p2[g]) for g in child)


def test_tournament_with_full_size_returns_best():
    pop = [{"x": i} for i in range(5)]
    fits = [0.1, 0.9, 0.3, 0.5, 0.2]
    assert tournament_selection(pop, fits, random.Random(0), k=5) == {"x": 1}


def test_roulette_prefers_fitter_individuals():
    pop = [{"x": 0}, {"x": 1}]
    fits = [0.90, 0.99]
    rng = random.Random(0)
    picks = [roulette_selection(pop, fits, rng)["x"] for _ in range(500)]
    assert picks.count(1) > picks.count(0)


def test_elitism_keeps_the_best():
    pop = [{"x": i} for i in range(4)]
    assert elite(pop, [0.2, 0.8, 0.5, 0.1], 2) == [{"x": 1}, {"x": 2}]


def test_seeded_population_contains_seed_and_has_right_size():
    seed = {"C": 1.0, "l1_ratio": 0.0, "class_weight": None}
    pop = initial_population("logistic_regression", 8, random.Random(0), [seed])
    assert len(pop) == 8 and pop[0] == seed


def test_individual_key_is_order_independent():
    assert individual_key({"a": 1, "b": 0.1234567}) == individual_key({"b": 0.12345671, "a": 1})
