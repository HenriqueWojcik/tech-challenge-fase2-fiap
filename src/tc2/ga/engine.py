"""O LAÇO EVOLUTIVO e o CRITÉRIO DE PARADA (AG Aulas 3 e 5).

O motor é genérico: recebe uma função fitness qualquer. Isso permite testá-lo com uma fitness
"de brinquedo" (ver tests/test_engine.py) sem treinar nenhum modelo.
"""
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from .operators import elite, mutate, select_parent, uniform_crossover
from .space import Individual, individual_key, initial_population

log = logging.getLogger("tc2.ga")


@dataclass
class GAConfig:
    """Parâmetros do AG (AG Aula 3: "parâmetros e métricas de desempenho")."""
    population_size: int = 16
    generations: int = 12
    crossover_rate: float = 0.8   # chance de um filho nascer de crossover (senão, cópia do pai)
    mutation_rate: float = 0.15   # chance de CADA gene mutar
    mutation_strength: float = 0.15
    selection: str = "tournament"  # "tournament" | "roulette"
    tournament_k: int = 3
    elitism: int = 2
    patience: int = 6             # para se o melhor não melhorar por N gerações
    seed_original: bool = False   # incluir o indivíduo da Fase 1 na população inicial
    random_seed: int = 42


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float          # fração de indivíduos únicos na população (0..1)
    new_evaluations: int      # treinos reais nesta geração (o resto veio do cache)
    seconds: float


@dataclass
class GAResult:
    best_individual: Individual
    best_fitness: float
    history: list[GenerationStats] = field(default_factory=list)
    stopped_reason: str = ""
    total_seconds: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d["history"] = [asdict(h) for h in self.history]
        return d


def run_ga(model_name: str, fitness_fn: Callable[[Individual], float], config: GAConfig,
           seed_individuals: list[Individual] | None = None) -> GAResult:
    rng = random.Random(config.random_seed)
    seeds = seed_individuals if config.seed_original else None
    population = initial_population(model_name, config.population_size, rng, seeds)

    best_ind, best_fit = None, float("-inf")
    history: list[GenerationStats] = []
    stale, reason = 0, f"atingiu o máximo de {config.generations} gerações"
    evals_before = getattr(fitness_fn, "evaluations", 0)
    t0 = time.perf_counter()

    for gen in range(config.generations):
        g0 = time.perf_counter()
        # 1) AVALIAÇÃO
        fitnesses = [fitness_fn(ind) for ind in population]

        # guarda o melhor de todos os tempos
        gen_best = max(range(len(population)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best] > best_fit + 1e-9:
            best_ind, best_fit, stale = dict(population[gen_best]), fitnesses[gen_best], 0
        else:
            stale += 1

        evals_now = getattr(fitness_fn, "evaluations", 0)
        stats = GenerationStats(
            generation=gen,
            best_fitness=max(fitnesses),
            mean_fitness=sum(fitnesses) / len(fitnesses),
            worst_fitness=min(fitnesses),
            diversity=len({individual_key(i) for i in population}) / len(population),
            new_evaluations=evals_now - evals_before,
            seconds=time.perf_counter() - g0,
        )
        evals_before = evals_now
        history.append(stats)
        log.info("[%s] geração %02d | melhor=%.4f média=%.4f diversidade=%.2f treinos=%d (%.1fs)",
                 model_name, gen, stats.best_fitness, stats.mean_fitness, stats.diversity,
                 stats.new_evaluations, stats.seconds)

        # 2) CRITÉRIO DE PARADA (AG Aula 5): máximo de gerações OU estagnação
        if stale >= config.patience:
            reason = f"sem melhora por {config.patience} gerações (convergiu)"
            break
        if gen == config.generations - 1:
            break

        # 3) NOVA GERAÇÃO: elitismo + (seleção -> crossover -> mutação)
        next_pop = elite(population, fitnesses, config.elitism)
        while len(next_pop) < config.population_size:
            p1 = select_parent(population, fitnesses, rng, config.selection, config.tournament_k)
            if rng.random() < config.crossover_rate:
                p2 = select_parent(population, fitnesses, rng, config.selection, config.tournament_k)
                child = uniform_crossover(p1, p2, rng)
            else:
                child = dict(p1)
            child = mutate(child, model_name, rng, config.mutation_rate, config.mutation_strength)
            next_pop.append(child)
        population = next_pop

    log.info("[%s] fim: %s | melhor fitness=%.4f | %s", model_name, reason, best_fit, best_ind)
    return GAResult(best_ind, best_fit, history, reason, time.perf_counter() - t0)
