"""OPERADORES GENÉTICOS: seleção (AG Aula 4), crossover e mutação (AG Aula 5).

Diferença importante em relação ao TSP das aulas:
- No TSP, o indivíduo é uma permutação. Por isso a aula usa Ordered Crossover (OX) e mutação
  Swap: trocar posições preserva a regra "cada cidade aparece exatamente uma vez".
- Aqui os genes são INDEPENDENTES (n_estimators não depende de max_depth). Não existe a
  restrição de permutação, então usamos operadores para genes independentes:
    * crossover UNIFORME: cada gene vem do pai 1 ou do pai 2 (moeda 50/50);
    * mutação POR GENE: cada gene, com probabilidade = taxa de mutação, sofre uma perturbação
      adequada ao seu tipo (gaussiana p/ números, sorteio de outra opção p/ categóricos).
Usar OX aqui não faria sentido - é o exemplo do "crossover depende da representação" (AG Aula 5).
"""
import math
import random

from .space import SEARCH_SPACES, Gene, Individual


# ----------------------------------------------------------------------------- SELEÇÃO
def tournament_selection(population: list[Individual], fitnesses: list[float],
                         rng: random.Random, k: int = 3) -> Individual:
    """Torneio: sorteia k indivíduos e devolve o melhor deles.

    k controla a PRESSÃO SELETIVA: k grande = quase sempre os melhores (converge rápido,
    perde diversidade); k pequeno = mais chance para os fracos (explora mais).
    """
    contenders = rng.sample(range(len(population)), k=min(k, len(population)))
    best = max(contenders, key=lambda i: fitnesses[i])
    return population[best]


def roulette_selection(population: list[Individual], fitnesses: list[float],
                       rng: random.Random) -> Individual:
    """Roleta (seleção probabilística da AG Aula 4, com random.choices).

    Probabilidade proporcional à fitness. Como as fitness aqui ficam todas perto de 0.9,
    subtraímos o mínimo da geração; senão todos teriam quase a mesma chance (pressão ~ zero).
    """
    floor = min(fitnesses)
    weights = [f - floor + 1e-6 for f in fitnesses]
    return rng.choices(population, weights=weights, k=1)[0]


def select_parent(population, fitnesses, rng, method: str = "tournament", k: int = 3):
    if method == "tournament":
        return tournament_selection(population, fitnesses, rng, k)
    if method == "roulette":
        return roulette_selection(population, fitnesses, rng)
    raise ValueError(f"Seleção desconhecida: {method}")


def elite(population: list[Individual], fitnesses: list[float], n: int) -> list[Individual]:
    """ELITISMO: os n melhores passam intactos para a próxima geração."""
    order = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
    return [dict(population[i]) for i in order[:n]]


# ----------------------------------------------------------------------------- CROSSOVER
def uniform_crossover(parent1: Individual, parent2: Individual, rng: random.Random) -> Individual:
    return {gene: (parent1[gene] if rng.random() < 0.5 else parent2[gene]) for gene in parent1}


# ----------------------------------------------------------------------------- MUTAÇÃO
def _mutate_gene(gene: Gene, value, rng: random.Random, strength: float):
    if gene.kind == "int":
        sigma = max(1.0, (gene.high - gene.low) * strength)
        return gene.clip(value + rng.gauss(0, sigma))
    if gene.kind == "float":
        return gene.clip(value + rng.gauss(0, (gene.high - gene.low) * strength))
    if gene.kind == "log_float":
        # perturba o expoente: C=1 pode virar 0.3 ou 3, nunca salta de 0.001 para 1000 de uma vez
        span = math.log10(gene.high) - math.log10(gene.low)
        return gene.clip(10 ** (math.log10(value) + rng.gauss(0, span * strength)))
    if gene.kind == "cat":
        others = [c for c in gene.choices if c != value]
        return rng.choice(others) if others else value
    raise ValueError(gene.kind)


def mutate(individual: Individual, model_name: str, rng: random.Random,
           rate: float, strength: float = 0.15) -> Individual:
    """Cada gene muta com probabilidade `rate`. `strength` = tamanho do passo (fração do domínio)."""
    child = dict(individual)
    for gene in SEARCH_SPACES[model_name]:
        if rng.random() < rate:
            child[gene.name] = _mutate_gene(gene, child[gene.name], rng, strength)
    return child
