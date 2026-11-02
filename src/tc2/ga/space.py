"""CODIFICAÇÃO dos indivíduos (AG Aula 3: binária, real, combinatória, híbrida).

No TSP das Aulas 4 e 5, um indivíduo era uma PERMUTAÇÃO de cidades (codificação combinatória).
Aqui um indivíduo é um CONJUNTO DE HIPERPARÂMETROS, e cada hiperparâmetro é um gene.
Como os genes têm tipos diferentes (inteiro, real, categórico), a codificação é HÍBRIDA:

    indivíduo = {"n_estimators": 250, "max_depth": 8, "max_features": "sqrt", ...}
                   gene inteiro        gene categórico   gene categórico

Por que não binária? Converter "max_features='sqrt'" ou "C=0.03" em bits cria
combinações inválidas e faz um único bit trocado mudar o valor de forma brusca.
A codificação direta (real/categórica) mantém cada gene sempre válido.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Any

Individual = dict[str, Any]


@dataclass(frozen=True)
class Gene:
    name: str
    kind: str  # "int" | "float" | "log_float" | "cat"
    low: float | None = None
    high: float | None = None
    choices: tuple = field(default_factory=tuple)

    def sample(self, rng: random.Random) -> Any:
        """Valor aleatório válido - usado na INICIALIZAÇÃO da população."""
        if self.kind == "int":
            return rng.randint(int(self.low), int(self.high))
        if self.kind == "float":
            return rng.uniform(self.low, self.high)
        if self.kind == "log_float":
            # Amostra uniforme no EXPOENTE: C=0.001 e C=1000 ficam igualmente prováveis.
            return 10 ** rng.uniform(math.log10(self.low), math.log10(self.high))
        if self.kind == "cat":
            return rng.choice(self.choices)
        raise ValueError(self.kind)

    def clip(self, value: Any) -> Any:
        """Garante que o gene continua dentro do domínio após a mutação."""
        if self.kind == "int":
            return int(min(max(round(value), self.low), self.high))
        if self.kind in ("float", "log_float"):
            return float(min(max(value, self.low), self.high))
        return value


# ----------------------------------------------------------------------------
# Espaços de busca. Os intervalos foram escolhidos em torno dos valores usados
# na Fase 1 e dos padrões do scikit-learn, com folga para o AG explorar.
# ----------------------------------------------------------------------------
SEARCH_SPACES: dict[str, list[Gene]] = {
    "random_forest": [
        Gene("n_estimators", "int", 50, 400),
        Gene("max_depth", "cat", choices=(None, 3, 5, 8, 12, 20)),
        Gene("min_samples_split", "int", 2, 20),
        Gene("min_samples_leaf", "int", 1, 10),
        Gene("max_features", "cat", choices=("sqrt", "log2", 0.5, None)),
        Gene("criterion", "cat", choices=("gini", "entropy")),
        # class_weight muda o "peso" dos erros na classe minoritária (maligno = 37%).
        Gene("class_weight", "cat", choices=(None, "balanced", "balanced_subsample")),
    ],
    "logistic_regression": [
        # C = inverso da regularização. Varia em ordens de grandeza -> escala log.
        Gene("C", "log_float", 1e-3, 1e3),
        # 0 = L2 (Ridge), 1 = L1 (Lasso), meio-termo = Elastic Net.
        # L1 zera coeficientes -> ajuda com a multicolinearidade vista na EDA da Fase 1.
        Gene("l1_ratio", "float", 0.0, 1.0),
        Gene("class_weight", "cat", choices=(None, "balanced")),
    ],
}


def random_individual(model_name: str, rng: random.Random) -> Individual:
    return {g.name: g.sample(rng) for g in SEARCH_SPACES[model_name]}


def initial_population(model_name: str, size: int, rng: random.Random,
                       seed_individuals: list[Individual] | None = None) -> list[Individual]:
    """INICIALIZAÇÃO (AG Aula 3): aleatória, com a opção de "semear" indivíduos conhecidos.

    Semear o indivíduo da Fase 1 garante que o AG nunca termina PIOR que o original
    (com elitismo, o melhor sempre sobrevive) - uma decisão de projeto discutida no relatório.
    """
    population = [dict(ind) for ind in (seed_individuals or [])][:size]
    while len(population) < size:
        population.append(random_individual(model_name, rng))
    return population


def individual_key(individual: Individual) -> tuple:
    """Chave hashable - usada no cache de fitness e para medir diversidade."""
    def norm(v):
        return round(v, 6) if isinstance(v, float) else v
    return tuple(sorted((k, norm(v)) for k, v in individual.items()))
