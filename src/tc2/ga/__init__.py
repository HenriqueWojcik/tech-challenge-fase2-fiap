"""Algoritmo genético para otimização de hiperparâmetros.

Fluxo (diagrama das Aulas 2 e 3 de Algoritmos Genéticos):

    codificação -> população inicial -> avaliação (fitness) -> [critério de parada?]
         ^                                                        | não
         +---- nova geração <- mutação <- crossover <- seleção <--+

Cada etapa do diagrama é um arquivo:
- space.py     -> codificação (genes) e inicialização
- fitness.py   -> função de aptidão
- operators.py -> seleção, crossover, mutação
- engine.py    -> o laço evolutivo e o critério de parada
"""
