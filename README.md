# Tech Challenge - Fase 2 (Pós IA para Devs) · Projeto 1

**Otimização dos modelos de diagnóstico da Fase 1 com algoritmos genéticos + interpretação dos resultados com LLM.**

Este projeto continua a [Fase 1](https://github.com/HenriqueWojcik/tech-challenge-fase1-fiap), que classifica câncer de mama (Breast Cancer Wisconsin) com Regressão Logística e Random Forest. Nesta fase:

1. **Algoritmo genético**, implementado do zero, busca os hiperparâmetros dos dois modelos. São 5 experimentos com configurações diferentes, comparados com o modelo original e com uma busca aleatória de mesmo custo.
2. **LLM** traduz a previsão do modelo e a explicação SHAP em um texto para o médico. Usa prompt engineering, e a qualidade das respostas é avaliada automaticamente.
3. **Logging e monitoramento**, testes automatizados e documentação da arquitetura.

📄 **Relatório técnico:** [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md)
🏗️ **Arquitetura:** [`docs/arquitetura.md`](docs/arquitetura.md)
📚 **Guia de estudo (código ↔ aulas):** [`docs/guia_de_estudo.md`](docs/guia_de_estudo.md)
🎥 **Vídeo:** _link do YouTube aqui_

## Resultados em uma frase

O AG melhorou o F2 em validação cruzada dos dois modelos: Random Forest de 0,952 para 0,970 e Regressão Logística de 0,967 para 0,975. Em validação repetida (50 partições), o modelo otimizado deixa passar **20 a 25% menos cânceres**, com uma pequena perda de precisão. No teste isolado de 114 pacientes, os falsos negativos ficaram iguais, porque a diferença esperada é menor que um paciente. Detalhes e discussão estão no relatório.

## Estrutura

```
src/tc2/
  config.py, data.py, models.py   -> dataset, split e modelos da Fase 1
  ga/space.py                     -> codificação dos genes (hiperparâmetros) e população inicial
  ga/fitness.py                   -> fitness: F2 da classe maligna em validação cruzada (com cache)
  ga/operators.py                 -> seleção (torneio/roleta/elitismo), crossover uniforme, mutação
  ga/engine.py                    -> laço evolutivo e critério de parada
  experiments.py                  -> 5 experimentos + busca aleatória + gráficos + comparação
  explain.py                      -> caso estruturado: previsão + SHAP + médias por classe
  llm/client.py                   -> cliente Groq / OpenAI / Ollama / offline
  llm/prompts.py                  -> prompts (engenheirado x ingênuo, relatório de modelos)
  llm/evaluation.py               -> avaliação automática + LLM-como-juiz
  logging_utils.py                -> logs em console e arquivo
scripts/run_optimization.py       -> roda a otimização
scripts/interpret_patients.py     -> gera e avalia explicações com LLM
notebooks/demo.ipynb              -> demonstração
tests/                            -> 23 testes (pytest)
results/                          -> CSV, JSON, gráficos, modelos e saídas da LLM
docs/                             -> relatório, arquitetura, guia de estudo, roteiro do vídeo
```

## Como executar

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt

# 1) Testes (segundos, sem internet)
pytest -q

# 2) Otimização com AG (~30-40 min completo; --quick ~2 min)
python scripts/run_optimization.py
python scripts/run_optimization.py --quick

# 2b) Checagem de robustez: original x otimizado em validação cruzada repetida 10x5 (~2 min)
python scripts/robustness_check.py

# 3) LLM: copie .env.example para .env e coloque a chave (Groq é gratuita: console.groq.com)
copy .env.example .env           # Linux/Mac: cp .env.example .env
python scripts/interpret_patients.py

# 4) Demonstração
jupyter notebook notebooks/demo.ipynb
```

Sem chave de API, o cliente usa o modo `offline`: o texto é montado por template, o que mantém o sistema e os testes funcionando.

## Requisitos do enunciado → onde estão

| Requisito | Onde |
|---|---|
| Codificação dos genes | `src/tc2/ga/space.py` |
| Seleção, cruzamento e mutação | `src/tc2/ga/operators.py` |
| Fitness baseada em métricas | `src/tc2/ga/fitness.py` |
| Comparação otimizado × original | `results/resumo_experimentos.csv`, relatório seção 3 |
| ≥ 3 experimentos com configurações diferentes | `src/tc2/experiments.py` (E1 a E5) |
| Monitoramento e logging | `src/tc2/logging_utils.py`, `logs/`, histórico por geração em `results/resultados_completos.json` |
| Documentar arquitetura e decisões | `docs/arquitetura.md` |
| LLM pré-treinada: explicações e insights | `src/tc2/llm/`, `scripts/interpret_patients.py`, `results/llm/` |
| Prompt engineering | `src/tc2/llm/prompts.py` |
| Avaliar a qualidade das interpretações | `src/tc2/llm/evaluation.py`, `results/llm/avaliacao_llm.csv` |
| Base para o Módulo 3 | `results/llm/casos_modulo3.jsonl` (casos estruturados) |
| Projeto estruturado + venv + testes | `requirements.txt`, `tests/` |

> Ferramenta acadêmica de apoio à decisão. Não substitui avaliação médica.
