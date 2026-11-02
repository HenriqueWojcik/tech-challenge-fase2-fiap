# Roteiro do vídeo (até 15 min)

O enunciado pede: demonstração do sistema rodando, explicação dos componentes, resultados do AG e demonstração da integração com LLM.

| Tempo | O que mostrar | O que falar |
|---|---|---|
| 0:00–1:00 | README e diagrama de `docs/arquitetura.md` | O problema: o hospital quer melhorar os modelos da Fase 1 e explicá-los aos médicos. Escolhi o Projeto 1 porque ele continua a Fase 1. |
| 1:00–2:30 | `data.py` e `models.py`, e o resultado do baseline | Mesmo split e mesmos modelos da Fase 1, com os números conferindo (RF: 3 falsos negativos; Regressão Logística: 1). |
| 2:30–5:00 | `ga/space.py` e `ga/fitness.py` | Codificação híbrida e por que não binária. Fitness = F2 em validação cruzada: por que F2 e por que não usar o teste. O cache, que é a ideia da tabela de distâncias. |
| 5:00–7:00 | `ga/operators.py` e `ga/engine.py` | Torneio × roleta, elitismo. Por que não OX/Swap como no TSP. Mutação na escala log. Critério de parada. |
| 7:00–8:00 | Terminal: `python scripts/run_optimization.py --quick` | Mostrar os logs geração a geração: melhor, média, diversidade e treinos (o cache reduzindo os treinos). |
| 8:00–10:30 | `results/convergencia_*.png`, `comparacao_abordagens.png`, `matriz_confusao_*.png` e a tabela do relatório | Os 5 experimentos, a comparação com o original e com a busca aleatória. Ser honesto sobre ganhos pequenos e o tamanho do teste. |
| 10:30–13:30 | Terminal: `python scripts/interpret_patients.py`, abrindo um `.md` gerado | Fluxo previsão → SHAP → prompt → LLM. Mostrar o prompt (papel, regras, `<caso>`, formato). Comparar o prompt ingênuo com o engenheirado em `avaliacao_llm.csv`. |
| 13:30–14:30 | `pytest -q` | 23 testes passando. |
| 14:30–15:00 | Conclusão | Limitações e próximos passos: nuvem com autoescala, Módulo 3 com textos e RAG. |

Dicas:
- Rode a otimização completa **antes** de gravar. No vídeo use `--quick`, só para mostrar os logs.
- Deixe os arquivos já abertos em abas, na ordem do roteiro.
