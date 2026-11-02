# Guia de estudo: cada parte do código ligada à aula de onde vem

Use este guia para entender o projeto na ordem em que ele foi construído. Cada etapa traz:
o arquivo, o conceito da aula, a decisão tomada aqui e uma pergunta para você conseguir
responder no vídeo ou numa arguição.

---

## Etapa 0: Estrutura do projeto
**Arquivos:** `requirements.txt`, `pyproject.toml`, `.env.example`, `src/tc2/`, `tests/`
**Origem:** requisito 4 do enunciado (projeto estruturado, ambiente virtual, testes).

- O código fica em `src/tc2` como um **pacote**, e não espalhado em notebooks. Assim ele pode ser importado pelos scripts, pelos testes e, no Módulo 3, por uma API.
- `.env` guarda a chave da LLM fora do código e está no `.gitignore`, para nunca subir para o GitHub.

> Pergunta: por que o código saiu do notebook nesta fase?
> Porque testes automatizados e reaproveitamento exigem funções importáveis. O notebook vira só a demonstração.

---

## Etapa 1: Reproduzir o baseline da Fase 1
**Arquivos:** `data.py`, `models.py`
**Origem:** seu notebook `03_pipeline_modelagem.ipynb`; AG Aula 1 (ótimo × subótimo)

- Mesmo split (80/20, `stratify=y`, `random_state=42`) e mesmos modelos. Os números conferem com a Fase 1: RF com recall 0,93 e 3 falsos negativos, Regressão Logística com recall 0,98 e 1 falso negativo.
- **AG Aula 1:** testar todas as combinações de hiperparâmetros é inviável (explosão combinatória, como no TSP). Por isso usamos uma **heurística**, que aceita uma solução "boa o suficiente" a um custo razoável.

> Pergunta: quantas combinações o espaço da Random Forest tem?
> Faça a conta: 351 × 6 × 19 × 10 × 4 × 2 × 3 ≈ 9,6 milhões. A ~1 s por treino, seriam quase 4 meses. O AG usa cerca de uma centena de treinos.

---

## Etapa 2: Codificação dos indivíduos
**Arquivo:** `ga/space.py`
**Origem:** AG Aula 3 (codificação binária, real, combinatória e híbrida); AG Aula 2 (gene, cromossomo, indivíduo)

| Biologia (AG Aula 2) | No TSP (AG Aulas 4 e 5) | Neste projeto |
|---|---|---|
| Gene | Uma cidade numa posição | Um hiperparâmetro (`max_depth`) |
| Cromossomo/indivíduo | Uma rota (permutação) | Um conjunto de hiperparâmetros |
| População | Várias rotas | Vários modelos candidatos |
| Aptidão | 1 / distância | F2-score |

- Escolhemos codificação **híbrida**: `int` (n_estimators), `log_float` (C), `float` (l1_ratio) e `cat` (criterion).
- `log_float`: C=0,001 e C=1000 diferem em 6 **ordens de grandeza**. Sortear no expoente dá chance igual a cada ordem.

> Pergunta: por que não codificação binária?
> Categóricos como `"sqrt"` não têm representação natural em bits, e um bit trocado pode gerar um valor inválido ou um salto enorme.

---

## Etapa 3: Função de aptidão
**Arquivo:** `ga/fitness.py`
**Origem:** AG Aula 3 (fitness); AG Aula 4 (tabela de distâncias = cache); Fase 1 (métricas e limitação do split único)

- Fitness = **F2 da classe maligna**, média de 5 folds estratificados, **somente no treino**.
- Por que F2? Recall puro é "trapaceável" (chamar todos de maligno dá recall 1,0). O F2 dá ao recall o dobro do peso da precisão.
- Por que não usar o teste? Se o AG olhasse o teste, ele deixaria de ser uma avaliação independente.
- **Cache:** é o mesmo raciocínio da tabela de distâncias da Aula 4, gastar memória para não recalcular.

> Pergunta: qual a diferença entre fitness e métrica de teste?
> A fitness guia a busca (validação cruzada no treino). A métrica de teste só é olhada no fim, para comparar.

---

## Etapa 4: Seleção
**Arquivo:** `ga/operators.py` (funções `tournament_selection`, `roulette_selection`, `elite`)
**Origem:** AG Aula 4 (seleção dos 10 primeiros + seleção probabilística com `random.choices`)

- **Torneio (k=3):** sorteia 3 indivíduos e fica com o melhor. O `k` regula a pressão seletiva.
- **Roleta:** é o `random.choices(weights=...)` da Aula 4. Detalhe importante: como todas as fitness ficam perto de 0,95, subtraímos o mínimo. Sem isso, todos teriam ~a mesma chance.
- **Elitismo:** os 2 melhores passam direto. É o equivalente a "nunca perder os 10 primeiros" da aula.

---

## Etapa 5: Crossover, mutação e parada
**Arquivos:** `ga/operators.py`, `ga/engine.py`
**Origem:** AG Aula 5 (OX, Swap, "o crossover depende do indivíduo", condição de término)

- A aula usa **Ordered Crossover** e **Swap** porque uma rota precisa ter cada cidade exatamente uma vez.
- Aqui os genes são independentes, então usamos:
  - **crossover uniforme**: cada gene vem de um dos pais;
  - **mutação por gene**: gaussiana para números (na escala log para C) e sorteio de outra opção para categóricos.
- **Critério de parada:** um máximo de gerações **ou** `patience` gerações sem melhora.

> Pergunta: o que aconteceria se usássemos OX aqui?
> Nada de útil. O OX preserva a *ordem relativa*, e ordem não significa nada num dicionário de hiperparâmetros.

---

## Etapa 6: Experimentos e comparação
**Arquivo:** `experiments.py`
**Origem:** AG Aula 3 (parâmetros, convergência × diversidade, métricas de desempenho); ML na Cloud Aula 3 (hyperparameter tuning)

| Experimento | O que muda | Hipótese (convergência × diversidade) |
|---|---|---|
| E1 base | referência | equilíbrio |
| E2 mutação alta | 40% de mutação, passo maior | mais diversidade, convergência mais lenta |
| E3 população maior | 28 indivíduos, 7 gerações | explora mais no início, refina menos |
| E4 roleta sem elitismo | menor pressão seletiva | pode "perder" o melhor; média oscila |
| E5 semeado | modelo da Fase 1 na população inicial | nunca termina pior que o original |
| Busca aleatória | sem evolução, mesmo nº de treinos | controle: o AG precisa vencê-la para se justificar |

Veja os gráficos `results/convergencia_*.png`. A curva **tracejada** (média) subindo e a **diversidade** caindo mostram a população convergindo.

---

## Etapa 7: Logging, monitoramento e arquitetura
**Arquivos:** `logging_utils.py`, `docs/arquitetura.md`
**Origem:** ML na Cloud Aula 1 (ciclo de vida e limites do ambiente local), Aula 2 (Azure ML, deploy), Aula 3 (paralelização)

- Cada geração registra melhor, média, diversidade, treinos novos e tempo.
- O `n_jobs=-1` paraleliza a validação cruzada. A versão em nuvem (cluster com autoescala) está descrita na arquitetura como trabalho futuro e pontuação extra.

---

## Etapa 8: LLM para interpretar os resultados
**Arquivos:** `explain.py`, `llm/client.py`, `llm/prompts.py`, `llm/evaluation.py`
**Origem:**
- LLMs Aula 1: Transformers, **alucinação** e viés.
- LLMs Aulas 3 e 4: **engenharia de prompt**, RAG, fine-tuning.
- LLMs Aula 5: on-premises × nuvem × **API**.
- n8n Aula 3: system prompt de agentes.
- n8n Aula 4: Groq e troca de provedor.
- n8n Aula 6: RAG e *grounding*.
- IA Generativa Aula 5: **ética**.

Fluxo:

```
paciente -> modelo otimizado -> probabilidade + SHAP + médias das classes -> JSON do caso
         -> prompt (papel + regras + dados entre <caso> + formato + exemplo) -> LLM -> texto
         -> avaliação (estrutura, classe, cobertura, fidelidade numérica, segurança, tamanho)
```

- **Por que não fine-tuning?** Não há textos médicos rotulados, e o custo é alto (LLMs Aula 4).
- **Por que não RAG completo?** O conhecimento necessário é a própria saída do modelo, que vai inteira no prompt. É um "RAG de um documento só". No Módulo 3, com laudos e textos, um RAG de verdade (n8n Aula 6) entra aqui.
- **Fidelidade numérica:** todo número do texto precisa existir no JSON do caso. É uma forma objetiva de medir alucinação.
- **Prompt ingênuo × engenheirado:** o relatório compara os dois, e é isso que mostra o efeito da engenharia de prompt.

---

## Etapa 9: Testes
**Pasta:** `tests/`. São 23 testes que rodam em segundos, sem internet:

- **Codificação e operadores:** genes sempre dentro do domínio; filho só herda genes dos pais; elitismo preserva o melhor.
- **Motor:** com uma fitness de brinquedo de ótimo conhecido, o AG chega perto do ótimo, é reprodutível com a mesma semente e para por estagnação.
- **Fitness:** valor entre 0 e 1 e cache funcionando; split idêntico ao da Fase 1.
- **LLM:** fallback offline, o gabarito nunca vai para a LLM, e a avaliação detecta número inventado e texto inseguro.
