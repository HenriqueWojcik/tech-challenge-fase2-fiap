# Relatório técnico - Tech Challenge Fase 2 · Projeto 1

**Otimização de modelos de diagnóstico com algoritmos genéticos e interpretação com LLMs**
Aluno: Henrique Wojcik Vieira · Pós IA para Devs (FIAP + Alura)

---

## 1. Contexto e objetivo

Na Fase 1 foram treinados dois classificadores de câncer de mama sobre o dataset Breast Cancer Wisconsin (569 pacientes, 30 medidas citológicas): uma **Regressão Logística** (recall 0,98 na classe maligna, 1 falso negativo no teste) e uma **Random Forest** (recall 0,93, 3 falsos negativos). Os dois usavam hiperparâmetros padrão.

Esta fase tem dois objetivos:
1. Otimizar os hiperparâmetros dos dois modelos com um **algoritmo genético (AG)** implementado do zero e comparar o resultado com os originais.
2. Usar uma **LLM pré-treinada** para transformar a saída numérica do modelo (probabilidade + SHAP) em uma explicação útil para o médico, com prompt engineering e avaliação da qualidade.

Para que a comparação seja justa, o dataset, o split (80/20 estratificado, `random_state=42`) e o pipeline (`StandardScaler` + classificador) são **idênticos** aos da Fase 1. O baseline foi reproduzido com os mesmos números.

---

## 2. Implementação do algoritmo genético

### 2.1 Codificação (`ga/space.py`)
Cada indivíduo é um dicionário de hiperparâmetros, e cada hiperparâmetro é um gene. Como os genes têm tipos diferentes, a codificação é **híbrida**:

| Modelo | Gene | Tipo | Domínio |
|---|---|---|---|
| Random Forest | `n_estimators` | inteiro | 50 a 400 |
| | `max_depth` | categórico | None, 3, 5, 8, 12, 20 |
| | `min_samples_split` | inteiro | 2 a 20 |
| | `min_samples_leaf` | inteiro | 1 a 10 |
| | `max_features` | categórico | sqrt, log2, 0,5, None |
| | `criterion` | categórico | gini, entropy |
| | `class_weight` | categórico | None, balanced, balanced_subsample |
| Regressão Logística | `C` | real (escala log) | 0,001 a 1000 |
| | `l1_ratio` | real | 0 (L2) a 1 (L1) |
| | `class_weight` | categórico | None, balanced |

A codificação binária foi descartada: genes categóricos não têm representação natural em bits, e trocar um bit pode gerar valores inválidos ou saltos grandes. O espaço da Random Forest tem cerca de 9,6 milhões de combinações. A ~1 s por treino, testar todas levaria meses, o que justifica uma heurística.

### 2.2 Função fitness (`ga/fitness.py`)
**Fitness = F2-score da classe maligna, média de validação cruzada estratificada de 5 folds, calculada apenas no conjunto de treino.**

- **F2 em vez de recall:** recall puro é maximizado trivialmente classificando todos como malignos. O F2 dá ao recall o dobro do peso da precisão. Isso reflete o custo clínico maior do falso negativo (câncer não detectado) em relação ao falso positivo (exame adicional), sem ignorar o segundo.
- **Validação cruzada em vez do teste:** otimizar no teste o contaminaria. A média de 5 folds também responde à limitação apontada na Fase 1 (dependência de um único split).
- **Cache:** indivíduos repetidos, que são frequentes por causa do elitismo e do crossover, não são re-treinados.

### 2.3 Operadores (`ga/operators.py`)
- **Seleção:** torneio (k=3) ou roleta (`random.choices` com pesos). Na roleta, a fitness é deslocada pelo mínimo da geração. Sem isso, como todas as fitness ficam entre 0,94 e 0,97, todos teriam praticamente a mesma probabilidade.
- **Elitismo:** os 2 melhores passam intactos para a próxima geração.
- **Crossover uniforme:** cada gene do filho vem de um dos pais, com 50% de chance. Os operadores das aulas (Ordered Crossover e Swap) foram feitos para permutações, que é a representação do TSP. Aqui os genes são independentes, e a ordem não tem significado.
- **Mutação por gene:** cada gene muta com probabilidade igual à taxa de mutação. Inteiros e reais recebem um ruído gaussiano proporcional ao domínio; `C` é mutado no expoente (escala log); categóricos sorteiam outra opção.
- **Critério de parada:** número máximo de gerações **ou** 6 gerações sem melhora.

### 2.4 Experimentos (`experiments.py`)
Foram feitos cinco experimentos e um controle. Cada um muda uma ideia em relação ao E1:

| | Configuração | Hipótese |
|---|---|---|
| E1 base | pop. 16, 12 gerações, torneio k=3, elitismo 2, mutação 15% | referência |
| E2 mutação alta | mutação 40%, passo 2× maior | mais diversidade, convergência mais lenta |
| E3 população maior | pop. 28, 7 gerações (orçamento semelhante) | mais exploração inicial, menos refinamento |
| E4 roleta sem elitismo | roleta, elitismo 0 | menor pressão seletiva; risco de perder o melhor |
| E5 semeado | E1 + o indivíduo da Fase 1 na população inicial | nunca termina pior que o original |
| Busca aleatória | sorteio puro, mesmo nº de treinos do E1 | controle: o AG precisa vencê-la |

---

## 3. Resultados

### 3.1 Tabela geral
*F2 em validação cruzada (fitness) e métricas no teste: 114 pacientes nunca vistas, sendo 42 com tumor maligno.*

**Random Forest**

| Abordagem | Treinos | F2 CV (± desvio) | Recall teste | Precisão teste | F2 teste | Falsos negativos |
|---|---|---|---|---|---|---|
| Original (Fase 1) | 1 | 0,952 ± 0,024 | 0,929 | 0,951 | 0,933 | 3 |
| E1 base | 92 | 0,963 ± 0,021 | 0,929 | 0,907 | 0,924 | 3 |
| E2 mutação alta | 138 | 0,963 ± **0,010** | **0,952** | 0,909 | **0,943** | **2** |
| E3 população maior | 160 | 0,965 ± 0,021 | 0,929 | 0,886 | 0,920 | 3 |
| E4 roleta sem elitismo | 130 | 0,960 ± 0,016 | 0,929 | 0,907 | 0,924 | 3 |
| **E5 semeado** (melhor em CV) | 132 | **0,970** ± 0,022 | 0,929 | 0,886 | 0,920 | 3 |
| Busca aleatória | 92 | 0,959 ± 0,018 | 0,952 | 0,870 | 0,935 | 2 |

Melhor indivíduo (E5): `n_estimators=184, max_depth=12, min_samples_split=2, min_samples_leaf=1, max_features=log2, criterion=gini, class_weight=balanced`

**Regressão Logística**

| Abordagem | Treinos | F2 CV (± desvio) | Recall teste | Precisão teste | F2 teste | Falsos negativos |
|---|---|---|---|---|---|---|
| Original (Fase 1) | 1 | 0,967 ± 0,022 | 0,976 | 0,976 | 0,976 | 1 |
| **E1 base** (melhor em CV) | 53 | **0,975** ± **0,011** | 0,976 | 0,911 | 0,962 | 1 |
| E2 mutação alta | 87 | 0,975 ± 0,011 | 0,976 | 0,911 | 0,962 | 1 |
| E3 população maior | 92 | 0,975 ± 0,011 | 0,976 | 0,911 | 0,962 | 1 |
| E4 roleta sem elitismo | 79 | 0,970 ± 0,009 | 0,976 | 0,976 | 0,976 | 1 |
| E5 semeado | 52 | 0,974 ± 0,012 | 0,976 | 0,911 | 0,962 | 1 |
| Busca aleatória | 53 | 0,975 ± 0,011 | 0,976 | 0,911 | 0,962 | 1 |

Melhor indivíduo (E1): `C=0,47, l1_ratio=0 (L2), class_weight=balanced`

Gráficos: `results/convergencia_random_forest.png`, `results/convergencia_logistic_regression.png`, `results/comparacao_abordagens.png`, `results/matriz_confusao_*.png`.

### 3.2 Checagem de robustez (validação cruzada repetida 10 × 5)
O teste tem só 42 casos malignos, então **um único paciente muda o recall em 2,4 pontos**. Para separar ganho real de variação do split, os hiperparâmetros originais e os otimizados foram comparados em 50 partições diferentes (outra semente, sobre as 569 pacientes). O script é `scripts/robustness_check.py`.

| Modelo | Versão | F2 | Recall | Precisão | Falsos negativos médios por fold (~42 malignos) |
|---|---|---|---|---|---|
| Random Forest | Original | 0,939 | 0,935 | 0,957 | 2,75 |
| Random Forest | Otimizado (E5) | **0,949** | **0,951** | 0,942 | **2,07 (−25%)** |
| Regressão Logística | Original | 0,962 | 0,958 | 0,977 | 1,78 |
| Regressão Logística | Otimizado (E1) | **0,967** | **0,967** | 0,968 | **1,40 (−21%)** |

### 3.3 Discussão

**O AG melhorou os modelos no critério que foi pedido a ele.** Nos dois modelos, a fitness em validação cruzada subiu: de 0,952 para 0,970 na Random Forest e de 0,967 para 0,975 na Regressão Logística. Na validação repetida, com partições que o AG nunca viu, o otimizado deixou passar cerca de **20 a 25% menos cânceres**, ao custo de uma queda de 1 a 1,5 ponto na precisão.

**O que o AG "descobriu":** quase todos os melhores indivíduos escolheram `class_weight="balanced"`. A classe maligna é minoritária (37%), e dar mais peso a ela aumenta o recall, exatamente o que a fitness F2 recompensa. Na Random Forest, o AG também preferiu `max_features=log2`, que gera árvores mais diversas. Na Regressão Logística, preferiu `C≈0,3–0,5` (mais regularização que o padrão C=1) e L2 puro (`l1_ratio≈0`).

**Por que o teste isolado não mostra o ganho:** no conjunto de teste os falsos negativos ficaram iguais (3 e 1), e a precisão caiu: a Random Forest passou de 2 para 5 falsos positivos, e a Regressão Logística de 1 para 4. Com 42 malignos, a diferença esperada (~0,5 a 0,7 caso) é menor que a granularidade do teste. O paciente que a Regressão Logística erra (id 73) continua errado no otimizado, com probabilidade de malignidade de 17%. É um caso difícil (textura abaixo da média dos benignos), e os hiperparâmetros sozinhos não o resolvem. **Conclusão honesta:** o ganho existe e é consistente em 50 partições, mas é pequeno. Ele troca alguns falsos positivos por menos falsos negativos, que é a troca desejada no contexto clínico. Essa troca deve ser apresentada ao corpo médico como uma decisão, e não como um "ganho grátis".

**Comparação entre experimentos (convergência × diversidade):**
- **E4 (roleta sem elitismo)** foi o pior AG nos dois modelos. No gráfico de convergência, a curva do melhor **cai** em algumas gerações: sem elitismo, o melhor indivíduo pode não ser selecionado e se perde. A diversidade também ficou mais alta (0,75 contra ~0,4 na Regressão Logística). É a demonstração prática de por que o elitismo importa.
- **E2 (mutação alta)** gastou mais treinos (138 contra 92) pelo mesmo melhor resultado. Em compensação, encontrou a solução de **menor variância** na Random Forest (desvio de 0,010), que também foi a única com 2 falsos negativos no teste.
- **E3 (população maior)** foi o segundo melhor na Random Forest (0,965) e parou pelo limite de 7 gerações, não por estagnação. A população maior compensou o menor número de gerações.
- **E5 (semeado com a Fase 1)** deu o melhor resultado na Random Forest e continuou melhorando até a geração 10. Partir de uma solução conhecida e boa acelerou a busca.
- **E1** estagnou cedo na Random Forest (melhor encontrado na geração 1), um sinal de pressão seletiva alta para uma população de 16.

**AG × busca aleatória:** com o mesmo orçamento (92 treinos), o AG (E1) superou a busca aleatória na Random Forest (0,963 contra 0,959 em CV), e o E5 chegou a 0,970. Na Regressão Logística houve **empate**: com apenas 3 genes, sortear 53 pontos já cobre bem o espaço. O AG compensa em espaços maiores e combinatórios, como o da Random Forest, com ~10 milhões de combinações. Em espaços pequenos, uma busca simples basta. É uma conclusão útil para decidir onde usar AG no Módulo 3.

---

## 4. Integração com LLM

### 4.1 Abordagem
```
paciente → modelo otimizado → probabilidade de malignidade
        → SHAP (5 fatores mais influentes) + médias dos casos malignos/benignos
        → JSON do caso → prompt → LLM → explicação para o médico → avaliação automática
```
A LLM **não decide nada**. O diagnóstico vem do modelo validado, e a LLM só traduz números em texto. Toda a informação necessária vai no prompt (*grounding*, a mesma ideia do RAG), o que reduz o espaço para alucinação.

O cliente (`llm/client.py`) usa a API compatível com OpenAI, então funciona com:
- **Groq** (gratuito, usado nos resultados). O padrão do código é o Llama 3.3 70B, mas ele não estava mais disponível na conta. Os resultados da seção 4.4 usaram `openai/gpt-oss-20b`, definido por `LLM_MODEL` no `.env`;
- **OpenAI**;
- **Ollama**, que roda localmente: nenhum dado sai do hospital (LGPD).

Sem chave, entra o modo **offline** por template, usado nos testes.

### 4.2 Prompt engineering (`llm/prompts.py`)
| Técnica | Como foi aplicada |
|---|---|
| Papel (*role prompting*) | "Assistente de apoio à decisão clínica, explicando para médicos." |
| Regras de segurança | Usar só os dados de `<caso>`; não inventar números; nunca dar diagnóstico definitivo; lembrar que não substitui o médico; destacar baixa confiança. |
| Dados estruturados com delimitadores | JSON dentro de `<caso>...</caso>`, separando dados de instruções. |
| Formato de saída fixo | 5 seções: Resumo, Fatores, Confiança, Próximos passos, Limitações. |
| Exemplo (*few-shot*) | Um fator explicado no estilo desejado, com aviso para não copiar os números. |
| Temperatura baixa (0,2) | Respostas consistentes; criatividade não é desejável aqui. |

Um **prompt ingênuo** ("Explique este resultado: {json}") é gerado para os mesmos casos, como linha de base.

Um segundo prompt (`build_model_report_prompt`) transforma a **tabela de experimentos** em um relatório para a diretoria clínica. Ele atende o item "transformar dados numéricos e estatísticos em insights acionáveis".

### 4.3 Avaliação da qualidade (`llm/evaluation.py`)
São seis métricas automáticas e reprodutíveis, mais uma avaliação opcional por LLM-juiz:

| Métrica | O que verifica |
|---|---|
| Estrutura | As 5 seções pedidas estão presentes. |
| Classe correta | O texto cita a classe prevista pelo modelo. |
| Cobertura de fatores | Os 3 fatores SHAP mais importantes são mencionados. |
| **Fidelidade numérica** | Todo número no texto existe nos dados do caso (tolerância de 1,5%). **É o detector de alucinação.** |
| Segurança | Há aviso de "não substitui o médico" e nenhuma afirmação definitiva ("com certeza é câncer"). |
| Tamanho | Entre 100 e 450 palavras. |
| LLM-juiz (opcional) | Notas de 1 a 5 para clareza, utilidade clínica, fidelidade e tom. |

Casos avaliados: o falso negativo que o modelo original deixava passar, o caso mais claramente maligno, o mais claramente benigno e o mais incerto (probabilidade mais próxima de 50%).

### 4.4 Resultados da LLM
Execução: `python scripts/interpret_patients.py`, com o provedor Groq e o modelo `openai/gpt-oss-20b`. O modelo classificador escolhido automaticamente foi a `logistic_regression` otimizada. Foram avaliados 4 pacientes do teste (73, 461, 296 e 526), cada um com os dois prompts, num total de 8 explicações. Os dados por caso estão em `results/llm/avaliacao_llm.csv`.

**Média das métricas automáticas por tipo de prompt**

| prompt | estrutura | classe_correta | cobertura_fatores | fidelidade_numerica | seguranca | tamanho_adequado | nota_geral |
|---|---|---|---|---|---|---|---|
| engenheirado | 1,0 | 1,0 | 1,0 | 1,000 | 1,00 | 1,0 | 1,00 |
| ingenuo | 0,8 | 1,0 | 1,0 | 0,892 | 0,75 | 0,0 | 0,74 |

**Média das notas do LLM-juiz (1 a 5)**

| prompt | clareza | utilidade clínica | fidelidade aos dados | tom adequado |
|---|---|---|---|---|
| engenheirado (4 casos) | 4,75 | 4,00 | 4,50 | 5,00 |
| ingenuo (3 casos*) | 4,33 | 4,33 | 5,00 | 4,00 |

\* No caso 526 com prompt ingênuo, o juiz não devolveu notas (campos vazios no CSV). A média do ingênuo usa só 3 casos.

**Trecho de `results/llm/caso_73_engenheirado.md`** (paciente maligno que o modelo classifica como benigno, com probabilidade de malignidade de 0,1693):

> ## Resumo
> O modelo de regressão logística, otimizado por algoritmo genético, indicou que o nódulo é **benigno** com uma probabilidade de malignidade de **16,9 %** (0,1693).
>
> ## Confiança do resultado
> O nível de confiança é **moderado**. [...] Esses indicadores sugerem que o modelo é robusto, mas a confiança moderada indica que fatores não capturados pelo modelo podem influenciar o resultado.
>
> ## Próximos passos sugeridos
> [...] 3. **Biópsia confirmatória** – se houver suspeita clínica ou radiológica, realizar biópsia por agulha fina ou núcleo.

*Comentário:* a LLM tratou a incerteza só em parte. Ela repetiu o nível de confiança "moderado" e a probabilidade de 16,9%. Também citou o único fator que aumenta o risco (pontos côncavos, pior valor: 0,1383, acima da média dos benignos, 0,0744) e recomendou avaliação clínica e biópsia condicional. Por outro lado, chamou o modelo de "robusto" e não disse que uma chance de malignidade de cerca de 17% merece atenção. Este paciente era de fato **maligno**, informação que não vai para a LLM. É o tipo de caso em que uma regra explícita no prompt ajudaria, por exemplo: "se a probabilidade de malignidade estiver acima de X%, recomende investigação adicional". Isso fica registrado como melhoria futura.

**Prompt ingênuo × engenheirado**
- **Formato e tamanho:** o engenheirado seguiu as 5 seções em todos os casos (estrutura 1,0) e ficou entre 286 e 355 palavras. O ingênuo produziu de 694 a 998 palavras, com tabelas e emojis. Ficou fora da faixa em todos os casos (tamanho_adequado 0,0) e nem sempre trouxe as seções pedidas (estrutura 0,8).
- **Segurança:** o engenheirado teve 1,00. O ingênuo teve 0,75, porque no caso 461 não incluiu o aviso exigido (seguranca 0,0). O LLM-juiz apontou ainda os emojis do ingênuo como "ligeiramente informais" (tom 4,0 contra 5,0).
- **Números sem suporte (`numeros_sem_suporte`):** o engenheirado não teve nenhum, em nenhum caso. O ingênuo teve em todos os casos: 73 → `[0.0, 0.7444, 0.4144, 0.2968, 83.1, 100.0, 100.0, 30.0, 83.1]`; 461 → `[0.0, 30.0]`; 296 → `[0.5, 1.1, 1.706, 0.9941, 0.8283, 0.8074, 5.33, 30.0, 10.0]`; 526 → `[0.5305, 30.0, 70.0]`. Nem todos são alucinações. No caso 73, 0,7444, 0,4144 e 0,2968 são os impactos SHAP do próprio caso, que são negativos. O texto escreveu o sinal com um hífen não padrão ("‑0,7444"), e o verificador leu os valores como positivos. 83,1 e 100 vêm da conta 100% − 16,9%, e 30 corresponde aos "30 atributos" do dataset, informação que não está no JSON. Isso mostra duas coisas: o prompt ingênuo faz a LLM trazer conhecimento e contas fora do caso, e a métrica automática de fidelidade é conservadora e pode gerar falsos positivos com a formatação.
- **LLM-juiz × métricas automáticas:** o juiz deu fidelidade 5,0 ao ingênuo e 4,5 ao engenheirado. A média do engenheirado caiu por causa do caso 526 (nota 3), em que o juiz viu "inconsistências nos valores comparativos (ex.: pontos côncavos e compacidade)". Ali todos os números existiam no caso (fidelidade_numerica 1,0), mas as comparações com as médias estavam erradas. As duas avaliações se complementam: a automática pega números inventados, e o juiz pega afirmações erradas feitas com números corretos. Com 4 casos e um único juiz que é o mesmo modelo gerador, as notas do juiz são indicativas, não conclusivas.

Em resumo, o prompt engenheirado ganhou em todas as métricas automáticas (nota geral 1,00 contra 0,74). O ganho vem de formato, concisão, segurança e ausência de números fora do caso, e não da classe prevista: os dois prompts acertaram a classe em 100% dos casos.

### 4.5 Base para o Módulo 3
Cada caso interpretado é salvo em `results/llm/casos_modulo3.jsonl`, um formato estruturado com previsão, fatores e classe real. Esse arquivo pode ser indexado numa base vetorial (RAG) e combinado com laudos textuais no assistente médico do Módulo 3.

---

## 5. Monitoramento, logging e arquitetura
- Cada geração registra melhor, média e pior fitness, diversidade, treinos novos (fora do cache) e tempo, no console e em `logs/*.log`. O histórico completo fica em `results/resultados_completos.json`.
- Escalabilidade local: a validação cruzada paraleliza os folds em todos os núcleos (`n_jobs=-1`), e o cache reduz os treinos conforme a população converge.
- A arquitetura, a tabela de decisões e o caminho para a nuvem (Azure ML com cluster de autoescala, sweep jobs e endpoint) estão em `docs/arquitetura.md`. A implementação em nuvem não foi feita (é opcional no enunciado).

---

## 6. Desafios e soluções

| Desafio | Solução |
|---|---|
| Treinar uma Random Forest com CV custa ~1 s, e o AG precisa de centenas de avaliações | Cache de fitness + folds em paralelo + parada por estagnação |
| OX e Swap (aulas de TSP) não servem para genes independentes | Crossover uniforme e mutação por tipo de gene |
| `C` varia em 6 ordens de grandeza | Amostragem e mutação na escala logarítmica |
| Roleta sem efeito com fitness muito próximas | Deslocar as fitness pelo mínimo da geração |
| `penalty` da LogisticRegression foi descontinuado no scikit-learn 1.8 | Gene `l1_ratio` (0 = L2, 1 = L1), com solver escolhido automaticamente |
| Teste pequeno demais para medir ganhos pequenos | Seleção pela CV + checagem com validação cruzada repetida 10×5 |
| Risco de alucinação da LLM em contexto médico | Grounding (só os dados do caso), regras explícitas e métrica automática de fidelidade numérica |
| Testes não podem depender de internet nem de chave | Modo offline do cliente LLM |

---

## 7. Limitações e próximos passos
- **Dataset pequeno** (569 pacientes): diferenças de 1 a 2 pontos estão perto do ruído. A validação repetida mitiga, mas não elimina.
- **Uma semente por experimento:** AGs são estocásticos. O ideal seria repetir cada configuração com várias sementes e comparar distribuições.
- **Limiar de decisão fixo em 0,5:** incluir o limiar como gene permitiria ao AG escolher explicitamente o ponto da troca entre recall e precisão. O caso id 73 (17% de probabilidade, maligno) mostra o valor disso.
- **Nuvem:** executar a avaliação da população em paralelo num cluster com autoescala.
- **LLM:** validar as explicações com médicos reais. As métricas automáticas medem forma e fidelidade, não utilidade clínica.

> Ferramenta acadêmica de apoio à decisão. Não substitui avaliação médica, exames complementares ou histopatológico.
