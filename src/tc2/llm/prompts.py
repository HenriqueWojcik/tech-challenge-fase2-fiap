"""Engenharia de prompt (LLMs Aulas 3 e 4; n8n Aula 3 - system prompt de agentes).

Técnicas usadas no prompt "engenheirado", cada uma com um motivo:
1. PAPEL (role prompting) no system prompt: "assistente de apoio à decisão clínica".
2. REGRAS EXPLÍCITAS DE SEGURANÇA: não dar diagnóstico definitivo, não inventar números,
   lembrar que não substitui o médico (IA Generativa Aula 5 - ética; LLMs Aula 1 - alucinação).
3. CONTEXTO ESTRUTURADO entre delimitadores (<caso>...</caso>) em JSON: a LLM separa dados
   de instruções e tem tudo o que precisa -> "grounding", a mesma ideia do RAG (n8n Aula 6),
   só que o "documento recuperado" é a saída do próprio modelo de ML.
4. FORMATO DE SAÍDA FIXO (seções obrigatórias): facilita a leitura do médico e permite
   avaliar automaticamente (evaluation.py verifica se as seções existem).
5. EXEMPLO (few-shot) de UM fator bem explicado, para fixar o estilo sem copiar conteúdo.
6. PÚBLICO-ALVO e tamanho definidos: médico, linguagem técnica mas direta, até ~300 palavras.

O prompt "ingênuo" existe só para COMPARAÇÃO: mostra no relatório o ganho da engenharia de prompt.
"""
import json

SYSTEM_PROMPT = """Você é um assistente de apoio à decisão clínica em um hospital universitário.
Você explica, para médicos, o resultado de um modelo de machine learning que classifica
nódulos mamários como malignos ou benignos a partir de exames citológicos (FNA).

Regras obrigatórias:
- Use SOMENTE os dados fornecidos dentro de <caso>. Não invente valores, exames ou histórico.
- Todo número que você citar deve aparecer nos dados fornecidos (pode arredondar).
- Nunca afirme um diagnóstico definitivo. O resultado é uma estimativa estatística.
- Deixe claro que a decisão final é do médico e que a ferramenta não substitui avaliação clínica,
  exame histopatológico ou outros exames.
- Se a confiança for baixa, destaque isso e recomende investigação adicional.
- Escreva em português do Brasil, linguagem técnica e objetiva, sem alarmismo."""

FORMAT_INSTRUCTIONS = """Responda exatamente com estas seções em Markdown:

## Resumo
1-2 frases: o que o modelo indicou e com que probabilidade.

## Fatores que mais pesaram
Uma linha por fator (os 3 a 5 principais): característica, valor da paciente comparado às
médias de casos malignos e benignos, e se aumenta ou reduz a chance de malignidade.

## Confiança do resultado
Interprete o nível de confiança e o desempenho do modelo no teste (recall e precisão).

## Próximos passos sugeridos
Sugestões de conduta compatíveis com o resultado, sempre como apoio ao médico.

## Limitações
O que este resultado NÃO considera.

Exemplo de estilo para UM fator (não copie os números):
- **Pontos côncavos (pior valor): 0,19** - acima da média dos casos malignos (0,18) e muito acima
  dos benignos (0,07); é o fator que mais aumenta a chance de malignidade neste caso."""


def build_explanation_prompt(case: dict) -> tuple[str, str]:
    """Prompt engenheirado -> (system, user)."""
    payload = json.dumps(case, ensure_ascii=False, indent=2)
    user = (f"Explique o resultado abaixo para o médico responsável.\n\n"
            f"<caso>\n{payload}\n</caso>\n\n{FORMAT_INSTRUCTIONS}")
    return SYSTEM_PROMPT, user


def build_naive_prompt(case: dict) -> tuple[str, str]:
    """Prompt ingênuo, sem técnica nenhuma - baseline de comparação."""
    return "Você é um assistente útil.", f"Explique este resultado: {json.dumps(case, ensure_ascii=False)}"


MODEL_REPORT_SYSTEM = """Você é um cientista de dados que reporta resultados para a diretoria
clínica de um hospital universitário. Transforme números em conclusões práticas, sem exagerar
ganhos pequenos e sem inventar números que não estejam nos dados. Português do Brasil."""


def build_model_report_prompt(summary_rows: list[dict]) -> tuple[str, str]:
    """Transforma a tabela de experimentos do AG em insights acionáveis (requisito 3)."""
    payload = json.dumps(summary_rows, ensure_ascii=False, indent=2)
    user = f"""Abaixo estão os resultados da otimização de hiperparâmetros por algoritmo genético
dos modelos de detecção de câncer de mama. 'cv_f2' é o F2-score médio em validação cruzada
(recall pesa 2x a precisão); colunas 'teste_*' são medidas em 114 pacientes nunca vistas;
'teste_falsos_negativos' = cânceres que o modelo deixaria passar.

<resultados>
{payload}
</resultados>

Escreva um relatório curto com as seções:
## Principais conclusões (3 a 5 bullets)
## Qual modelo recomendamos e por quê
## O que significa para os pacientes (em falsos negativos evitados ou não)
## Cuidados antes de usar em produção"""
    return MODEL_REPORT_SYSTEM, user


# --------------------------------------------------------------------------- fallback offline
def offline_explanation(case: dict) -> str:
    """Texto montado por template, sem LLM. Serve para testes e quando não há chave de API."""
    lines = ["## Resumo",
             f"O modelo ({case['modelo']}) classificou o caso como **{case['previsao']}**, com "
             f"probabilidade estimada de malignidade de {case['probabilidade_maligno_pct']}%.", "",
             "## Fatores que mais pesaram"]
    for f in case["fatores_principais"]:
        lines.append(f"- **{f['caracteristica'].capitalize()}: {f['valor_paciente']}** - média em casos "
                     f"malignos {f['media_casos_malignos']} e em benignos {f['media_casos_benignos']}; "
                     f"{f['efeito']}.")
    m = case.get("desempenho_do_modelo_no_teste", {})
    lines += ["", "## Confiança do resultado",
              f"Nível de confiança: {case['nivel_confianca']}. No conjunto de teste o modelo teve "
              f"recall de {m.get('recall_maligno', 'n/d')} e precisão de {m.get('precision_maligno', 'n/d')} "
              f"para casos malignos.", "",
              "## Próximos passos sugeridos",
              "- Correlacionar com exame clínico e de imagem.",
              "- Em caso de suspeita ou baixa confiança, considerar biópsia/histopatológico.", "",
              "## Limitações",
              "Ferramenta de apoio à decisão: não substitui a avaliação do médico. Considera apenas "
              "as medidas citológicas do exame, sem histórico clínico, imagem ou fatores de risco."]
    return "\n".join(lines)
