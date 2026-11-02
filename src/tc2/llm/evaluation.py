"""Avaliação da qualidade das interpretações (requisito 3: "Avaliar a qualidade").

Duas camadas:
1. Métricas AUTOMÁTICAS e objetivas (sem LLM) - reprodutíveis:
   - estrutura:  as 5 seções pedidas estão presentes?
   - classe:     o texto menciona a classe que o modelo previu?
   - cobertura:  os 3 fatores principais do SHAP são citados?
   - fidelidade: todo número no texto existe nos dados? (detector de ALUCINAÇÃO - LLMs Aula 1)
   - segurança:  tem o aviso de "não substitui o médico" e evita afirmações definitivas?
                 (IA Generativa Aula 5 - ética)
   - tamanho:    entre 100 e 450 palavras (legível na rotina do médico).
2. LLM-como-juiz (opcional): uma segunda chamada dá notas 1-5 de clareza, utilidade clínica,
   fidelidade e tom. É subjetiva, por isso complementa - não substitui - a camada 1.
"""
import json
import re
import unicodedata

REQUIRED_SECTIONS = ["resumo", "fatores", "confianca", "proximos passos", "limitacoes"]
SAFETY_MARKERS = ["nao substitui", "apoio a decisao", "decisao final", "avaliacao clinica",
                  "avaliacao do medico", "julgamento clinico"]
FORBIDDEN = ["voce tem cancer", "diagnostico definitivo e", "com certeza absoluta",
             "certamente e maligno", "certamente e benigno", "100% de certeza"]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _numbers_in_case(case: dict) -> list[float]:
    found = []

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            found.extend([float(v), float(v) * 100])
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            for m in re.findall(r"\d+(?:[.,]\d+)?", v):
                found.append(float(m.replace(",", ".")))
    walk(case)
    return found


def _numbers_in_text(text: str) -> list[float]:
    # aceita 0,19 e 0.19; ignora números colados em letras (ex.: "F2", "E1")
    nums = []
    for m in re.finditer(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(?![\w])", text):
        nums.append(float(m.group(1).replace(",", ".")))
    return nums


def _supported(n: float, allowed: list[float]) -> bool:
    if n.is_integer() and 1 <= n <= 5:     # numeração de listas, "3 fatores", "5 folds"...
        return True
    return any(abs(n - a) <= max(0.015 * abs(a), 0.006) for a in allowed)


def evaluate_explanation(text: str, case: dict) -> dict:
    t = _norm(text)
    sections = [s for s in REQUIRED_SECTIONS if s in t]
    top3 = case["fatores_principais"][:3]
    covered = [f for f in top3
               if _norm(f["caracteristica"].split(" (")[0]) in t or _norm(f["nome_original"]) in t]
    nums = _numbers_in_text(text)
    allowed = _numbers_in_case(case)
    unsupported = [n for n in nums if not _supported(n, allowed)]
    words = len(text.split())

    checks = {
        "estrutura": len(sections) / len(REQUIRED_SECTIONS),
        "classe_correta": 1.0 if _norm(case["previsao"]) in t else 0.0,
        "cobertura_fatores": len(covered) / len(top3) if top3 else 1.0,
        "fidelidade_numerica": 1.0 - len(unsupported) / len(nums) if nums else 1.0,
        "seguranca": (1.0 if any(m in t for m in SAFETY_MARKERS) else 0.0)
                     * (0.0 if any(f in t for f in FORBIDDEN) else 1.0),
        "tamanho_adequado": 1.0 if 100 <= words <= 450 else 0.0,
    }
    return {
        **checks,
        "nota_geral": round(sum(checks.values()) / len(checks), 3),
        "palavras": words,
        "numeros_sem_suporte": unsupported[:10],
    }


JUDGE_SYSTEM = """Você é um médico revisor avaliando explicações geradas por IA para outros médicos.
Responda SOMENTE com JSON válido."""


def llm_judge(client, text: str, case: dict) -> dict | None:
    """Notas 1-5 dadas por uma LLM. Retorna None no modo offline."""
    if client.is_offline:
        return None
    user = f"""Dados originais do caso:
<caso>{json.dumps(case, ensure_ascii=False)}</caso>

Explicação a avaliar:
<explicacao>{text}</explicacao>

Dê notas de 1 a 5 e uma justificativa curta, neste JSON:
{{"clareza": n, "utilidade_clinica": n, "fidelidade_aos_dados": n, "tom_adequado": n, "justificativa": "..."}}"""
    raw = client.complete(JUDGE_SYSTEM, user)
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None
