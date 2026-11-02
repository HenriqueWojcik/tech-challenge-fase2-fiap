"""Cliente de LLM via API (LLMs Aula 5: on-premises x nuvem x API -> escolhemos API).

Groq, OpenAI e Ollama falam o MESMO protocolo (API compatível com a OpenAI). Por isso um único
cliente atende os três; trocar de provedor é só mudar o .env - a mesma ideia da n8n Aula 4,
onde o workflow troca Groq por DeepSeek sem mudar a estrutura da automação.

Provedores:
- groq    : nuvem, gratuito para estudo, rápido (Llama 3.3 70B). Chave em console.groq.com
- openai  : nuvem, pago (gpt-4o-mini)
- ollama  : roda LOCAL (on-premises) - nenhum dado de paciente sai da máquina (LGPD)
- offline : sem LLM; gera o texto por template. Usado nos testes e como fallback.
"""
import logging
import os
import time

log = logging.getLogger("tc2.llm")

PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "openai": {"base_url": None, "model": "gpt-4o-mini"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.1"},
    "offline": {"base_url": None, "model": "template"},
}


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 api_key: str | None = None, temperature: float = 0.2):
        _load_env()
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "offline").lower()
        if self.provider not in PROVIDERS:
            raise ValueError(f"Provedor inválido: {self.provider}. Use {list(PROVIDERS)}")
        self.model = model or os.getenv("LLM_MODEL") or PROVIDERS[self.provider]["model"]
        # Temperatura baixa: em contexto médico queremos respostas consistentes, não criativas.
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv(f"{self.provider.upper()}_API_KEY")
        self._client = None

        if self.provider in ("groq", "openai") and not self.api_key:
            log.warning("Sem chave para '%s' - usando modo offline (template).", self.provider)
            self.provider, self.model = "offline", "template"
        if self.provider != "offline":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key or "ollama",
                                  base_url=PROVIDERS[self.provider]["base_url"])

    @property
    def is_offline(self) -> bool:
        return self.provider == "offline"

    def complete(self, system: str, user: str, offline_fallback=None) -> str:
        """Envia (system, user) e devolve o texto. No modo offline chama `offline_fallback()`."""
        if self.is_offline:
            return offline_fallback() if offline_fallback else "[modo offline: sem LLM configurada]"
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = getattr(resp, "usage", None)
        log.info("LLM %s/%s respondeu em %.1fs (tokens: %s)", self.provider, self.model,
                 time.perf_counter() - start, getattr(usage, "total_tokens", "?"))
        return resp.choices[0].message.content.strip()
