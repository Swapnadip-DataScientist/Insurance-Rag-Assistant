from __future__ import annotations

from src.generation.llm_client import (
    StructuredLLMClient,
)

from src.generation.generator import (
    OllamaConfig,
    OllamaQwenClient,
)

from src.generation.providers.groq import (
    GroqConfig,
    GroqLLMClient,
)

from src.generation.providers.openai import (
    OpenAIConfig,
    OpenAILLMClient,
)



# ============================================================
# LLM PROVIDER FACTORY
# ============================================================


def build_llm_client(
    *,
    provider: str,
    generation_timeout_seconds: float,

    # Ollama
    ollama_model: str,
    ollama_base_url: str,
    ollama_num_ctx: int,

    # Groq
    groq_api_key: str | None,
    groq_model: str,

    # OpenAI
    openai_api_key: str | None,
    openai_model: str,

) -> StructuredLLMClient:
    """
    Create the configured LLM provider.

    The rest of the RAG application depends only on the
    StructuredLLMClient contract.

    Provider-specific construction is centralized here.
    """

    normalized_provider = (
        provider.strip().lower()
    )

    # ========================================================
    # OLLAMA
    # ========================================================

    if normalized_provider == "ollama":

        config = OllamaConfig(
            model_name=ollama_model,
            base_url=ollama_base_url,
            read_timeout_seconds=(
                generation_timeout_seconds
            ),
            num_ctx=ollama_num_ctx,
        )

        return OllamaQwenClient(
            config
        )

    # ========================================================
    # GROQ
    # ========================================================

    if normalized_provider == "groq":

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required "
                "when LLM_PROVIDER=groq."
            )

        config = GroqConfig(
            api_key=groq_api_key,
            model_name=groq_model,
            read_timeout_seconds=(
                generation_timeout_seconds
            ),
        )

        return GroqLLMClient(
            config
        )

    # ========================================================
    # OPENAI
    # ========================================================

    if normalized_provider == "openai":

        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required "
                "when LLM_PROVIDER=openai."
            )

        config = OpenAIConfig(
            api_key=openai_api_key,
            model_name=openai_model,
            read_timeout_seconds=(
                generation_timeout_seconds
            ),
        )

        return OpenAILLMClient(
            config
        )

    # ========================================================
    # UNKNOWN PROVIDER
    # ========================================================

    raise ValueError(
        "Unsupported LLM provider: "
        f"{provider!r}. "
        "Supported providers are: "
        "ollama, groq, openai."
    )