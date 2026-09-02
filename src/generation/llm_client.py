from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


# ============================================================
# GENERIC LLM EXCEPTIONS
# ============================================================


class GenerationError(RuntimeError):
    """Base exception for LLM generation failures."""


class LLMUnavailableError(GenerationError):
    """Raised when the configured LLM provider is unavailable."""


class ModelNotAvailableError(GenerationError):
    """Raised when the configured model is unavailable."""


class InvalidModelResponseError(GenerationError):
    """Raised when an LLM returns an invalid response."""


# ============================================================
# NORMALIZED LLM RESULT
# ============================================================


@dataclass(frozen=True)
class StructuredLLMResult:
    parsed_output: dict[str, Any]
    actual_model_name: str
    prompt_token_count: int | None = None
    output_token_count: int | None = None
    total_duration_ns: int | None = None


# ============================================================
# PROVIDER CONTRACT
# ============================================================


class StructuredLLMClient(Protocol):
    """
    Contract implemented by every LLM provider.
    """

    @property
    def model_name(self) -> str:
        ...

    def ensure_model_available(self) -> None:
        ...

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> StructuredLLMResult:
        ...

    def close(self) -> None:
        ...