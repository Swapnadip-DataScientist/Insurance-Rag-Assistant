from __future__ import annotations

import json
import math

from dataclasses import dataclass
from typing import Any

import httpx

from src.generation.llm_client import (
    GenerationError,
    InvalidModelResponseError,
    LLMUnavailableError,
    ModelNotAvailableError,
    StructuredLLMResult,
)


# ============================================================
# GROQ CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class GroqConfig:
    api_key: str

    model_name: str = "openai/gpt-oss-20b"

    base_url: str = (
        "https://api.groq.com/openai/v1"
    )

    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 120.0

    temperature: float = 0.1

    # GPT-OSS supports low / medium / high reasoning effort.
    # Low is a reasonable starting point for grounded RAG
    # where retrieval has already selected the evidence.
    reasoning_effort: str = "low"

    def __post_init__(self) -> None:

        if not self.api_key.strip():
            raise ValueError(
                "Groq API key cannot be empty."
            )

        if not self.model_name.strip():
            raise ValueError(
                "Groq model name cannot be empty."
            )

        if (
            not math.isfinite(self.temperature)
            or self.temperature < 0
            or self.temperature > 2
        ):
            raise ValueError(
                "temperature must be finite "
                "and between 0 and 2."
            )

        if self.reasoning_effort not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError(
                "Groq reasoning_effort must be "
                "low, medium or high."
            )


# ============================================================
# GROQ STRUCTURED LLM CLIENT
# ============================================================


class GroqLLMClient:
    """
    Structured-generation client using the Groq API.

    This class contains only provider-specific behaviour.

    Insurance grounding, citation validation and business
    rules remain inside GroundedGenerator.
    """

    def __init__(
        self,
        config: GroqConfig,
    ) -> None:

        self.config = config

        self.base_url = (
            self.config.base_url.rstrip("/")
        )

        timeout = httpx.Timeout(
            connect=(
                self.config.connect_timeout_seconds
            ),
            read=(
                self.config.read_timeout_seconds
            ),
            write=30.0,
            pool=10.0,
        )

        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": (
                    f"Bearer {self.config.api_key}"
                ),
                "Content-Type": (
                    "application/json"
                ),
            },
        )

        self._model_verified = False

    # --------------------------------------------------------
    # GENERIC PROVIDER CONTRACT
    # --------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def close(self) -> None:
        self._client.close()

    def __enter__(
        self,
    ) -> "GroqLLMClient":

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        self.close()

    # ========================================================
    # MODEL PREFLIGHT
    # ========================================================

    def ensure_model_available(self) -> None:
        """
        Confirm that Groq is reachable and that the configured
        model is available to the API key.
        """

        if self._model_verified:
            return

        try:

            response = self._client.get(
                f"{self.base_url}/models"
            )

            response.raise_for_status()

            response_body = response.json()

        except httpx.ConnectError as exc:

            raise LLMUnavailableError(
                "Cannot connect to the Groq API."
            ) from exc

        except httpx.TimeoutException as exc:

            raise LLMUnavailableError(
                "Timed out while contacting Groq."
            ) from exc

        except httpx.HTTPStatusError as exc:

            raise LLMUnavailableError(
                "Groq model-list request failed "
                f"with HTTP "
                f"{exc.response.status_code}."
            ) from exc

        except (ValueError, TypeError) as exc:

            raise LLMUnavailableError(
                "Groq returned an invalid "
                "model-list response."
            ) from exc

        raw_models = response_body.get(
            "data",
            [],
        )

        available_models: set[str] = set()

        if isinstance(raw_models, list):

            for model in raw_models:

                if not isinstance(
                    model,
                    dict,
                ):
                    continue

                model_id = model.get("id")

                if isinstance(
                    model_id,
                    str,
                ):
                    available_models.add(
                        model_id
                    )

        if (
            self.config.model_name
            not in available_models
        ):

            raise ModelNotAvailableError(
                "Groq model "
                f"{self.config.model_name!r} "
                "is not available."
            )

        self._model_verified = True

    # ========================================================
    # STRUCTURED GENERATION
    # ========================================================

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> StructuredLLMResult:

        self.ensure_model_available()

        request_payload = {
            "model": self.config.model_name,

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            "temperature": (
                self.config.temperature
            ),

            "reasoning_effort": (
                self.config.reasoning_effort
            ),

            # Groq strict structured output.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": (
                        "insurance_rag_answer"
                    ),
                    "strict": True,
                    "schema": response_schema,
                },
            },
        }

        try:

            response = self._client.post(
                (
                    f"{self.base_url}"
                    "/chat/completions"
                ),
                json=request_payload,
            )

            response.raise_for_status()

            response_body = response.json()

        except httpx.ConnectError as exc:

            raise LLMUnavailableError(
                "Connection to Groq failed."
            ) from exc

        except httpx.TimeoutException as exc:

            raise LLMUnavailableError(
                "Groq generation exceeded "
                "the configured timeout."
            ) from exc

        except httpx.HTTPStatusError as exc:

            raise GenerationError(
                "Groq generation request failed "
                f"with HTTP "
                f"{exc.response.status_code}."
            ) from exc

        except (ValueError, TypeError) as exc:

            raise InvalidModelResponseError(
                "Groq returned an invalid "
                "HTTP response."
            ) from exc

        # ----------------------------------------------------
        # Parse OpenAI-compatible chat response
        # ----------------------------------------------------

        choices = response_body.get(
            "choices"
        )

        if (
            not isinstance(choices, list)
            or not choices
        ):
            raise InvalidModelResponseError(
                "Groq response contains "
                "no completion choices."
            )

        first_choice = choices[0]

        if not isinstance(
            first_choice,
            dict,
        ):
            raise InvalidModelResponseError(
                "Groq returned an invalid "
                "completion choice."
            )

        message = first_choice.get(
            "message"
        )

        if not isinstance(
            message,
            dict,
        ):
            raise InvalidModelResponseError(
                "Groq response does not "
                "contain a valid message."
            )

        raw_content = message.get(
            "content"
        )

        if (
            not isinstance(raw_content, str)
            or not raw_content.strip()
        ):
            raise InvalidModelResponseError(
                "Groq returned an empty response."
            )

        try:

            parsed_output = json.loads(
                raw_content
            )

        except json.JSONDecodeError as exc:

            raise InvalidModelResponseError(
                "Groq did not return valid JSON."
            ) from exc

        if not isinstance(
            parsed_output,
            dict,
        ):
            raise InvalidModelResponseError(
                "Groq JSON output must "
                "be an object."
            )

        # ----------------------------------------------------
        # Usage metadata
        # ----------------------------------------------------

        usage = response_body.get(
            "usage",
            {},
        )

        if not isinstance(
            usage,
            dict,
        ):
            usage = {}

        actual_model_name = (
            response_body.get("model")
        )

        if not isinstance(
            actual_model_name,
            str,
        ):
            actual_model_name = (
                self.config.model_name
            )

        return StructuredLLMResult(
            parsed_output=parsed_output,

            actual_model_name=(
                actual_model_name
            ),

            prompt_token_count=(
                _optional_int(
                    usage.get(
                        "prompt_tokens"
                    )
                )
            ),

            output_token_count=(
                _optional_int(
                    usage.get(
                        "completion_tokens"
                    )
                )
            ),

            # Groq does not expose the same
            # Ollama nanosecond field.
            total_duration_ns=None,
        )


# ============================================================
# SAFE OPTIONAL INTEGER
# ============================================================


def _optional_int(
    value: Any,
) -> int | None:

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    return None