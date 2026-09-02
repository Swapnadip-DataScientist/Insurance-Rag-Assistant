from __future__ import annotations

import json

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
# OPENAI CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str

    model_name: str = "gpt-5.6-luna"

    base_url: str = "https://api.openai.com/v1"

    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:

        if not self.api_key.strip():
            raise ValueError(
                "OpenAI API key cannot be empty."
            )

        if not self.model_name.strip():
            raise ValueError(
                "OpenAI model name cannot be empty."
            )


# ============================================================
# OPENAI STRUCTURED LLM CLIENT
# ============================================================


class OpenAILLMClient:
    """
    Structured-generation client using the OpenAI
    Responses API.

    Provider-specific HTTP behaviour stays here.

    Insurance grounding, citations and business rules
    remain inside GroundedGenerator.
    """

    def __init__(
        self,
        config: OpenAIConfig,
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
                "Content-Type": "application/json",
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
    ) -> "OpenAILLMClient":

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
        Verify that OpenAI is reachable and the configured
        model is available to the API key.
        """

        if self._model_verified:
            return

        try:

            response = self._client.get(
                (
                    f"{self.base_url}/models/"
                    f"{self.config.model_name}"
                )
            )

            response.raise_for_status()

            response_body = response.json()

        except httpx.ConnectError as exc:

            raise LLMUnavailableError(
                "Cannot connect to the OpenAI API."
            ) from exc

        except httpx.TimeoutException as exc:

            raise LLMUnavailableError(
                "Timed out while contacting OpenAI."
            ) from exc

        except httpx.HTTPStatusError as exc:

            if exc.response.status_code == 404:

                raise ModelNotAvailableError(
                    "OpenAI model "
                    f"{self.config.model_name!r} "
                    "is not available."
                ) from exc

            raise LLMUnavailableError(
                "OpenAI model validation failed "
                f"with HTTP "
                f"{exc.response.status_code}."
            ) from exc

        except (ValueError, TypeError) as exc:

            raise LLMUnavailableError(
                "OpenAI returned an invalid "
                "model response."
            ) from exc

        returned_model = response_body.get("id")

        if (
            not isinstance(returned_model, str)
            or returned_model
            != self.config.model_name
        ):
            raise ModelNotAvailableError(
                "OpenAI did not confirm model "
                f"{self.config.model_name!r}."
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

            # System/developer instructions.
            "instructions": system_prompt,

            # User question + retrieved evidence prompt.
            "input": user_prompt,

            # Do not persist response state for this
            # stateless RAG request.
            "store": False,

            # OpenAI Responses API structured output.
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "insurance_rag_answer",
                    "schema": response_schema,
                    "strict": True,
                }
            },
        }

        try:

            response = self._client.post(
                f"{self.base_url}/responses",
                json=request_payload,
            )

            response.raise_for_status()

            response_body = response.json()

        except httpx.ConnectError as exc:

            raise LLMUnavailableError(
                "Connection to OpenAI failed."
            ) from exc

        except httpx.TimeoutException as exc:

            raise LLMUnavailableError(
                "OpenAI generation exceeded "
                "the configured timeout."
            ) from exc

        except httpx.HTTPStatusError as exc:

            raise GenerationError(
                "OpenAI generation request failed "
                f"with HTTP "
                f"{exc.response.status_code}."
            ) from exc

        except (ValueError, TypeError) as exc:

            raise InvalidModelResponseError(
                "OpenAI returned an invalid "
                "HTTP response."
            ) from exc

        # ----------------------------------------------------
        # RESPONSE STATUS
        # ----------------------------------------------------

        status = response_body.get("status")

        if status not in {
            None,
            "completed",
        }:
            raise InvalidModelResponseError(
                "OpenAI response did not complete "
                f"successfully. Status: {status!r}."
            )

        # ----------------------------------------------------
        # EXTRACT OUTPUT TEXT
        # ----------------------------------------------------

        raw_content = _extract_output_text(
            response_body
        )

        if not raw_content:
            raise InvalidModelResponseError(
                "OpenAI returned no structured "
                "text output."
            )

        # ----------------------------------------------------
        # PARSE STRUCTURED JSON
        # ----------------------------------------------------

        try:

            parsed_output = json.loads(
                raw_content
            )

        except json.JSONDecodeError as exc:

            raise InvalidModelResponseError(
                "OpenAI did not return valid JSON."
            ) from exc

        if not isinstance(
            parsed_output,
            dict,
        ):
            raise InvalidModelResponseError(
                "OpenAI JSON output must "
                "be an object."
            )

        # ----------------------------------------------------
        # USAGE
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
                        "input_tokens"
                    )
                )
            ),

            output_token_count=(
                _optional_int(
                    usage.get(
                        "output_tokens"
                    )
                )
            ),

            total_duration_ns=None,
        )


# ============================================================
# RESPONSE TEXT EXTRACTION
# ============================================================


def _extract_output_text(
    response_body: dict[str, Any],
) -> str | None:
    """
    Extract assistant output text from the raw
    Responses API JSON structure.
    """

    output = response_body.get(
        "output"
    )

    if not isinstance(
        output,
        list,
    ):
        return None

    for output_item in output:

        if not isinstance(
            output_item,
            dict,
        ):
            continue

        if output_item.get("type") != "message":
            continue

        content = output_item.get(
            "content"
        )

        if not isinstance(
            content,
            list,
        ):
            continue

        for content_item in content:

            if not isinstance(
                content_item,
                dict,
            ):
                continue

            # Explicit refusal should not be
            # interpreted as structured JSON.
            if (
                content_item.get("type")
                == "refusal"
            ):
                raise InvalidModelResponseError(
                    "OpenAI refused the "
                    "generation request."
                )

            if (
                content_item.get("type")
                != "output_text"
            ):
                continue

            text = content_item.get(
                "text"
            )

            if (
                isinstance(text, str)
                and text.strip()
            ):
                return text

    return None


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