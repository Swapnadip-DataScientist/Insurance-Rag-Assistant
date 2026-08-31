from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AnswerStatus = Literal[
    "covered",
    "not_covered",
    "conditional",
    "ambiguous",
    "insufficient_evidence",
]


class AskRequest(BaseModel):
    """
    External request accepted by POST /ask.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Insurance policy question.",
        examples=[
            (
                "Will Aviva cover a home charging "
                "point rated above 32 amps?"
            )
        ],
    )


class SourceResponse(BaseModel):
    """
    Source provenance returned to the client.

    Full policy text is deliberately not returned.
    """

    citation_id: str

    point_id: str | None = None

    document_id: str | None = None

    source_file: str | None = None

    page_number: int | None = None

    page_chunk_index: int | None = None


class AskResponse(BaseModel):
    """
    Stable public response contract for the RAG service.
    """

    query: str

    answer_status: AnswerStatus

    answer: str

    citations: list[str]

    conditions: list[str]

    limitations: list[str]

    sources: list[SourceResponse]

    latency_ms: float = Field(
        ge=0,
        description="Total API RAG pipeline latency.",
    )


class HealthResponse(BaseModel):

    status: Literal[
        "ok",
        "ready",
    ]