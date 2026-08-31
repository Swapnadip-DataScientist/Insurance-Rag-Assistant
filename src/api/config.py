from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiSettings:
    """
    Runtime configuration for the Insurance RAG API.

    Keep infrastructure/runtime configuration outside
    the endpoint implementation.
    """

    app_name: str = "Insurance RAG Assistant API"
    app_version: str = "1.0.0"

    # ---------------------------------------------------------
    # Qdrant
    # ---------------------------------------------------------

    qdrant_host: str = os.getenv(
        "QDRANT_HOST",
        "localhost",
    )

    qdrant_port: int = int(
        os.getenv(
            "QDRANT_PORT",
            "6333",
        )
    )

    qdrant_collection: str = os.getenv(
        "QDRANT_COLLECTION",
        "insurance_policy_chunks_bge_m3_v1",
    )

    qdrant_vector_name: str = os.getenv(
        "QDRANT_VECTOR_NAME",
        "dense",
    )

    qdrant_text_field: str = os.getenv(
        "QDRANT_TEXT_FIELD",
        "text",
    )

    duplicate_threshold: float = float(
        os.getenv(
            "DUPLICATE_THRESHOLD",
            "0.85",
        )
    )

    # ---------------------------------------------------------
    # Retrieval / reranking
    # ---------------------------------------------------------

    retrieval_top_k: int = int(
        os.getenv(
            "RETRIEVAL_TOP_K",
            "10",
        )
    )

    rerank_top_k: int = int(
        os.getenv(
            "RERANK_TOP_K",
            "2",
        )
    )

    reranker_model: str = os.getenv(
        "RERANKER_MODEL",
        "BAAI/bge-reranker-v2-m3",
    )

    # ---------------------------------------------------------
    # Ollama
    # ---------------------------------------------------------

    ollama_model: str = os.getenv(
        "OLLAMA_MODEL",
        "qwen3.5:4b-q4_K_M",
    )

    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    )

    generation_timeout_seconds: float = float(
        os.getenv(
            "GENERATION_TIMEOUT_SECONDS",
            "180",
        )
    )

    num_ctx: int = int(
        os.getenv(
            "OLLAMA_NUM_CTX",
            "8192",
        )
    )

    # ---------------------------------------------------------
    # API capacity
    # ---------------------------------------------------------

    # CPU-only deployment:
    # avoid several expensive reranker/LLM jobs running
    # simultaneously on the laptop.
    max_concurrent_requests: int = int(
        os.getenv(
            "MAX_CONCURRENT_REQUESTS",
            "1",
        )
    )

    def validate(self) -> None:

        if self.retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than 0."
            )

        if self.rerank_top_k <= 0:
            raise ValueError(
                "rerank_top_k must be greater than 0."
            )

        if self.rerank_top_k > self.retrieval_top_k:
            raise ValueError(
                "rerank_top_k cannot exceed retrieval_top_k."
            )

        if self.max_concurrent_requests <= 0:
            raise ValueError(
                "max_concurrent_requests must be greater than 0."
            )