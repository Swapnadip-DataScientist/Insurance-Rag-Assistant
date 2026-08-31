from __future__ import annotations

import logging
import time

from copy import copy
from dataclasses import is_dataclass, replace
from typing import Any

from src.generation.generator import (
    GroundedGenerator,
)

from src.retrieval.reranker import (
    CrossEncoderReranker,
)

from src.retrieval.retriever import (
    ProductionRetriever,
)


LOGGER = logging.getLogger(__name__)


class RAGService:
    """
    Application service coordinating the completed
    retrieval -> reranking -> generation pipeline.

    FastAPI must not implement retrieval or generation
    algorithms directly.
    """

    def __init__(
        self,
        *,
        retriever: ProductionRetriever,
        reranker: CrossEncoderReranker,
        generator: GroundedGenerator,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 2,
    ) -> None:

        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    # =========================================================
    # CREATE GENERATOR INPUT AFTER RERANKING
    # =========================================================

    def _replace_results(
        self,
        retrieval_response: Any,
        reranked_wrappers: list[Any],
    ) -> Any:
        """
        Preserve the existing RetrievalResponse metadata
        while replacing its result ordering with the final
        reranked evidence.

        The reranker deliberately preserves the original
        candidate objects, so no Qdrant payload or metadata
        is reconstructed here.
        """

        final_results = tuple(
            item.candidate
            for item in reranked_wrappers
        )

        diagnostics = dict(
            getattr(
                retrieval_response,
                "diagnostics",
                {},
            )
        )

        diagnostics["reranking"] = {
            "candidate_k": self.retrieval_top_k,
            "evidence_k": self.rerank_top_k,
            "results": [
                {
                    "retrieval_rank": item.retrieval_rank,
                    "rerank_rank": item.rerank_rank,
                    "rerank_score": round(
                        float(item.rerank_score),
                        6,
                    ),
                }
                for item in reranked_wrappers
            ],
        }

        # Your project uses immutable dataclass-style
        # domain objects in several pipeline components.
        if is_dataclass(retrieval_response):

            return replace(
                retrieval_response,
                results=final_results,
                diagnostics=diagnostics,
            )

        # Defensive fallback if RetrievalResponse is later
        # changed from a dataclass to a normal class.
        cloned_response = copy(
            retrieval_response
        )

        cloned_response.results = final_results
        cloned_response.diagnostics = diagnostics

        return cloned_response

    # =========================================================
    # COMPLETE RAG REQUEST
    # =========================================================

    def ask(
        self,
        query: str,
    ) -> dict[str, Any]:

        started_at = time.perf_counter()

        # -----------------------------------------------------
        # Stage 1
        # BGE-M3 query embedding + Qdrant candidate retrieval
        # -----------------------------------------------------

        retrieval_response = (
            self.retriever.retrieve(
                query,
                top_k=self.retrieval_top_k,
            )
        )

        candidates = list(
            retrieval_response.results
        )

        # -----------------------------------------------------
        # Stage 2
        # Cross-encoder reranking
        # -----------------------------------------------------

        reranked_wrappers = (
            self.reranker.rerank_candidates(
                query=query,
                candidates=candidates,
                top_n=self.rerank_top_k,
            )
        )

        # -----------------------------------------------------
        # Stage 3
        # Present ONLY final Top-2 evidence to Qwen
        # -----------------------------------------------------

        generator_input = (
            self._replace_results(
                retrieval_response,
                reranked_wrappers,
            )
        )

        # -----------------------------------------------------
        # Stage 4
        # Grounded structured generation
        # -----------------------------------------------------

        generation_response = (
            self.generator.generate(
                generator_input
            )
        )

        response = (
            generation_response.to_dict()
        )

        # -----------------------------------------------------
        # Stage 5
        # End-to-end latency
        #
        # Existing generation_latency_ms only measures Qwen.
        # This measures:
        #
        # embedding
        # + Qdrant
        # + reranking
        # + generation
        # -----------------------------------------------------

        response["latency_ms"] = round(
            (
                time.perf_counter()
                - started_at
            )
            * 1000,
            3,
        )

        LOGGER.info(
            (
                "RAG request complete | "
                "candidate_k=%d | "
                "evidence_k=%d | "
                "latency_ms=%.3f"
            ),
            self.retrieval_top_k,
            self.rerank_top_k,
            response["latency_ms"],
        )

        return response