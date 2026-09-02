from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from qdrant_client import QdrantClient

from src.api.config import ApiSettings
from src.api.service import RAGService

from src.generation.generator import (
    GenerationConfig,
    GroundedGenerator
)
from src.generation.provider_factory import (
    build_llm_client,
)

from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
)

from src.retrieval.retriever import (
    BgeM3QueryEncoder,
    ProductionRetriever,
)


@contextmanager
def build_rag_service(
    settings: ApiSettings,
) -> Iterator[RAGService]:
    """
    Build long-lived infrastructure/model objects.

    They live for the FastAPI application lifetime rather
    than being recreated for each /ask request.
    """

    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=60,
    )

    try:

        # -----------------------------------------------------
        # Fail fast if Qdrant or the collection is unavailable.
        # -----------------------------------------------------

        qdrant_client.get_collection(
            settings.qdrant_collection
        )

        # -----------------------------------------------------
        # BGE-M3 query encoder
        # -----------------------------------------------------

        query_encoder = BgeM3QueryEncoder(
            model_name="BAAI/bge-m3",
            device="cpu",
            use_fp16=False,
            max_length=512,
        )

        # -----------------------------------------------------
        # Production Qdrant retriever
        # -----------------------------------------------------

        retriever = ProductionRetriever(
            client=qdrant_client,
            collection_name=(
                settings.qdrant_collection
            ),
            query_encoder=query_encoder,
            vector_name=(
                settings.qdrant_vector_name
            ),
            text_field=(
                settings.qdrant_text_field
            ),
            duplicate_threshold=(
                settings.duplicate_threshold
            ),
        )

        # -----------------------------------------------------
        # CPU BGE cross-encoder reranker
        # -----------------------------------------------------

        reranker = CrossEncoderReranker(
            RerankerConfig(
                model_name=(
                    settings.reranker_model
                ),
                device="cpu",
                batch_size=8,
                max_length=512,
                normalize_scores=True,
            )
        )

        # -----------------------------------------------------
        # Configured LLM provider
        # -----------------------------------------------------

        llm_client = build_llm_client(
            provider=settings.llm_provider,

            generation_timeout_seconds=(
                settings.generation_timeout_seconds
            ),

            # Ollama
            ollama_model=settings.ollama_model,
            ollama_base_url=(
                settings.ollama_base_url
            ),
            ollama_num_ctx=(
                settings.num_ctx           ),

            # Groq
            groq_api_key=settings.groq_api_key,
            groq_model=settings.groq_model,

            # OpenAI
            openai_api_key=(
                settings.openai_api_key
            ),
            openai_model=(
                settings.openai_model
            ),
        )

        try:

            # Fail fast if selected provider/model
            # cannot be reached.
            llm_client.ensure_model_available()

            generator = GroundedGenerator(
                llm_client=llm_client,
                config=GenerationConfig(
                    max_evidence_chunks=(
                        settings.rerank_top_k
                    ),
                ),
            )

            service = RAGService(
                retriever=retriever,
                reranker=reranker,
                generator=generator,
            )

            yield service

        finally:

            # Close HTTP connection owned by
            # Ollama / Groq / OpenAI client.
            llm_client.close()

    finally:

        # Close Qdrant network resources.
        qdrant_client.close()