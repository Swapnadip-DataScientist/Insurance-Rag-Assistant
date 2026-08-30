from __future__ import annotations

import argparse
import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from src.retrieval.chunk_quality import (
    ChunkQualityResult,
    validate_chunk_text,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    point_id: str
    score: float
    text: str
    document_id: str | None
    source_file: str | None
    page_number: int | None
    page_chunk_index: int | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class RejectedCandidate:
    point_id: str
    score: float
    document_id: str | None
    source_file: str | None
    page_number: int | None
    text_repr: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalResponse:
    query: str
    requested_top_k: int
    candidate_k: int
    returned_count: int
    results: tuple[RetrievedChunk, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Query encoder
# ---------------------------------------------------------------------------

class BgeM3QueryEncoder:
    """
    Lazily load BAAI/bge-m3 and generate normalized dense query vectors.

    The model is loaded once per retriever process, not once per query.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        use_fp16: bool = False,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from FlagEmbedding import BGEM3FlagModel

        LOGGER.info("Loading query embedding model: %s", self.model_name)

        self._model = BGEM3FlagModel(
            self.model_name,
            use_fp16=self.use_fp16,
            device=self.device,
        )

    def encode(self, query: str) -> list[float]:
        self._load_model()

        encoded = self._model.encode(
            [query],
            batch_size=1,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        vector = np.asarray(
            encoded["dense_vecs"][0],
            dtype=np.float32,
        )

        if vector.shape != (1024,):
            raise ValueError(
                f"Expected query vector shape (1024,), got {vector.shape}."
            )

        if not np.isfinite(vector).all():
            raise ValueError("Query vector contains NaN or infinity.")

        norm = float(np.linalg.norm(vector))

        if norm == 0:
            raise ValueError("Query embedding has zero norm.")

        # Defensive normalization. Your stored vectors are also normalized.
        vector = vector / norm

        return vector.tolist()


# ---------------------------------------------------------------------------
# Text validation
# ---------------------------------------------------------------------------

def calculate_control_character_ratio(text: str) -> float:
    if not text:
        return 0.0

    control_count = sum(
        1
        for character in text
        if unicodedata.category(character) == "Cc"
        and character not in {"\n", "\r", "\t"}
    )

    return control_count / len(text)


def hard_rejection_reasons(
    text: object,
    quality: ChunkQualityResult,
) -> tuple[str, ...]:
    """
    Return only hard rejection reasons.

    The audit validator is deliberately strict. A short heading such as
    'Winter sports exclusions' may receive a warning but should not
    automatically be removed.

    Hard rejection is reserved for clearly unusable content.
    """

    reasons: list[str] = []

    if not isinstance(text, str):
        reasons.append("text_not_string")
        return tuple(reasons)

    stripped_text = text.strip()

    if not stripped_text:
        reasons.append("empty_or_whitespace_only")
        return tuple(reasons)

    control_ratio = calculate_control_character_ratio(text)

    if quality.printable_ratio < 0.50:
        reasons.append("severely_low_printable_ratio")

    if quality.alphanumeric_ratio < 0.10:
        reasons.append("severely_low_alphanumeric_ratio")

    if control_ratio >= 0.05:
        reasons.append("high_control_character_ratio")

    # Reject examples such as '\x17', '1', '---' and isolated page markers.
    if (
        quality.stripped_length <= 3
        and quality.word_count == 0
    ):
        reasons.append("control_or_symbol_only_text")

    return tuple(dict.fromkeys(reasons))


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def normalize_text_for_comparison(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text.strip()

def compact_text_for_comparison(text: str) -> str:
    """
    Remove whitespace and punctuation for extraction-tolerant comparison.

    This detects overlapping chunks even when PDF extraction removes spaces.
    """

    return "".join(
        character.casefold()
        for character in text
        if character.isalnum()
    )

def character_ngrams(
    text: str,
    n: int = 5,
) -> set[str]:
    compact_text = compact_text_for_comparison(text)

    if not compact_text:
        return set()

    if len(compact_text) < n:
        return {compact_text}

    return {
        compact_text[index:index + n]
        for index in range(len(compact_text) - n + 1)
    }

def text_tokens(text: str) -> set[str]:
    return set(
        re.findall(
            r"\b\w+\b",
            normalize_text_for_comparison(text),
            flags=re.UNICODE,
        )
    )


def duplicate_similarity(
    first_text: str,
    second_text: str,
) -> float:
    """
    Detect exact, contained and heavily overlapping chunks.

    Character n-grams are used because some PDFs lose spaces during
    extraction, making word-token comparison unreliable.
    """

    first_compact = compact_text_for_comparison(first_text)
    second_compact = compact_text_for_comparison(second_text)

    if not first_compact or not second_compact:
        return 0.0

    if first_compact == second_compact:
        return 1.0

    shorter, longer = sorted(
        (first_compact, second_compact),
        key=len,
    )

    # One chunk is fully contained inside another.
    if shorter in longer:
        return 1.0

    first_ngrams = character_ngrams(first_text, n=5)
    second_ngrams = character_ngrams(second_text, n=5)

    if not first_ngrams or not second_ngrams:
        return 0.0

    common_ngrams = first_ngrams & second_ngrams
    smaller_ngram_count = min(
        len(first_ngrams),
        len(second_ngrams),
    )

    return len(common_ngrams) / smaller_ngram_count


def duplicate_rejection_reason(
    *,
    candidate_text: str,
    candidate_document_id: str | None,
    candidate_page_number: int | None,
    candidate_page_chunk_index: int | None,
    accepted_results: Sequence[RetrievedChunk],
    duplicate_threshold: float,
    adjacent_overlap_threshold: float = 0.50,
) -> str | None:
    """
    Detect exact duplicates and overlapping adjacent chunks.

    General near-duplicate detection remains strict at 0.85.
    Adjacent chunks from the same page use a lower threshold because
    intentional chunk overlap causes partial duplication.
    """

    candidate_compact = compact_text_for_comparison(candidate_text)

    for accepted in accepted_results:
        similarity = duplicate_similarity(
            candidate_text,
            accepted.text,
        )

        accepted_compact = compact_text_for_comparison(
            accepted.text
        )

        same_document = (
            candidate_document_id is not None
            and candidate_document_id == accepted.document_id
        )

        same_page = (
            same_document
            and candidate_page_number is not None
            and candidate_page_number == accepted.page_number
        )

        adjacent_chunk = (
            same_page
            and candidate_page_chunk_index is not None
            and accepted.page_chunk_index is not None
            and abs(
                candidate_page_chunk_index
                - accepted.page_chunk_index
            ) == 1
        )

        exact_duplicate = (
            candidate_compact == accepted_compact
        )

        if exact_duplicate:
            return "exact_duplicate"

        if same_document and similarity >= duplicate_threshold:
            return "near_duplicate"

        if (
            adjacent_chunk
            and similarity >= adjacent_overlap_threshold
        ):
            return "adjacent_chunk_overlap"

    return None


# ---------------------------------------------------------------------------
# Qdrant filters
# ---------------------------------------------------------------------------

def match_value_condition(
    field_name: str,
    value: Any,
) -> models.FieldCondition:
    return models.FieldCondition(
        key=field_name,
        match=models.MatchValue(value=value),
    )


def build_qdrant_filter(
    *,
    metadata_filters: dict[str, Any] | None = None,
    include_document_ids: Sequence[str] | None = None,
    exclude_document_ids: Sequence[str] | None = None,
) -> models.Filter | None:
    must_conditions: list[models.Condition] = []
    must_not_conditions: list[models.Condition] = []

    for field_name, value in (metadata_filters or {}).items():
        if value is not None:
            must_conditions.append(
                match_value_condition(field_name, value)
            )

    if include_document_ids:
        must_conditions.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=list(include_document_ids)),
            )
        )

    if exclude_document_ids:
        must_not_conditions.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=list(exclude_document_ids)),
            )
        )

    if not must_conditions and not must_not_conditions:
        return None

    return models.Filter(
        must=must_conditions or None,
        must_not=must_not_conditions or None,
    )


# ---------------------------------------------------------------------------
# Production retriever
# ---------------------------------------------------------------------------

class ProductionRetriever:
    def __init__(
        self,
        *,
        client: QdrantClient,
        collection_name: str,
        query_encoder: BgeM3QueryEncoder,
        vector_name: str = "dense",
        text_field: str = "text",
        oversampling_factor: int = 4,
        minimum_candidate_k: int = 20,
        maximum_candidate_k: int = 100,
        duplicate_threshold: float = 0.85,
    ) -> None:
        if oversampling_factor < 1:
            raise ValueError("oversampling_factor must be at least 1.")

        if not 0.0 <= duplicate_threshold <= 1.0:
            raise ValueError(
                "duplicate_threshold must be between 0 and 1."
            )

        self.client = client
        self.collection_name = collection_name
        self.query_encoder = query_encoder
        self.vector_name = vector_name
        self.text_field = text_field
        self.oversampling_factor = oversampling_factor
        self.minimum_candidate_k = minimum_candidate_k
        self.maximum_candidate_k = maximum_candidate_k
        self.duplicate_threshold = duplicate_threshold

    @staticmethod
    def validate_query(query: object) -> str:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        cleaned_query = re.sub(r"\s+", " ", query).strip()

        if not cleaned_query:
            raise ValueError("Query cannot be empty.")

        if len(cleaned_query) < 3:
            raise ValueError("Query must contain at least 3 characters.")

        if len(cleaned_query) > 2_000:
            raise ValueError("Query exceeds the 2,000-character limit.")

        return cleaned_query

    def _calculate_candidate_k(self, top_k: int) -> int:
        candidate_k = max(
            top_k * self.oversampling_factor,
            self.minimum_candidate_k,
        )

        return min(candidate_k, self.maximum_candidate_k)

    def _search_qdrant(
        self,
        *,
        query_vector: list[float],
        candidate_k: int,
        query_filter: models.Filter | None,
        score_threshold: float | None,
    ) -> list[Any]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=self.vector_name,
            query_filter=query_filter,
            limit=candidate_k,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )

        return list(response.points)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
        include_document_ids: Sequence[str] | None = None,
        exclude_document_ids: Sequence[str] | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResponse:
        cleaned_query = self.validate_query(query)

        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50.")

        if score_threshold is not None and not math.isfinite(
            score_threshold
        ):
            raise ValueError("score_threshold must be finite.")

        candidate_k = self._calculate_candidate_k(top_k)
        query_vector = self.query_encoder.encode(cleaned_query)

        query_filter = build_qdrant_filter(
            metadata_filters=metadata_filters,
            include_document_ids=include_document_ids,
            exclude_document_ids=exclude_document_ids,
        )

        candidates = self._search_qdrant(
            query_vector=query_vector,
            candidate_k=candidate_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        accepted: list[RetrievedChunk] = []
        rejected: list[RejectedCandidate] = []

        rejection_counts: dict[str, int] = {}

        for candidate in candidates:
            payload = dict(candidate.payload or {})
            raw_text = payload.get(self.text_field)
            text = raw_text if isinstance(raw_text, str) else ""

            quality = validate_chunk_text(raw_text)
            rejection_reasons = list(
                hard_rejection_reasons(raw_text, quality)
            )

            if not rejection_reasons:
                duplicate_reason = duplicate_rejection_reason(
                    candidate_text=text,
                    candidate_document_id=payload.get("document_id"),
                    candidate_page_number=payload.get("page_number"),
                    candidate_page_chunk_index=payload.get(
                        "page_chunk_index"
                    ),
                    accepted_results=accepted,
                    duplicate_threshold=self.duplicate_threshold,
                    adjacent_overlap_threshold=0.50,
                )

                if duplicate_reason:
                    rejection_reasons.append(duplicate_reason)

            if rejection_reasons:
                for reason in rejection_reasons:
                    rejection_counts[reason] = (
                        rejection_counts.get(reason, 0) + 1
                    )

                rejected.append(
                    RejectedCandidate(
                        point_id=str(candidate.id),
                        score=float(candidate.score),
                        document_id=payload.get("document_id"),
                        source_file=payload.get("source_file"),
                        page_number=payload.get("page_number"),
                        text_repr=repr(raw_text)[:500],
                        reasons=tuple(rejection_reasons),
                    )
                )
                continue

            accepted.append(
                RetrievedChunk(
                    rank=len(accepted) + 1,
                    point_id=str(candidate.id),
                    score=float(candidate.score),
                    text=text.strip(),
                    document_id=payload.get("document_id"),
                    source_file=payload.get("source_file"),
                    page_number=payload.get("page_number"),
                    page_chunk_index=payload.get(
                        "page_chunk_index"
                    ),
                    payload=payload,
                )
            )

            if len(accepted) >= top_k:
                break

        diagnostics = {
            "qdrant_candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "rejection_counts": rejection_counts,
            "metadata_filters": metadata_filters or {},
            "include_document_ids": list(
                include_document_ids or []
            ),
            "exclude_document_ids": list(
                exclude_document_ids or []
            ),
            "duplicate_threshold": self.duplicate_threshold,
            "score_threshold": score_threshold,
        }

        return RetrievalResponse(
            query=cleaned_query,
            requested_top_k=top_k,
            candidate_k=candidate_k,
            returned_count=len(accepted),
            results=tuple(accepted),
            rejected_candidates=tuple(rejected),
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_metadata_filters(
    filter_values: Iterable[str],
) -> dict[str, str]:
    filters: dict[str, str] = {}

    for item in filter_values:
        if "=" not in item:
            raise ValueError(
                f"Invalid filter {item!r}. Expected FIELD=VALUE."
            )

        field_name, value = item.split("=", maxsplit=1)
        field_name = field_name.strip()
        value = value.strip()

        if not field_name or not value:
            raise ValueError(
                f"Invalid filter {item!r}. Expected FIELD=VALUE."
            )

        filters[field_name] = value

    return filters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validated production Qdrant dense retriever."
    )

    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument(
        "--collection",
        default="insurance_policy_chunks_bge_m3_v1",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--vector-name", default="dense")
    parser.add_argument("--text-field", default="text")

    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help=(
            "Exact-match payload filter. May be specified multiple times."
        ),
    )

    parser.add_argument(
        "--include-document-id",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--exclude-document-id",
        action="append",
        default=[],
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--show-rejected",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    client = QdrantClient(
        host=args.host,
        port=args.port,
        timeout=60,
    )

    query_encoder = BgeM3QueryEncoder(
        model_name="BAAI/bge-m3",
        device="cpu",
        use_fp16=False,
        max_length=512,
    )

    retriever = ProductionRetriever(
        client=client,
        collection_name=args.collection,
        query_encoder=query_encoder,
        vector_name=args.vector_name,
        text_field=args.text_field,
        duplicate_threshold=args.duplicate_threshold,
    )
    # ========================================================================
    # SECOND-STAGE CROSS-ENCODER RERANKER
    # ========================================================================
    # The retriever uses BGE-M3 + Qdrant to efficiently identify a candidate
    # set using vector similarity.
    #
    # The cross-encoder then evaluates the original query together with each
    # retrieved chunk and produces a more precise relevance ordering.
    #
    # The model is initialized here once and reused for all candidates in the
    # current execution.
    # ========================================================================

    reranker = CrossEncoderReranker(
    RerankerConfig(
        device="cpu",
        batch_size=8,
        max_length=512,
    )
    )
    metadata_filters = parse_metadata_filters(args.filter)

    response = retriever.retrieve(
        args.query,
        top_k=args.top_k,
        metadata_filters=metadata_filters,
        include_document_ids=args.include_document_id,
        exclude_document_ids=args.exclude_document_id,
        score_threshold=args.score_threshold,
    )
    # ========================================================================
    # SECOND-STAGE RERANKING
    # ========================================================================
    # response.results contains the chunks already selected and validated by
    # the ProductionRetriever.
    #
    # The reranker does NOT query Qdrant again and does NOT change the original
    # retrieval objects. It simply reorders them according to cross-encoder
    # relevance.
    # ========================================================================    

    reranked_results = reranker.rerank_candidates(
        query=args.query,
        candidates=response.results,
        top_n=3,
    )

    print("\n" + "=" * 100)
    print("RETRIEVAL SUMMARY")
    print("=" * 100)
    print(f"Query: {response.query}")
    print(f"Requested top-k: {response.requested_top_k}")
    print(f"Candidate-k: {response.candidate_k}")
    print(f"Returned: {response.returned_count}")
    print(f"Diagnostics: {response.diagnostics}")

    #for result in response.results:
    for result in reranked_results:
        candidate = result.candidate

        print("\n" + "=" * 100)

        print(f"Rerank Rank: {result.rerank_rank}")
        print(f"Original Rank: {result.retrieval_rank}")
        print(f"Reranker Score: {result.rerank_score:.6f}")

        print("-" * 100)

        print(f"Original Retrieval Rank: {candidate.rank}")
        print(f"Vector Score: {candidate.score:.6f}")
        print(f"Point ID: {candidate.point_id}")
        print(f"Document ID: {candidate.document_id}")
        print(f"Source file: {candidate.source_file}")
        print(f"Page: {candidate.page_number}")
        print(
            f"Page chunk index: "
            f"{candidate.page_chunk_index}"
        )
        print(f"Text: {candidate.text}")

    if args.show_rejected:
        print("\n" + "=" * 100)
        print("REJECTED CANDIDATES")
        print("=" * 100)

        for rejected in response.rejected_candidates:
            print(f"\nPoint ID: {rejected.point_id}")
            print(f"Score: {rejected.score:.6f}")
            print(f"Source: {rejected.source_file}")
            print(f"Page: {rejected.page_number}")
            print(f"Reasons: {rejected.reasons}")
            print(f"Text repr: {rejected.text_repr}")


if __name__ == "__main__":
    main()