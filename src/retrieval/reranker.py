from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Generic, TypeVar

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_RERANKER_MODEL = (
    "BAAI/bge-reranker-v2-m3"
)

T = TypeVar("T")


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RerankerConfig:
    """
    Configuration for the BGE cross-encoder style reranker.

    The project currently runs CPU-only.
    """

    model_name: str = DEFAULT_RERANKER_MODEL

    device: str = "cpu"

    # Conservative CPU value.
    # Can increase later after latency benchmarking.
    batch_size: int = 4

    # Query + passage combined length.
    max_length: int = 512

    # Convert raw reranker logits to 0..1 using sigmoid.
    #
    # IMPORTANT:
    # This remains a relevance score, NOT a probability that
    # an insurance claim is covered.
    normalize_scores: bool = True


# =============================================================================
# RERANKED RESULT
# =============================================================================


@dataclass(frozen=True)
class RerankedCandidate(Generic[T]):
    """
    Preserve the original retrieval result while attaching
    second-stage ranking metadata.
    """

    candidate: T

    # Original BGE-M3 / Qdrant ranking.
    retrieval_rank: int

    # New BGE reranker ranking.
    rerank_rank: int

    # BGE reranker relevance score.
    rerank_score: float


# =============================================================================
# DEFAULT TEXT GETTER
# =============================================================================


def default_text_getter(
    candidate: Any,
) -> str:
    """
    Extract passage text without modifying retriever.py.

    Supported candidate shapes:

        candidate.text
        candidate.chunk_text
        candidate.content

        candidate.payload["text"]

        dictionary equivalents
    """

    text_fields = (
        "text",
        "chunk_text",
        "content",
    )

    # -------------------------------------------------------------------------
    # Dictionary-like result
    # -------------------------------------------------------------------------

    if isinstance(
        candidate,
        Mapping,
    ):

        for field_name in text_fields:

            value = candidate.get(
                field_name
            )

            if (
                isinstance(value, str)
                and value.strip()
            ):
                return value.strip()

        payload = candidate.get(
            "payload"
        )

        if isinstance(
            payload,
            Mapping,
        ):

            for field_name in text_fields:

                value = payload.get(
                    field_name
                )

                if (
                    isinstance(value, str)
                    and value.strip()
                ):
                    return value.strip()

    # -------------------------------------------------------------------------
    # Dataclass / Python object
    # -------------------------------------------------------------------------

    for field_name in text_fields:

        value = getattr(
            candidate,
            field_name,
            None,
        )

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value.strip()

    # -------------------------------------------------------------------------
    # Qdrant-style payload attribute
    # -------------------------------------------------------------------------

    payload = getattr(
        candidate,
        "payload",
        None,
    )

    if isinstance(
        payload,
        Mapping,
    ):

        for field_name in text_fields:

            value = payload.get(
                field_name
            )

            if (
                isinstance(value, str)
                and value.strip()
            ):
                return value.strip()

    raise ValueError(
        "Could not extract candidate text. "
        "Expected text/chunk_text/content either directly "
        "on the candidate or inside candidate.payload."
    )


# =============================================================================
# BGE RERANKER
# =============================================================================


class CrossEncoderReranker:
    """
    BGE reranker implementation.

    NOTE:
    We intentionally retain the existing class name
    CrossEncoderReranker so that evaluator.py and the rest of
    the completed pipeline do not need to change.

    Production flow:

        BGE-M3 embeddings
             ↓
        Qdrant retrieval
             ↓
        Top-K candidates
             ↓
        BGE reranker-v2-m3
             ↓
        Top-N evidence
             ↓
        Generator
    """

    def __init__(
        self,
        config: RerankerConfig | None = None,
    ) -> None:

        self.config = (
            config
            or RerankerConfig()
        )

        self._validate_config()

        # Lazy-loaded Hugging Face components.
        self._tokenizer = None

        self._model: (
            AutoModelForSequenceClassification
            | None
        ) = None

        self._load_lock = Lock()


    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_config(
        self,
    ) -> None:

        if not self.config.model_name.strip():

            raise ValueError(
                "model_name must not be empty."
            )

        if self.config.device != "cpu":

            raise ValueError(
                "Current project baseline is CPU-only. "
                "Set device='cpu'."
            )

        if self.config.batch_size <= 0:

            raise ValueError(
                "batch_size must be greater than 0."
            )

        if self.config.max_length <= 0:

            raise ValueError(
                "max_length must be greater than 0."
            )


    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def _load_model(
        self,
    ) -> None:
        """
        Load tokenizer + sequence-classification model once.

        BAAI/bge-reranker-v2-m3 is an encoder reranker that
        produces one relevance logit for each query/passage pair.
        """

        if (
            self._model is not None
            and self._tokenizer is not None
        ):
            return

        with self._load_lock:

            if (
                self._model is not None
                and self._tokenizer is not None
            ):
                return

            logger.info(
                (
                    "Loading BGE reranker | "
                    "model=%s | "
                    "device=%s | "
                    "batch_size=%d | "
                    "max_length=%d"
                ),
                self.config.model_name,
                self.config.device,
                self.config.batch_size,
                self.config.max_length,
            )

            started = time.perf_counter()

            # -----------------------------------------------------------------
            # TOKENIZER
            # -----------------------------------------------------------------

            self._tokenizer = (
                AutoTokenizer.from_pretrained(
                    self.config.model_name,
                )
            )

            # -----------------------------------------------------------------
            # MODEL
            # -----------------------------------------------------------------
            #
            # Official BGE usage uses
            # AutoModelForSequenceClassification.
            #
            # We deliberately do NOT use:
            #
            #     sentence_transformers.CrossEncoder
            #
            # because that was the source of the previous compatibility error.
            # -----------------------------------------------------------------

            self._model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    self.config.model_name,
                )
            )

            self._model.to(
                self.config.device
            )

            self._model.eval()

            elapsed_seconds = (
                time.perf_counter()
                - started
            )

            logger.info(
                (
                    "BGE reranker loaded | "
                    "elapsed_seconds=%.2f"
                ),
                elapsed_seconds,
            )


    # =========================================================================
    # SCORE ONE BATCH
    # =========================================================================

    def _score_batch(
        self,
        query: str,
        passages: Sequence[str],
    ) -> np.ndarray:
        """
        Score one batch of query/passage pairs.

        Returns:
            numpy float32 array with one score per passage.
        """

        self._load_model()

        if (
            self._model is None
            or self._tokenizer is None
        ):
            raise RuntimeError(
                "Reranker model failed to load."
            )

        # ---------------------------------------------------------------------
        # BGE expects paired inputs:
        #
        # [
        #     [query, passage_1],
        #     [query, passage_2],
        #     ...
        # ]
        # ---------------------------------------------------------------------

        pairs = [
            [
                query,
                passage,
            ]
            for passage in passages
        ]

        encoded = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=(
                self.config.max_length
            ),
        )

        # Move token tensors to CPU.
        encoded = {
            key: value.to(
                self.config.device
            )
            for key, value
            in encoded.items()
        }

        # ---------------------------------------------------------------------
        # INFERENCE
        # ---------------------------------------------------------------------

        with torch.no_grad():

            output = self._model(
                **encoded,
                return_dict=True,
            )

            logits = (
                output.logits
                .view(-1)
                .float()
            )

            if (
                self.config
                .normalize_scores
            ):

                logits = torch.sigmoid(
                    logits
                )

        scores = (
            logits
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        return scores


    # =========================================================================
    # RERANK PASSAGES
    # =========================================================================

    def rerank_passages(
        self,
        query: str,
        passages: Sequence[str],
        *,
        top_n: int | None = None,
    ) -> list[
        tuple[int, float]
    ]:
        """
        Rerank passage strings.

        Returns:

            [
                (
                    original_zero_based_index,
                    rerank_score
                ),
                ...
            ]

        sorted from highest relevance to lowest.
        """

        # ---------------------------------------------------------------------
        # QUERY VALIDATION
        # ---------------------------------------------------------------------

        if not isinstance(
            query,
            str,
        ):

            raise TypeError(
                "query must be a string."
            )

        clean_query = query.strip()

        if not clean_query:

            raise ValueError(
                "query must not be empty."
            )

        # ---------------------------------------------------------------------
        # EMPTY RETRIEVAL
        # ---------------------------------------------------------------------

        if not passages:

            return []

        # ---------------------------------------------------------------------
        # PASSAGE VALIDATION
        # ---------------------------------------------------------------------

        clean_passages: list[str] = []

        for index, passage in enumerate(
            passages
        ):

            if (
                not isinstance(
                    passage,
                    str,
                )
                or not passage.strip()
            ):

                raise ValueError(
                    f"Passage at index "
                    f"{index} must be a "
                    "non-empty string."
                )

            clean_passages.append(
                passage.strip()
            )

        # ---------------------------------------------------------------------
        # TOP-N
        # ---------------------------------------------------------------------

        requested_top_n = (
            len(clean_passages)
            if top_n is None
            else top_n
        )

        if requested_top_n <= 0:

            raise ValueError(
                "top_n must be greater than 0."
            )

        effective_top_n = min(
            requested_top_n,
            len(clean_passages),
        )

        # ---------------------------------------------------------------------
        # BATCHED CPU SCORING
        # ---------------------------------------------------------------------

        started = time.perf_counter()

        all_scores: list[
            float
        ] = []

        batch_size = (
            self.config.batch_size
        )

        for start_index in range(
            0,
            len(clean_passages),
            batch_size,
        ):

            end_index = min(
                start_index
                + batch_size,
                len(clean_passages),
            )

            batch_passages = (
                clean_passages[
                    start_index:end_index
                ]
            )

            batch_scores = (
                self._score_batch(
                    query=clean_query,
                    passages=batch_passages,
                )
            )

            all_scores.extend(
                float(score)
                for score
                in batch_scores
            )

        scores = np.asarray(
            all_scores,
            dtype=np.float32,
        )

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        # ---------------------------------------------------------------------
        # DEFENSIVE CHECKS
        # ---------------------------------------------------------------------

        if (
            scores.size
            != len(clean_passages)
        ):

            raise RuntimeError(
                "Reranker score count mismatch: "
                f"scores={scores.size}, "
                f"passages={len(clean_passages)}."
            )

        if not np.isfinite(
            scores
        ).all():

            raise RuntimeError(
                "BGE reranker returned "
                "NaN or Inf scores."
            )

        # ---------------------------------------------------------------------
        # DETERMINISTIC DESCENDING SORT
        # ---------------------------------------------------------------------
        #
        # stable:
        # If two candidates have identical scores,
        # preserve original retrieval ordering.
        # ---------------------------------------------------------------------

        ordered_indices = np.argsort(
            -scores,
            kind="stable",
        )[
            :effective_top_n
        ]

        reranked = [
            (
                int(index),
                float(
                    scores[index]
                ),
            )
            for index
            in ordered_indices
        ]

        logger.info(
            (
                "BGE reranking complete | "
                "candidates=%d | "
                "returned=%d | "
                "elapsed_ms=%.2f"
            ),
            len(clean_passages),
            len(reranked),
            elapsed_ms,
        )

        return reranked


    # =========================================================================
    # RERANK EXISTING RETRIEVAL OBJECTS
    # =========================================================================

    def rerank_candidates(
        self,
        query: str,
        candidates: Sequence[T],
        *,
        top_n: int,
        text_getter: Callable[
            [T],
            str,
        ] = default_text_getter,
    ) -> list[
        RerankedCandidate[T]
    ]:
        """
        Rerank the objects returned by ProductionRetriever.

        The original candidate remains untouched.

        This preserves:

            point_id
            document_id
            source_file
            page_number
            page_chunk_index
            vector retrieval score
            text
            metadata
        """

        if not candidates:

            return []

        passages = [
            text_getter(
                candidate
            )
            for candidate
            in candidates
        ]

        rankings = (
            self.rerank_passages(
                query=query,
                passages=passages,
                top_n=top_n,
            )
        )

        results: list[
            RerankedCandidate[T]
        ] = []

        for (
            rerank_rank,
            (
                original_index,
                rerank_score,
            ),
        ) in enumerate(
            rankings,
            start=1,
        ):

            results.append(
                RerankedCandidate(
                    candidate=(
                        candidates[
                            original_index
                        ]
                    ),
                    retrieval_rank=(
                        original_index
                        + 1
                    ),
                    rerank_rank=(
                        rerank_rank
                    ),
                    rerank_score=(
                        rerank_score
                    ),
                )
            )

        return results


# =============================================================================
# CLI SMOKE TEST
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test "
            "BAAI/bge-reranker-v2-m3."
        )
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--passage",
        action="append",
        required=True,
        help=(
            "Repeat --passage for "
            "multiple candidates."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    args = parse_args()

    reranker = (
        CrossEncoderReranker(
            RerankerConfig(
                model_name=(
                    DEFAULT_RERANKER_MODEL
                ),
                device="cpu",
                batch_size=(
                    args.batch_size
                ),
                max_length=512,
                normalize_scores=True,
            )
        )
    )

    rankings = (
        reranker.rerank_passages(
            query=args.query,
            passages=args.passage,
            top_n=args.top_n,
        )
    )

    output = []

    for (
        rerank_rank,
        (
            original_index,
            rerank_score,
        ),
    ) in enumerate(
        rankings,
        start=1,
    ):

        output.append(
            {
                "rerank_rank": (
                    rerank_rank
                ),
                "retrieval_rank": (
                    original_index
                    + 1
                ),
                "original_index": (
                    original_index
                ),
                "rerank_score": round(
                    rerank_score,
                    6,
                ),
                "passage": (
                    args.passage[
                        original_index
                    ]
                ),
            }
        )

    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()