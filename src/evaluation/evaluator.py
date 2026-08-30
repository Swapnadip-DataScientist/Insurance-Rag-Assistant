from __future__ import annotations

import argparse
import json
import logging
import math

from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from qdrant_client import QdrantClient

from src.retrieval.retriever import (
    BgeM3QueryEncoder,
    ProductionRetriever,
)

from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
)


LOGGER = logging.getLogger(__name__)


# =============================================================================
# GOLDEN DATASET
# =============================================================================


def load_golden_queries(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Load golden evaluation queries from JSONL.

    Each non-empty line must contain one JSON object.

    Example:
    {
        "id": "Q001",
        "query": "...",
        "relevant_sources": [
            {
                "document_id": "...",
                "page_number": 10
            }
        ],
        "expected_answer_status": "conditional"
    }
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Golden query file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):

            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {path}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    f"Line {line_number} must "
                    "contain a JSON object."
                )

            query_id = record.get("id")
            query = record.get("query")
            relevant_sources = record.get(
                "relevant_sources"
            )

            if (
                not isinstance(query_id, str)
                or not query_id.strip()
            ):
                raise ValueError(
                    f"Missing/invalid id "
                    f"at line {line_number}."
                )

            if (
                not isinstance(query, str)
                or not query.strip()
            ):
                raise ValueError(
                    f"Missing/invalid query "
                    f"at line {line_number}."
                )

            if (
                not isinstance(
                    relevant_sources,
                    list,
                )
                or not relevant_sources
            ):
                raise ValueError(
                    f"relevant_sources must be "
                    f"a non-empty list at line "
                    f"{line_number}."
                )

            for source_number, source in enumerate(
                relevant_sources,
                start=1,
            ):

                if not isinstance(
                    source,
                    dict,
                ):
                    raise ValueError(
                        f"Relevant source "
                        f"{source_number} "
                        f"for {query_id} "
                        "must be an object."
                    )

                has_point_id = bool(
                    source.get("point_id")
                )

                has_document_page = (
                    source.get("document_id")
                    is not None
                    and source.get("page_number")
                    is not None
                )

                if not (
                    has_point_id
                    or has_document_page
                ):
                    raise ValueError(
                        f"Relevant source "
                        f"{source_number} "
                        f"for {query_id} must "
                        "contain either point_id "
                        "or document_id + page_number."
                    )

            records.append(record)

    if not records:
        raise ValueError(
            "Golden query file contains "
            "no evaluation queries."
        )

    return records


# =============================================================================
# SAFE ATTRIBUTE ACCESS
# =============================================================================


def get_value(
    candidate: Any,
    field_name: str,
    default: Any = None,
) -> Any:
    """
    Read a value safely from either:

        candidate.field_name

    or:

        candidate["field_name"]

    or:

        candidate.payload[field_name]
    """

    if isinstance(candidate, dict):

        if field_name in candidate:
            return candidate.get(
                field_name,
                default,
            )

        payload = candidate.get(
            "payload"
        )

        if isinstance(payload, dict):
            return payload.get(
                field_name,
                default,
            )

        return default

    value = getattr(
        candidate,
        field_name,
        None,
    )

    if value is not None:
        return value

    payload = getattr(
        candidate,
        "payload",
        None,
    )

    if isinstance(payload, dict):
        return payload.get(
            field_name,
            default,
        )

    return default


# =============================================================================
# GOLD SOURCE MATCHING
# =============================================================================


def source_matches_gold(
    candidate,
    relevant_sources,
) -> bool:

    candidate_point_id = get_value(
        candidate,
        "point_id",
    )

    candidate_document_id = get_value(
        candidate,
        "document_id",
    )

    candidate_page_number = get_value(
        candidate,
        "page_number",
    )

    for source in relevant_sources:

        gold_point_id = source.get(
            "point_id"
        )

        # -------------------------------------------------------------
        # STRICT CHUNK-LEVEL MATCH
        # -------------------------------------------------------------
        # If the golden source contains a point_id,
        # only the exact Qdrant chunk counts as relevant.
        # -------------------------------------------------------------

        if gold_point_id is not None:

            if (
                candidate_point_id is not None
                and str(candidate_point_id)
                == str(gold_point_id)
            ):
                return True

            # Do NOT fall back to page matching.
            continue

        # -------------------------------------------------------------
        # PAGE-LEVEL FALLBACK
        # -------------------------------------------------------------
        # Used only when the golden dataset does not contain point_id.
        # -------------------------------------------------------------

        gold_document_id = source.get(
            "document_id"
        )

        gold_page_number = source.get(
            "page_number"
        )

        if (
            gold_document_id is not None
            and gold_page_number is not None
            and candidate_document_id
            == gold_document_id
            and candidate_page_number
            == gold_page_number
        ):
            return True

    return False

# =============================================================================
# HIT@K
# =============================================================================


def hit_at_k(
    results: Sequence[Any],
    relevant_sources: Sequence[
        dict[str, Any]
    ],
    k: int,
) -> float:
    """
    Hit@K = 1 if at least one relevant result
    appears within Top K.

    Otherwise 0.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    return float(
        any(
            source_matches_gold(
                result,
                relevant_sources,
            )
            for result in results[:k]
        )
    )


# =============================================================================
# RECALL@K
# =============================================================================


def recall_at_k(
    results: Sequence[Any],
    relevant_sources: Sequence[
        dict[str, Any]
    ],
    k: int,
) -> float:
    """
    Recall@K:

        number of expected gold sources found
        -------------------------------------
        total number of expected gold sources

    Example:

        Gold sources = 2
        Found in Top 10 = 1

        Recall@10 = 0.5
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    if not relevant_sources:
        return 0.0

    matched = 0

    for gold_source in relevant_sources:

        found = any(
            source_matches_gold(
                result,
                [gold_source],
            )
            for result in results[:k]
        )

        if found:
            matched += 1

    return (
        matched
        / len(relevant_sources)
    )


# =============================================================================
# RECIPROCAL RANK / MRR
# =============================================================================


def reciprocal_rank(
    results: Sequence[Any],
    relevant_sources: Sequence[
        dict[str, Any]
    ],
    k: int,
) -> float:
    """
    Reciprocal Rank:

        first relevant result at rank 1 -> 1.0
        first relevant result at rank 2 -> 0.5
        first relevant result at rank 3 -> 0.3333
        no relevant result              -> 0
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):

        if source_matches_gold(
            result,
            relevant_sources,
        ):
            return 1.0 / rank

    return 0.0


# =============================================================================
# PRECISION@K
# =============================================================================


def precision_at_k(
    results: Sequence[Any],
    relevant_sources: Sequence[
        dict[str, Any]
    ],
    k: int,
) -> float:
    """
    Precision@K:

        relevant results returned
        -------------------------
                  K

    Example:

        Top 3:
            Rank 1 relevant
            Rank 2 irrelevant
            Rank 3 irrelevant

        Precision@3 = 1/3
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    relevant_count = sum(
        1
        for result in results[:k]
        if source_matches_gold(
            result,
            relevant_sources,
        )
    )

    return relevant_count / k


# =============================================================================
# nDCG@K
# =============================================================================


def ndcg_at_k(
    results: Sequence[Any],
    relevant_sources: Sequence[
        dict[str, Any]
    ],
    k: int,
) -> float:
    """
    Binary nDCG@K.

    Relevant result:
        relevance = 1

    Irrelevant result:
        relevance = 0

    Unlike MRR, nDCG considers the ranking positions
    of all relevant results.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    # -------------------------------------------------------------------------
    # DCG
    # -------------------------------------------------------------------------

    dcg = 0.0

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):

        relevance = (
            1.0
            if source_matches_gold(
                result,
                relevant_sources,
            )
            else 0.0
        )

        if relevance:

            dcg += (
                relevance
                / math.log2(
                    rank + 1
                )
            )

    # -------------------------------------------------------------------------
    # IDEAL DCG
    # -------------------------------------------------------------------------

    ideal_relevant_count = min(
        len(relevant_sources),
        k,
    )

    if ideal_relevant_count == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(
            rank + 1
        )
        for rank in range(
            1,
            ideal_relevant_count + 1,
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# =============================================================================
# FIRST RELEVANT RANK
# =============================================================================


def find_first_relevant_rank(
    results: Sequence[Any],
    relevant_sources: Sequence[
        dict[str, Any]
    ],
    k: int,
) -> int | None:
    """
    Find the position of the first correct source.
    """

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):

        if source_matches_gold(
            result,
            relevant_sources,
        ):
            return rank

    return None


# =============================================================================
# CANDIDATE SERIALIZATION
# =============================================================================


def serialize_candidate(
    candidate: Any,
    *,
    rank: int,
    relevant_sources: Sequence[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """
    Convert one retrieval result into JSON-safe
    evaluation metadata.
    """

    score = get_value(
        candidate,
        "score",
    )

    return {
        "rank": rank,
        "relevant": source_matches_gold(
            candidate,
            relevant_sources,
        ),
        "point_id": str(
            get_value(
                candidate,
                "point_id",
                "",
            )
        ),
        "document_id": get_value(
            candidate,
            "document_id",
        ),
        "source_file": get_value(
            candidate,
            "source_file",
        ),
        "page_number": get_value(
            candidate,
            "page_number",
        ),
        "page_chunk_index": get_value(
            candidate,
            "page_chunk_index",
        ),
        "score": (
            round(
                float(score),
                6,
            )
            if score is not None
            else None
        ),
    }


# =============================================================================
# SINGLE QUERY EVALUATION
# =============================================================================


def evaluate_query(
    *,
    retriever: ProductionRetriever,
    reranker: CrossEncoderReranker,
    golden_query: dict[str, Any],
    candidate_k: int,
    rerank_k: int,
) -> dict[str, Any]:
    """
    Evaluate one golden query.

    Fair comparison:

        Retriever Top 10
              │
              ├── Baseline first 3
              │
              └── Rerank SAME 10 -> first 3

    The reranker does NOT perform another retrieval.
    """

    query_id = golden_query["id"]
    query = golden_query["query"]

    relevant_sources = golden_query[
        "relevant_sources"
    ]

    # =========================================================================
    # STAGE 1: BGE-M3 + QDRANT
    # =========================================================================

    retrieval_response = retriever.retrieve(
        query,
        top_k=candidate_k,
    )

    candidates = list(
        retrieval_response.results
    )

    # -------------------------------------------------------------------------
    # Candidate-set evaluation
    # -------------------------------------------------------------------------

    candidate_hit = hit_at_k(
        candidates,
        relevant_sources,
        candidate_k,
    )

    candidate_recall = recall_at_k(
        candidates,
        relevant_sources,
        candidate_k,
    )

    candidate_rr = reciprocal_rank(
        candidates,
        relevant_sources,
        candidate_k,
    )

    candidate_first_relevant_rank = (
        find_first_relevant_rank(
            candidates,
            relevant_sources,
            candidate_k,
        )
    )

    # =========================================================================
    # BASELINE TOP N
    # =========================================================================
    #
    # Fair reranker comparison:
    #
    # baseline first 3
    # vs
    # reranked first 3
    #
    # NOT baseline Top 10 versus reranked Top 3.
    # =========================================================================

    baseline_top = candidates[
        :rerank_k
    ]

    baseline_hit = hit_at_k(
        baseline_top,
        relevant_sources,
        rerank_k,
    )

    baseline_rr = reciprocal_rank(
        baseline_top,
        relevant_sources,
        rerank_k,
    )

    baseline_precision = precision_at_k(
        baseline_top,
        relevant_sources,
        rerank_k,
    )

    baseline_ndcg = ndcg_at_k(
        baseline_top,
        relevant_sources,
        rerank_k,
    )

    baseline_first_relevant_rank = (
        find_first_relevant_rank(
            baseline_top,
            relevant_sources,
            rerank_k,
        )
    )

    # =========================================================================
    # STAGE 2: CROSS-ENCODER RERANKING
    # =========================================================================
    #
    # rerank_candidates() is your existing production reranker interface.
    #
    # It takes the exact candidates returned above.
    # =========================================================================

    reranked_wrappers = (
        reranker.rerank_candidates(
            query=query,
            candidates=candidates,
            top_n=rerank_k,
        )
    )

    # RerankedCandidate wraps the original retrieval result.
    #
    # Metrics should evaluate those original result objects.
    reranked_candidates = [
        item.candidate
        for item in reranked_wrappers
    ]

    reranked_hit = hit_at_k(
        reranked_candidates,
        relevant_sources,
        rerank_k,
    )

    reranked_rr = reciprocal_rank(
        reranked_candidates,
        relevant_sources,
        rerank_k,
    )

    reranked_precision = precision_at_k(
        reranked_candidates,
        relevant_sources,
        rerank_k,
    )

    reranked_ndcg = ndcg_at_k(
        reranked_candidates,
        relevant_sources,
        rerank_k,
    )

    reranked_first_relevant_rank = (
        find_first_relevant_rank(
            reranked_candidates,
            relevant_sources,
            rerank_k,
        )
    )

    # =========================================================================
    # BASELINE TRACE
    # =========================================================================

    baseline_trace = [
        serialize_candidate(
            candidate,
            rank=rank,
            relevant_sources=(
                relevant_sources
            ),
        )
        for rank, candidate in enumerate(
            candidates,
            start=1,
        )
    ]

    # =========================================================================
    # RERANK TRACE
    # =========================================================================
    #
    # This is especially important for Q002:
    #
    # retrieval rank 6
    #       ↓
    # rerank rank ?
    #
    # It tells us whether the cross encoder actually improved ordering.
    # =========================================================================

    rerank_trace: list[
        dict[str, Any]
    ] = []

    for item in reranked_wrappers:

        candidate = item.candidate

        rerank_trace.append(
            {
                "retrieval_rank": (
                    item.retrieval_rank
                ),
                "rerank_rank": (
                    item.rerank_rank
                ),
                "rerank_score": round(
                    float(
                        item.rerank_score
                    ),
                    6,
                ),
                "relevant": (
                    source_matches_gold(
                        candidate,
                        relevant_sources,
                    )
                ),
                "point_id": str(
                    get_value(
                        candidate,
                        "point_id",
                        "",
                    )
                ),
                "document_id": get_value(
                    candidate,
                    "document_id",
                ),
                "source_file": get_value(
                    candidate,
                    "source_file",
                ),
                "page_number": get_value(
                    candidate,
                    "page_number",
                ),
                "page_chunk_index": (
                    get_value(
                        candidate,
                        "page_chunk_index",
                    )
                ),
            }
        )

    return {
        "id": query_id,
        "query": query,
        "expected_answer_status": (
            golden_query.get(
                "expected_answer_status"
            )
        ),
        "relevant_sources": (
            relevant_sources
        ),

        "candidate_retrieval": {
            "k": candidate_k,
            "first_relevant_rank": (
                candidate_first_relevant_rank
            ),
            "hit": round(
                candidate_hit,
                4,
            ),
            "recall": round(
                candidate_recall,
                4,
            ),
            "reciprocal_rank": round(
                candidate_rr,
                4,
            ),
        },

        "baseline_top_n": {
            "k": rerank_k,
            "first_relevant_rank": (
                baseline_first_relevant_rank
            ),
            "hit": round(
                baseline_hit,
                4,
            ),
            "reciprocal_rank": round(
                baseline_rr,
                4,
            ),
            "precision": round(
                baseline_precision,
                4,
            ),
            "ndcg": round(
                baseline_ndcg,
                4,
            ),
        },

        "reranked_top_n": {
            "k": rerank_k,
            "first_relevant_rank": (
                reranked_first_relevant_rank
            ),
            "hit": round(
                reranked_hit,
                4,
            ),
            "reciprocal_rank": round(
                reranked_rr,
                4,
            ),
            "precision": round(
                reranked_precision,
                4,
            ),
            "ndcg": round(
                reranked_ndcg,
                4,
            ),
        },

        "baseline_trace": (
            baseline_trace
        ),

        "rerank_trace": (
            rerank_trace
        ),
    }


# =============================================================================
# COMPLETE DATASET EVALUATION
# =============================================================================


def evaluate_dataset(
    *,
    retriever: ProductionRetriever,
    reranker: CrossEncoderReranker,
    golden_queries: list[
        dict[str, Any]
    ],
    candidate_k: int,
    rerank_k: int,
) -> dict[str, Any]:
    """
    Run baseline + reranked evaluation
    across the complete golden dataset.
    """

    query_results: list[
        dict[str, Any]
    ] = []

    for golden_query in golden_queries:

        print(
            "\n"
            + "=" * 78
        )

        print(
            f"Evaluating: "
            f"{golden_query['id']}"
        )

        print(
            f"Query: "
            f"{golden_query['query']}"
        )

        result = evaluate_query(
            retriever=retriever,
            reranker=reranker,
            golden_query=golden_query,
            candidate_k=candidate_k,
            rerank_k=rerank_k,
        )

        query_results.append(
            result
        )

        candidate = result[
            "candidate_retrieval"
        ]

        baseline = result[
            "baseline_top_n"
        ]

        reranked = result[
            "reranked_top_n"
        ]

        print(
            "\nCandidate Retrieval"
        )

        print(
            "-" * 40
        )

        print(
            f"Relevant rank        : "
            f"{candidate['first_relevant_rank']}"
        )

        print(
            f"Hit@{candidate_k}              : "
            f"{candidate['hit']}"
        )

        print(
            f"Recall@{candidate_k}           : "
            f"{candidate['recall']}"
        )

        print(
            f"RR@{candidate_k}               : "
            f"{candidate['reciprocal_rank']}"
        )

        print(
            "\nBaseline Top "
            f"{rerank_k}"
        )

        print(
            "-" * 40
        )

        print(
            f"Relevant rank        : "
            f"{baseline['first_relevant_rank']}"
        )

        print(
            f"Hit@{rerank_k}               : "
            f"{baseline['hit']}"
        )

        print(
            f"RR@{rerank_k}                : "
            f"{baseline['reciprocal_rank']}"
        )

        print(
            f"Precision@{rerank_k}         : "
            f"{baseline['precision']}"
        )

        print(
            f"nDCG@{rerank_k}              : "
            f"{baseline['ndcg']}"
        )

        print(
            "\nReranked Top "
            f"{rerank_k}"
        )

        print(
            "-" * 40
        )

        print(
            f"Relevant rank        : "
            f"{reranked['first_relevant_rank']}"
        )

        print(
            f"Hit@{rerank_k}               : "
            f"{reranked['hit']}"
        )

        print(
            f"RR@{rerank_k}                : "
            f"{reranked['reciprocal_rank']}"
        )

        print(
            f"Precision@{rerank_k}         : "
            f"{reranked['precision']}"
        )

        print(
            f"nDCG@{rerank_k}              : "
            f"{reranked['ndcg']}"
        )

        print(
            "\nRerank movement"
        )

        print(
            "-" * 40
        )

        for trace in result[
            "rerank_trace"
        ]:

            marker = (
                " <-- RELEVANT"
                if trace["relevant"]
                else ""
            )

            print(
                f"Retriever Rank "
                f"{trace['retrieval_rank']:>2}"
                f" -> "
                f"Rerank "
                f"{trace['rerank_rank']:>2}"
                f" | score="
                f"{trace['rerank_score']:.6f}"
                f" | "
                f"{trace['document_id']}"
                f" p{trace['page_number']}"
                f"{marker}"
            )

    # =========================================================================
    # DATASET AGGREGATION
    # =========================================================================

    candidate_hits = [
        item[
            "candidate_retrieval"
        ]["hit"]
        for item in query_results
    ]

    candidate_recalls = [
        item[
            "candidate_retrieval"
        ]["recall"]
        for item in query_results
    ]

    candidate_rrs = [
        item[
            "candidate_retrieval"
        ]["reciprocal_rank"]
        for item in query_results
    ]

    baseline_hits = [
        item[
            "baseline_top_n"
        ]["hit"]
        for item in query_results
    ]

    baseline_rrs = [
        item[
            "baseline_top_n"
        ]["reciprocal_rank"]
        for item in query_results
    ]

    baseline_precisions = [
        item[
            "baseline_top_n"
        ]["precision"]
        for item in query_results
    ]

    baseline_ndcgs = [
        item[
            "baseline_top_n"
        ]["ndcg"]
        for item in query_results
    ]

    reranked_hits = [
        item[
            "reranked_top_n"
        ]["hit"]
        for item in query_results
    ]

    reranked_rrs = [
        item[
            "reranked_top_n"
        ]["reciprocal_rank"]
        for item in query_results
    ]

    reranked_precisions = [
        item[
            "reranked_top_n"
        ]["precision"]
        for item in query_results
    ]

    reranked_ndcgs = [
        item[
            "reranked_top_n"
        ]["ndcg"]
        for item in query_results
    ]

    candidate_summary = {
        f"hit_at_{candidate_k}": round(
            mean(candidate_hits),
            4,
        ),
        f"recall_at_{candidate_k}": round(
            mean(candidate_recalls),
            4,
        ),
        f"mrr_at_{candidate_k}": round(
            mean(candidate_rrs),
            4,
        ),
    }

    baseline_summary = {
        f"hit_at_{rerank_k}": round(
            mean(baseline_hits),
            4,
        ),
        f"mrr_at_{rerank_k}": round(
            mean(baseline_rrs),
            4,
        ),
        f"precision_at_{rerank_k}": round(
            mean(
                baseline_precisions
            ),
            4,
        ),
        f"ndcg_at_{rerank_k}": round(
            mean(
                baseline_ndcgs
            ),
            4,
        ),
    }

    reranked_summary = {
        f"hit_at_{rerank_k}": round(
            mean(reranked_hits),
            4,
        ),
        f"mrr_at_{rerank_k}": round(
            mean(reranked_rrs),
            4,
        ),
        f"precision_at_{rerank_k}": round(
            mean(
                reranked_precisions
            ),
            4,
        ),
        f"ndcg_at_{rerank_k}": round(
            mean(
                reranked_ndcgs
            ),
            4,
        ),
    }

    improvement = {
        f"hit_at_{rerank_k}_delta": round(
            reranked_summary[
                f"hit_at_{rerank_k}"
            ]
            - baseline_summary[
                f"hit_at_{rerank_k}"
            ],
            4,
        ),

        f"mrr_at_{rerank_k}_delta": round(
            reranked_summary[
                f"mrr_at_{rerank_k}"
            ]
            - baseline_summary[
                f"mrr_at_{rerank_k}"
            ],
            4,
        ),

        f"precision_at_{rerank_k}_delta": round(
            reranked_summary[
                f"precision_at_{rerank_k}"
            ]
            - baseline_summary[
                f"precision_at_{rerank_k}"
            ],
            4,
        ),

        f"ndcg_at_{rerank_k}_delta": round(
            reranked_summary[
                f"ndcg_at_{rerank_k}"
            ]
            - baseline_summary[
                f"ndcg_at_{rerank_k}"
            ],
            4,
        ),
    }

    return {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "evaluation_type": (
            "baseline_vs_reranked_retrieval"
        ),

        "configuration": {
            "candidate_k": (
                candidate_k
            ),
            "rerank_k": (
                rerank_k
            ),
            "query_count": len(
                query_results
            ),
            "embedding_model": (
                "BAAI/bge-m3"
            ),
            "reranker_model": (
                "BAAI/bge-reranker-v2-m3"
            ),
        },

        "candidate_retrieval": (
            candidate_summary
        ),

        "baseline_top_n": (
            baseline_summary
        ),

        "reranked_top_n": (
            reranked_summary
        ),

        "improvement": improvement,

        "queries": query_results,
    }


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Insurance RAG baseline "
            "retrieval versus cross-encoder "
            "reranked retrieval."
        )
    )

    parser.add_argument(
        "--golden-file",
        type=Path,
        default=Path(
            "data/evaluation/"
            "golden_queries.jsonl"
        ),
    )

    # -------------------------------------------------------------------------
    # Keep --top-k as an alias so your existing command still works:
    #
    # python -m src.evaluation.evaluator --top-k 10
    #
    # New clearer syntax is:
    #
    # --candidate-k 10
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--candidate-k",
        "--top-k",
        dest="candidate_k",
        type=int,
        default=10,
        help=(
            "Number of first-stage retrieval "
            "candidates."
        ),
    )

    parser.add_argument(
        "--rerank-k",
        type=int,
        default=3,
        help=(
            "Number of candidates retained "
            "after cross-encoder reranking."
        ),
    )

    parser.add_argument(
        "--collection",
        default=(
            "insurance_policy_chunks_"
            "bge_m3_v1"
        ),
    )

    parser.add_argument(
        "--host",
        default="localhost",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=6333,
    )

    parser.add_argument(
        "--vector-name",
        default="dense",
    )

    parser.add_argument(
        "--text-field",
        default="text",
    )

    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/evaluation/"
            "retrieval_reranker_"
            "evaluation.json"
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================


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

    if args.candidate_k <= 0:
        raise ValueError(
            "candidate_k must be "
            "greater than zero."
        )

    if args.rerank_k <= 0:
        raise ValueError(
            "rerank_k must be "
            "greater than zero."
        )

    if (
        args.rerank_k
        > args.candidate_k
    ):
        raise ValueError(
            "rerank_k cannot be greater "
            "than candidate_k."
        )

    # =========================================================================
    # LOAD GOLDEN QUERIES
    # =========================================================================

    golden_queries = (
        load_golden_queries(
            args.golden_file
        )
    )

    print(
        f"Loaded "
        f"{len(golden_queries)} "
        "golden queries."
    )

    print(
        f"Candidate K = "
        f"{args.candidate_k}"
    )

    print(
        f"Rerank K    = "
        f"{args.rerank_k}"
    )

    # =========================================================================
    # QDRANT
    # =========================================================================

    qdrant_client = QdrantClient(
        host=args.host,
        port=args.port,
        timeout=60,
    )

    try:

        # =====================================================================
        # EXISTING BGE-M3 QUERY ENCODER
        # =====================================================================

        query_encoder = (
            BgeM3QueryEncoder(
                model_name=(
                    "BAAI/bge-m3"
                ),
                device="cpu",
                use_fp16=False,
                max_length=512,
            )
        )

        # =====================================================================
        # EXISTING PRODUCTION RETRIEVER
        # =====================================================================

        retriever = (
            ProductionRetriever(
                client=qdrant_client,
                collection_name=(
                    args.collection
                ),
                query_encoder=(
                    query_encoder
                ),
                vector_name=(
                    args.vector_name
                ),
                text_field=(
                    args.text_field
                ),
                duplicate_threshold=(
                    args.duplicate_threshold
                ),
            )
        )

        # =====================================================================
        # EXISTING CROSS-ENCODER RERANKER
        # =====================================================================
        #
        # The CrossEncoderReranker lazily loads its model on the
        # first reranking request, so it is instantiated only once
        # and reused for all golden queries.
        # =====================================================================

        reranker = (CrossEncoderReranker(RerankerConfig(
            model_name=(
                "BAAI/bge-reranker-v2-m3"
            ),
            device="cpu",
            batch_size=4,
            max_length=512,
            normalize_scores=True,
        )))

        # =====================================================================
        # EVALUATE
        # =====================================================================

        report = evaluate_dataset(
            retriever=retriever,
            reranker=reranker,
            golden_queries=(
                golden_queries
            ),
            candidate_k=(
                args.candidate_k
            ),
            rerank_k=(
                args.rerank_k
            ),
        )

        # =====================================================================
        # SAVE REPORT
        # =====================================================================

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # =====================================================================
        # FINAL COMPARISON
        # =====================================================================

        candidate_summary = (
            report[
                "candidate_retrieval"
            ]
        )

        baseline_summary = (
            report[
                "baseline_top_n"
            ]
        )

        reranked_summary = (
            report[
                "reranked_top_n"
            ]
        )

        improvement = report[
            "improvement"
        ]

        print(
            "\n"
            + "=" * 78
        )

        print(
            "FINAL RETRIEVAL "
            "EVALUATION"
        )

        print(
            "=" * 78
        )

        print(
            f"Queries: "
            f"{len(golden_queries)}"
        )

        print(
            "\nFIRST-STAGE CANDIDATE "
            "RETRIEVAL"
        )

        print(
            "-" * 50
        )

        print(
            f"Hit@{args.candidate_k}       : "
            f"{candidate_summary[
                f'hit_at_{args.candidate_k}'
            ]}"
        )

        print(
            f"Recall@{args.candidate_k}    : "
            f"{candidate_summary[
                f'recall_at_{args.candidate_k}'
            ]}"
        )

        print(
            f"MRR@{args.candidate_k}       : "
            f"{candidate_summary[
                f'mrr_at_{args.candidate_k}'
            ]}"
        )

        print(
            "\nBASELINE TOP "
            f"{args.rerank_k}"
        )

        print(
            "-" * 50
        )

        print(
            f"Hit@{args.rerank_k}       : "
            f"{baseline_summary[
                f'hit_at_{args.rerank_k}'
            ]}"
        )

        print(
            f"MRR@{args.rerank_k}       : "
            f"{baseline_summary[
                f'mrr_at_{args.rerank_k}'
            ]}"
        )

        print(
            f"Precision@{args.rerank_k} : "
            f"{baseline_summary[
                f'precision_at_{args.rerank_k}'
            ]}"
        )

        print(
            f"nDCG@{args.rerank_k}      : "
            f"{baseline_summary[
                f'ndcg_at_{args.rerank_k}'
            ]}"
        )

        print(
            "\nRERANKED TOP "
            f"{args.rerank_k}"
        )

        print(
            "-" * 50
        )

        print(
            f"Hit@{args.rerank_k}       : "
            f"{reranked_summary[
                f'hit_at_{args.rerank_k}'
            ]}"
        )

        print(
            f"MRR@{args.rerank_k}       : "
            f"{reranked_summary[
                f'mrr_at_{args.rerank_k}'
            ]}"
        )

        print(
            f"Precision@{args.rerank_k} : "
            f"{reranked_summary[
                f'precision_at_{args.rerank_k}'
            ]}"
        )

        print(
            f"nDCG@{args.rerank_k}      : "
            f"{reranked_summary[
                f'ndcg_at_{args.rerank_k}'
            ]}"
        )

        print(
            "\nIMPROVEMENT"
        )

        print(
            "-" * 50
        )

        print(
            f"Hit@{args.rerank_k} delta       : "
            f"{improvement[
                f'hit_at_{args.rerank_k}_delta'
            ]:+.4f}"
        )

        print(
            f"MRR@{args.rerank_k} delta       : "
            f"{improvement[
                f'mrr_at_{args.rerank_k}_delta'
            ]:+.4f}"
        )

        print(
            f"Precision@{args.rerank_k} delta : "
            f"{improvement[
                f'precision_at_{args.rerank_k}_delta'
            ]:+.4f}"
        )

        print(
            f"nDCG@{args.rerank_k} delta      : "
            f"{improvement[
                f'ndcg_at_{args.rerank_k}_delta'
            ]:+.4f}"
        )

        print(
            "\nReport saved to:"
        )

        print(
            args.output
        )

    finally:

        qdrant_client.close()


if __name__ == "__main__":
    main()