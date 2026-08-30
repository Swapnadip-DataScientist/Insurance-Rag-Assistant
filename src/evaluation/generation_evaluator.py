from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import httpx

from qdrant_client import QdrantClient

from src.retrieval.retriever import (
    BgeM3QueryEncoder,
    ProductionRetriever,
)

from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankerConfig,
)

from src.generation.generator import (
    GenerationConfig,
    GroundedGenerator,
    OllamaConfig,
    OllamaQwenClient,
)


LOGGER = logging.getLogger(__name__)


# =============================================================================
# JUDGE JSON SCHEMA
# =============================================================================
#
# The generation model already produces the insurance answer.
#
# This second structured call acts only as an evaluator.
#
# IMPORTANT:
# The judge must use ONLY:
#
#   user question
#   generated answer
#   cited policy evidence
#
# It must not use external insurance knowledge.
# =============================================================================


JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "groundedness_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 4,
        },
        "answer_correctness_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 4,
        },
        "unsupported_claims": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
            },
            "maxItems": 10,
        },
        "reason": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1500,
        },
    },
    "required": [
        "groundedness_score",
        "answer_correctness_score",
        "unsupported_claims",
        "reason",
    ],
}


JUDGE_SYSTEM_PROMPT = """
You are an evaluator for a grounded insurance RAG system.

You are NOT answering the insurance question.

Your only job is to evaluate the generated answer against the supplied
policy evidence.

RULES

1. Use only the supplied policy evidence.
2. Do not use outside insurance knowledge.
3. Do not infer facts that are absent from the evidence.
4. Evaluate every substantive statement in the generated answer.
5. Do not penalize cautious statements about limitations or uncertainty.
6. Do not treat a citation ID itself as proof. Check the cited text.
7. If a generated claim is not supported by the evidence, list it under
   unsupported_claims.
8. A claim that directly contradicts evidence is unsupported.
9. A claim that adds a limit, condition, exclusion, amount or guarantee that
   is absent from the evidence is unsupported.
10. Do not require identical wording. Judge semantic meaning.

GROUNDEDNESS SCORE

4 = Every substantive factual claim is supported by the evidence.
3 = Mostly grounded; one minor unsupported detail.
2 = Mixed; important parts are supported but meaningful unsupported claims exist.
1 = Mostly unsupported.
0 = Answer is contradicted by or essentially unsupported by the evidence.

ANSWER CORRECTNESS SCORE

4 = Correctly and directly answers the user's question from the evidence.
3 = Mostly correct with a minor omission or imprecision.
2 = Partly correct but misses an important condition or conclusion.
1 = Mostly incorrect.
0 = Incorrect or opposite to the evidence.

Return only the required JSON.
""".strip()


# =============================================================================
# GOLDEN QUERY LOADER
# =============================================================================


def load_golden_queries(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Load the same golden_queries.jsonl used by retrieval evaluation.
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
                    "contain one JSON object."
                )

            query_id = record.get("id")
            query = record.get("query")
            relevant_sources = record.get(
                "relevant_sources"
            )
            expected_status = record.get(
                "expected_answer_status"
            )

            if (
                not isinstance(query_id, str)
                or not query_id.strip()
            ):
                raise ValueError(
                    f"Invalid id at line "
                    f"{line_number}."
                )

            if (
                not isinstance(query, str)
                or not query.strip()
            ):
                raise ValueError(
                    f"Invalid query at line "
                    f"{line_number}."
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

            if (
                not isinstance(
                    expected_status,
                    str,
                )
                or not expected_status.strip()
            ):
                raise ValueError(
                    f"expected_answer_status "
                    f"is missing for {query_id}."
                )

            records.append(
                record
            )

    if not records:
        raise ValueError(
            "No golden queries found."
        )

    return records


# =============================================================================
# GENERATOR RETRIEVAL ADAPTER
# =============================================================================
#
# GroundedGenerator needs only:
#
#     retrieval_response.query
#     retrieval_response.results
#     retrieval_response.diagnostics
#
# Therefore we create a tiny adapter containing the RERANKED Top-3.
#
# We do NOT modify ProductionRetriever or GroundedGenerator.
# =============================================================================


@dataclass(frozen=True)
class GeneratorRetrievalAdapter:
    query: str
    results: tuple[Any, ...]
    diagnostics: dict[str, Any]


# =============================================================================
# SAFE VALUE ACCESS
# =============================================================================


def get_value(
    item: Any,
    field_name: str,
    default: Any = None,
) -> Any:

    if isinstance(
        item,
        dict,
    ):

        if field_name in item:
            return item.get(
                field_name,
                default,
            )

        payload = item.get(
            "payload"
        )

        if isinstance(
            payload,
            dict,
        ):
            return payload.get(
                field_name,
                default,
            )

        return default

    value = getattr(
        item,
        field_name,
        None,
    )

    if value is not None:
        return value

    payload = getattr(
        item,
        "payload",
        None,
    )

    if isinstance(
        payload,
        dict,
    ):
        return payload.get(
            field_name,
            default,
        )

    return default


# =============================================================================
# GOLD SOURCE MATCHING
# =============================================================================


def source_matches_gold(
    source: Any,
    relevant_sources: Sequence[
        dict[str, Any]
    ],
) -> bool:
    """
    Matching rules:

    If golden source contains point_id:
        exact point_id match ONLY.

    Otherwise:
        document_id + page_number.

    This is the same strict behaviour we introduced in retrieval evaluation.
    """

    source_point_id = get_value(
        source,
        "point_id",
    )

    source_document_id = get_value(
        source,
        "document_id",
    )

    source_page_number = get_value(
        source,
        "page_number",
    )

    for gold_source in relevant_sources:

        gold_point_id = gold_source.get(
            "point_id"
        )

        # ---------------------------------------------------------------------
        # STRICT CHUNK MATCH
        # ---------------------------------------------------------------------

        if gold_point_id is not None:

            if (
                source_point_id is not None
                and str(source_point_id)
                == str(gold_point_id)
            ):
                return True

            continue

        # ---------------------------------------------------------------------
        # PAGE MATCH
        # ---------------------------------------------------------------------

        gold_document_id = (
            gold_source.get(
                "document_id"
            )
        )

        gold_page_number = (
            gold_source.get(
                "page_number"
            )
        )

        if (
            gold_document_id is not None
            and gold_page_number is not None
            and source_document_id
            == gold_document_id
            and source_page_number
            == gold_page_number
        ):
            return True

    return False


# =============================================================================
# CITATION CORRECTNESS
# =============================================================================


def evaluate_citations(
    *,
    cited_sources: Sequence[Any],
    relevant_sources: Sequence[
        dict[str, Any]
    ],
    answer_status: str,
) -> dict[str, Any]:
    """
    Citation correctness answers:

        Of the sources Qwen cited,
        how many are actually gold evidence?

    We also calculate citation recall:

        Of all expected gold sources,
        how many were cited?
    """

    if not cited_sources:

        # No citation is legitimate for insufficient evidence.
        if answer_status == (
            "insufficient_evidence"
        ):
            return {
                "citation_correctness": None,
                "citation_recall": None,
                "correct_citations": 0,
                "total_citations": 0,
            }

        return {
            "citation_correctness": 0.0,
            "citation_recall": 0.0,
            "correct_citations": 0,
            "total_citations": 0,
        }

    correct_count = sum(
        1
        for source in cited_sources
        if source_matches_gold(
            source,
            relevant_sources,
        )
    )

    citation_correctness = (
        correct_count
        / len(cited_sources)
    )

    # -------------------------------------------------------------------------
    # GOLD SOURCE COVERAGE
    # -------------------------------------------------------------------------

    matched_gold_count = 0

    for gold_source in relevant_sources:

        found = any(
            source_matches_gold(
                source,
                [gold_source],
            )
            for source in cited_sources
        )

        if found:
            matched_gold_count += 1

    citation_recall = (
        matched_gold_count
        / len(relevant_sources)
    )

    return {
        "citation_correctness": round(
            citation_correctness,
            4,
        ),
        "citation_recall": round(
            citation_recall,
            4,
        ),
        "correct_citations": (
            correct_count
        ),
        "total_citations": len(
            cited_sources
        ),
    }


# =============================================================================
# LOCAL SEMANTIC JUDGE
# =============================================================================


class LocalGenerationJudge:
    """
    Separate evaluation call through local Ollama.

    This does NOT modify generator.py.

    It evaluates:
        groundedness
        answer correctness
        unsupported claims
    """

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        timeout_seconds: float,
    ) -> None:

        self.model_name = (
            model_name
        )

        self.base_url = (
            base_url.rstrip("/")
        )

        self.client = httpx.Client(
            timeout=httpx.Timeout(
                connect=5.0,
                read=timeout_seconds,
                write=30.0,
                pool=5.0,
            )
        )

    def close(
        self,
    ) -> None:

        self.client.close()

    def __enter__(
        self,
    ) -> "LocalGenerationJudge":

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        self.close()

    def evaluate(
        self,
        *,
        query: str,
        generated_answer: str,
        answer_status: str,
        expected_answer_status: str,
        cited_sources: Sequence[Any],
    ) -> dict[str, Any]:

        evidence_blocks: list[str] = []

        for index, source in enumerate(
            cited_sources,
            start=1,
        ):

            evidence_blocks.append(
                "\n".join(
                    [
                        (
                            f"<EVIDENCE_"
                            f"{index}_START>"
                        ),
                        (
                            "Document: "
                            f"{get_value(
                                source,
                                'document_id',
                                'unknown'
                            )}"
                        ),
                        (
                            "Page: "
                            f"{get_value(
                                source,
                                'page_number',
                                'unknown'
                            )}"
                        ),
                        "<TEXT>",
                        str(
                            get_value(
                                source,
                                "text",
                                "",
                            )
                        ),
                        "</TEXT>",
                        (
                            f"<EVIDENCE_"
                            f"{index}_END>"
                        ),
                    ]
                )
            )

        if evidence_blocks:
            evidence_text = (
                "\n\n".join(
                    evidence_blocks
                )
            )

        else:
            evidence_text = (
                "NO CITED POLICY EVIDENCE"
            )

        user_prompt = "\n\n".join(
            [
                "<USER_QUESTION>",
                query,
                "</USER_QUESTION>",
                "<EXPECTED_STATUS>",
                expected_answer_status,
                "</EXPECTED_STATUS>",
                "<GENERATED_STATUS>",
                answer_status,
                "</GENERATED_STATUS>",
                "<GENERATED_ANSWER>",
                generated_answer,
                "</GENERATED_ANSWER>",
                "<CITED_POLICY_EVIDENCE>",
                evidence_text,
                "</CITED_POLICY_EVIDENCE>",
                (
                    "Evaluate the generated "
                    "answer now."
                ),
            ]
        )

        request_payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        JUDGE_SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "think": False,
            "format": JUDGE_JSON_SCHEMA,
            "options": {
                "temperature": 0.0,
                "seed": 42,
                "num_ctx": 8192,
                "num_predict": 700,
            },
            "keep_alive": "5m",
        }

        response = self.client.post(
            f"{self.base_url}/api/chat",
            json=request_payload,
        )

        response.raise_for_status()

        response_body = (
            response.json()
        )

        message = response_body.get(
            "message"
        )

        if not isinstance(
            message,
            dict,
        ):
            raise RuntimeError(
                "Judge returned no message."
            )

        raw_content = message.get(
            "content"
        )

        if (
            not isinstance(
                raw_content,
                str,
            )
            or not raw_content.strip()
        ):
            raise RuntimeError(
                "Judge returned empty output."
            )

        parsed = json.loads(
            raw_content
        )

        groundedness_raw = int(
            parsed[
                "groundedness_score"
            ]
        )

        correctness_raw = int(
            parsed[
                "answer_correctness_score"
            ]
        )

        unsupported_claims = parsed[
            "unsupported_claims"
        ]

        if not isinstance(
            unsupported_claims,
            list,
        ):
            raise RuntimeError(
                "Judge unsupported_claims "
                "must be a list."
            )

        return {
            # Convert 0..4 into 0..1.
            "groundedness": round(
                groundedness_raw / 4.0,
                4,
            ),

            "answer_correctness": round(
                correctness_raw / 4.0,
                4,
            ),

            "unsupported_claim_count": len(
                unsupported_claims
            ),

            # 1 = clean answer
            # 0 = at least one unsupported claim
            "unsupported_claim_free": (
                1.0
                if not unsupported_claims
                else 0.0
            ),

            "unsupported_claims": (
                unsupported_claims
            ),

            "judge_reason": (
                parsed["reason"]
            ),
        }


# =============================================================================
# ONE QUERY
# =============================================================================


def evaluate_one_query(
    *,
    retriever: ProductionRetriever,
    reranker: CrossEncoderReranker,
    generator: GroundedGenerator,
    judge: LocalGenerationJudge,
    golden_query: dict[str, Any],
    candidate_k: int,
    rerank_k: int,
) -> dict[str, Any]:

    query_id = golden_query[
        "id"
    ]

    query = golden_query[
        "query"
    ]

    relevant_sources = golden_query[
        "relevant_sources"
    ]

    expected_status = golden_query[
        "expected_answer_status"
    ]

    # =========================================================================
    # 1. FIRST-STAGE RETRIEVAL
    # =========================================================================

    retrieval_response = (
        retriever.retrieve(
            query,
            top_k=candidate_k,
        )
    )

    candidates = list(
        retrieval_response.results
    )

    # =========================================================================
    # 2. BGE RERANKING
    # =========================================================================

    reranked_wrappers = (
        reranker.rerank_candidates(
            query=query,
            candidates=candidates,
            top_n=rerank_k,
        )
    )

    reranked_candidates = tuple(
        item.candidate
        for item in reranked_wrappers
    )

    rerank_trace = [
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
            "point_id": str(
                get_value(
                    item.candidate,
                    "point_id",
                    "",
                )
            ),
            "document_id": get_value(
                item.candidate,
                "document_id",
            ),
            "page_number": get_value(
                item.candidate,
                "page_number",
            ),
            "relevant": (
                source_matches_gold(
                    item.candidate,
                    relevant_sources,
                )
            ),
        }
        for item in reranked_wrappers
    ]

    # =========================================================================
    # 3. ADAPT RERANKED TOP-3 FOR EXISTING GENERATOR
    # =========================================================================

    generator_input = (
        GeneratorRetrievalAdapter(
            query=query,
            results=(
                reranked_candidates
            ),
            diagnostics={
                **dict(
                    retrieval_response
                    .diagnostics
                ),
                "evaluation_reranked": True,
                "candidate_k": (
                    candidate_k
                ),
                "rerank_k": (
                    rerank_k
                ),
                "reranker_model": (
                    "BAAI/"
                    "bge-reranker-v2-m3"
                ),
            },
        )
    )

    # =========================================================================
    # 4. EXISTING QWEN GENERATOR
    # =========================================================================

    generation_response = (
        generator.generate(
            generator_input
        )
    )

    generated = (
        generation_response.to_dict()
    )

    actual_status = generated[
        "answer_status"
    ]

    answer = generated[
        "answer"
    ]

    # =========================================================================
    # 5. ANSWER-STATUS CORRECTNESS
    # =========================================================================

    status_correctness = float(
        actual_status
        == expected_status
    )

    # =========================================================================
    # 6. CITATION CORRECTNESS
    # =========================================================================

    citation_result = (
        evaluate_citations(
            cited_sources=(
                generation_response
                .sources
            ),
            relevant_sources=(
                relevant_sources
            ),
            answer_status=(
                actual_status
            ),
        )
    )

    # =========================================================================
    # 7. SEMANTIC QUALITY JUDGE
    # =========================================================================

    judge_result = (
        judge.evaluate(
            query=query,
            generated_answer=answer,
            answer_status=(
                actual_status
            ),
            expected_answer_status=(
                expected_status
            ),
            cited_sources=(
                generation_response
                .sources
            ),
        )
    )

    print("\nDEBUG JUDGE RESULT:")
    print(judge_result)

    unsupported_claims = judge_result.get("unsupported_claims",[])

    return {
        "id": query_id,
        "query": query,

        "expected": {
            "answer_status": (
                expected_status
            ),
            "relevant_sources": (
                relevant_sources
            ),
        },

        "generated": {
            "answer_status": (
                actual_status
            ),
            "answer": answer,
            "citations": generated[
                "citations"
            ],
            "conditions": generated[
                "conditions"
            ],
            "limitations": generated[
                "limitations"
            ],
            "sources": generated[
                "sources"
            ],
        },

        

        "metrics": {
            "citation_correctness": (
                citation_result[
                    "citation_correctness"
                ]
            ),
            "citation_recall": (
                citation_result[
                    "citation_recall"
                ]
            ),
            "answer_status_correctness": (
                status_correctness
            ),
            "groundedness": (
                judge_result[
                    "groundedness"
                ]
            ),
            "answer_correctness": (
                judge_result[
                    "answer_correctness"
                ]
            ),
            "unsupported_claim_free": (
                judge_result[
                    "unsupported_claim_free"
                ]
            ),
            "unsupported_claims": unsupported_claims,

            "unsupported_claim_count": (
                judge_result[
                    "unsupported_claim_count"
                ]
            ),
        },

        "unsupported_claims": (
            judge_result[
                "unsupported_claims"
            ]
        ),

        "judge_reason": (
            judge_result[
                "judge_reason"
            ]
        ),

        "rerank_trace": (
            rerank_trace
        ),

        "generation_latency_ms": (
            generation_response
            .generation_latency_ms
        ),
    }


# =============================================================================
# SAFE MEAN
# =============================================================================


def safe_mean(
    values: Sequence[
        float | None
    ],
) -> float | None:

    valid = [
        float(value)
        for value in values
        if value is not None
    ]

    if not valid:
        return None

    return round(
        mean(valid),
        4,
    )


# =============================================================================
# DATASET EVALUATION
# =============================================================================


def evaluate_dataset(
    *,
    retriever: ProductionRetriever,
    reranker: CrossEncoderReranker,
    generator: GroundedGenerator,
    judge: LocalGenerationJudge,
    golden_queries: list[
        dict[str, Any]
    ],
    candidate_k: int,
    rerank_k: int,
) -> dict[str, Any]:

    query_results: list[
        dict[str, Any]
    ] = []

    for index, golden_query in enumerate(
        golden_queries,
        start=1,
    ):

        print(
            "\n"
            + "=" * 78
        )

        print(
            f"[{index}/"
            f"{len(golden_queries)}] "
            f"{golden_query['id']}"
        )

        print(
            f"Query: "
            f"{golden_query['query']}"
        )

        result = evaluate_one_query(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            judge=judge,
            golden_query=golden_query,
            candidate_k=candidate_k,
            rerank_k=rerank_k,
        )

        query_results.append(
            result
        )

        metrics = result[
            "metrics"
        ]

        generated = result[
            "generated"
        ]

        print(
            f"Expected status     : "
            f"{result['expected']['answer_status']}"
        )

        print(
            f"Generated status    : "
            f"{generated['answer_status']}"
        )

        print(
            f"Status correct      : "
            f"{metrics[
                'answer_status_correctness'
            ]}"
        )

        print(
            f"Citation correctness: "
            f"{metrics[
                'citation_correctness'
            ]}"
        )

        print(
            f"Groundedness        : "
            f"{metrics['groundedness']}"
        )

        print(
            f"Answer correctness  : "
            f"{metrics[
                'answer_correctness'
            ]}"
        )

        print(
            f"Unsupported claims  : "
            f"{metrics[
                'unsupported_claim_count'
            ]}"
        )

        unsupported_claims = metrics.get("unsupported_claims", [])

        if unsupported_claims:
            print("Unsupported claim details:")

            for i, claim in enumerate(unsupported_claims, start=1):
                print(f"  {i}. {claim}")
        else:
            print("Unsupported claim details: None")


        print(
            f"Answer              : "
            f"{generated['answer']}"
        )

    # =========================================================================
    # AGGREGATE
    # =========================================================================

    summary = {
        "query_count": len(
            query_results
        ),

        "citation_correctness": (
            safe_mean(
                [
                    item["metrics"][
                        "citation_correctness"
                    ]
                    for item
                    in query_results
                ]
            )
        ),

        "citation_recall": (
            safe_mean(
                [
                    item["metrics"][
                        "citation_recall"
                    ]
                    for item
                    in query_results
                ]
            )
        ),

        "answer_status_accuracy": (
            safe_mean(
                [
                    item["metrics"][
                        "answer_status_correctness"
                    ]
                    for item
                    in query_results
                ]
            )
        ),

        "groundedness": (
            safe_mean(
                [
                    item["metrics"][
                        "groundedness"
                    ]
                    for item
                    in query_results
                ]
            )
        ),

        "answer_correctness": (
            safe_mean(
                [
                    item["metrics"][
                        "answer_correctness"
                    ]
                    for item
                    in query_results
                ]
            )
        ),

        "unsupported_claim_free_rate": (
            safe_mean(
                [
                    item["metrics"][
                        "unsupported_claim_free"
                    ]
                    for item
                    in query_results
                ]
            )
        ),

        "mean_generation_latency_ms": (
            safe_mean(
                [
                    item[
                        "generation_latency_ms"
                    ]
                    for item
                    in query_results
                ]
            )
        ),
    }

    return {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "evaluation_type": (
            "grounded_generation"
        ),

        "configuration": {
            "embedding_model": (
                "BAAI/bge-m3"
            ),
            "reranker_model": (
                "BAAI/"
                "bge-reranker-v2-m3"
            ),
            "generator_model": (
                generator
                .llm_client
                .config
                .model_name
            ),
            "candidate_k": (
                candidate_k
            ),
            "rerank_k": (
                rerank_k
            ),
        },

        "summary": summary,

        "queries": (
            query_results
        ),
    }


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Insurance RAG "
            "grounded generation."
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

    parser.add_argument(
        "--candidate-k",
        "--top-k",
        dest="candidate_k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--rerank-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of golden "
            "queries to test."
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
        "--ollama-model",
        default=(
            "qwen3.5:4b-q4_K_M"
        ),
    )

    parser.add_argument(
        "--ollama-base-url",
        default=(
            "http://127.0.0.1:11434"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/evaluation/"
            "generation_evaluation.json"
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

    golden_queries = (
        load_golden_queries(
            args.golden_file
        )
    )

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be "
                "greater than zero."
            )

        golden_queries = (
            golden_queries[
                :args.limit
            ]
        )

    print(
        f"Loaded "
        f"{len(golden_queries)} "
        "golden queries."
    )

    qdrant_client = QdrantClient(
        host=args.host,
        port=args.port,
        timeout=60,
    )

    try:

        # =====================================================================
        # EXISTING QUERY ENCODER
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
        # EXISTING RETRIEVER
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
                vector_name="dense",
                text_field="text",
                duplicate_threshold=0.85,
            )
        )

        # =====================================================================
        # APPROVED BGE RERANKER
        # =====================================================================

        reranker = (
            CrossEncoderReranker(
                RerankerConfig(
                    model_name=(
                        "BAAI/"
                        "bge-reranker-v2-m3"
                    ),
                    device="cpu",
                    batch_size=4,
                    max_length=512,
                    normalize_scores=True,
                )
            )
        )

        # =====================================================================
        # EXISTING QWEN GENERATOR
        # =====================================================================

        ollama_config = (
            OllamaConfig(
                model_name=(
                    args.ollama_model
                ),
                base_url=(
                    args.ollama_base_url
                ),
                read_timeout_seconds=(
                    args.timeout
                ),
                temperature=0.1,
                seed=42,
                num_ctx=8192,
                num_predict=1000,
                keep_alive="5m",
            )
        )

        with OllamaQwenClient(
            ollama_config
        ) as llm_client:

            # Fail early if Ollama/Qwen
            # is unavailable.
            llm_client.ensure_model_available()

            generator = (
                GroundedGenerator(
                    llm_client=(
                        llm_client
                    ),
                    config=(
                        GenerationConfig(
                            max_evidence_chunks=(
                                args.rerank_k
                            ),
                            max_chars_per_chunk=3000,
                            max_total_evidence_chars=9000,
                        )
                    ),
                )
            )

            # ================================================================
            # LOCAL JUDGE
            # ================================================================

            with LocalGenerationJudge(
                model_name=(
                    args.ollama_model
                ),
                base_url=(
                    args.ollama_base_url
                ),
                timeout_seconds=(
                    args.timeout
                ),
            ) as judge:

                report = (
                    evaluate_dataset(
                        retriever=(
                            retriever
                        ),
                        reranker=(
                            reranker
                        ),
                        generator=(
                            generator
                        ),
                        judge=judge,
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
                allow_nan=False,
            )

        summary = report[
            "summary"
        ]

        print(
            "\n"
            + "=" * 78
        )

        print(
            "FINAL GENERATION "
            "EVALUATION"
        )

        print(
            "=" * 78
        )

        print(
            f"Queries                     : "
            f"{summary['query_count']}"
        )

        print(
            f"Citation correctness        : "
            f"{summary[
                'citation_correctness'
            ]}"
        )

        print(
            f"Citation recall             : "
            f"{summary[
                'citation_recall'
            ]}"
        )

        print(
            f"Answer-status accuracy      : "
            f"{summary[
                'answer_status_accuracy'
            ]}"
        )

        print(
            f"Groundedness                : "
            f"{summary[
                'groundedness'
            ]}"
        )

        print(
            f"Answer correctness          : "
            f"{summary[
                'answer_correctness'
            ]}"
        )

        print(
            f"Unsupported-claim-free rate : "
            f"{summary[
                'unsupported_claim_free_rate'
            ]}"
        )

        print(
            f"Mean generation latency ms  : "
            f"{summary[
                'mean_generation_latency_ms'
            ]}"
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