from __future__ import annotations

import re
import math
from typing import Any, Sequence

# ============================================================================
# TEXT NORMALIZATION

def normalize_text(text: str) -> str:
    """
    Normalize generated text before deterministic fact checking.
    WHY: punctuation, capitalization or hyphenation differences to create false evaluation failures.
    """

    text = text.lower()     # Treat hyphenated and non-hyphenated wording similarly.
    text = text.replace("-", " ") # Remove punctuation while retaining alphanumeric characters.
    text = re.sub( r"[^a-z0-9\s]", " ", text,) # Collapse repeated whitespace.
    text = re.sub( r"\s+", " ", text,)
    return text.strip()

# ============================================================================
# SOURCE MATCHING
# ============================================================================

def source_matches_gold(result: Any,relevant_sources: Sequence[dict[str, Any]],) -> bool:
    """
    Determine whether a retrieved result matches any gold source.
    Matching hierarchy:
        1. point_id
        2. document_id + page_number
    Exact point IDs provide the strongest regression signal. Document/page matching allows more than one acceptable chunk from the
    same policy page where overlapping chunking exists.
    """

    result_point_id = str(getattr(result, "point_id", ""))
    result_document_id = getattr(result,"document_id",None,)
    result_page_number = getattr(result,"page_number",None,)

    for source in relevant_sources:

        gold_point_id = source.get("point_id")

        if ( gold_point_id and result_point_id == str(gold_point_id)):
            return True

        gold_document_id = source.get("document_id")
        gold_page_number = source.get("page_number")

        if (
            gold_document_id is not None
            and gold_page_number is not None
            and result_document_id == gold_document_id
            and result_page_number == gold_page_number
        ):
            return True

    return False

# ============================================================================
# HIT@K

def hit_at_k(results: Sequence[Any], relevant_sources: Sequence[dict[str, Any]], k: int,) -> float:
    """
    Hit@K = 1 when at least one relevant result appears in the first K results.
    Otherwise = 0.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    for result in results[:k]:
        if source_matches_gold(result,relevant_sources,):
            return 1.0

    return 0.0


# ============================================================================
# RECIPROCAL RANK
# ============================================================================

def reciprocal_rank(results: Sequence[Any], relevant_sources: Sequence[dict[str, Any]], k: int | None = None,) -> float:
    """
    Reciprocal rank:
        Rank 1 relevant -> 1.0
        Rank 2 relevant -> 0.5
        Rank 3 relevant -> 0.333...
        No relevant result -> 0
    MRR is the mean reciprocal rank across all evaluation queries.
    """

    evaluation_results = (
        results[:k]
        if k is not None
        else results
    )

    for rank, result in enumerate(evaluation_results, start=1,):
        if source_matches_gold(result,relevant_sources,):
            return 1.0 / rank

    return 0.0


# ============================================================================
# CITATION CORRECTNESS
# 
def citation_correctness(
    generated_sources: Sequence[dict[str, Any]],
    relevant_sources: Sequence[dict[str, Any]],
) -> float:
    """
    Measure whether citations returned with the generated answer correspond
    to gold evidence.
    Citation correctness: correct cited sources / total cited sources
    Example:
        S1 -> correct page
        S2 -> irrelevant page
        correctness = 1 / 2 = 0.5
    """

    if not generated_sources:
        return 0.0

    correct_count = 0

    for generated_source in generated_sources:

        class SourceAdapter:
            pass

        adapter = SourceAdapter()

        adapter.point_id = generated_source.get("point_id")
        adapter.document_id = generated_source.get("document_id")
        adapter.page_number = generated_source.get("page_number")
        if source_matches_gold(
            adapter,
            relevant_sources,
        ):
            correct_count += 1

    return (correct_count/ len(generated_sources))


# ============================================================================
# REQUIRED FACT COVERAGE
# ============================================================================


def required_fact_score(
    answer: str,
    required_facts: Sequence[Sequence[str]],
) -> float:
    """
    Determine how many required facts are represented in the answer. Each required fact may contain multiple acceptable textual variants.
    Example:
        [
            ["on-piste skiing", "on piste skiing"],
            ["winter-sports extension", "winter sports extension"]
        ]
    If both concepts occur -> 1.0
    If only one occurs -> 0.5
    """

    if not required_facts:
        return 1.0

    normalized_answer = normalize_text( answer )

    matched = 0

    for alternatives in required_facts:
        alternative_found = False
        for alternative in alternatives:
            normalized_alternative = normalize_text( alternative)
            if ( normalized_alternative in normalized_answer):
                alternative_found = True
                break

        if alternative_found:
            matched += 1

    return matched / len(required_facts)


# ============================================================================
# FORBIDDEN CLAIM CHECK
# ============================================================================


def forbidden_claim_score(
    answer: str,
    forbidden_claims: Sequence[str],
) -> float:
    """
    Return 1 when no forbidden claim occurs.
    Return 0 when the answer contains a prohibited claim.
    This is especially useful in insurance because the assistant must not
    convert policy information into guaranteed claim approval.
    """

    normalized_answer = normalize_text( answer )

    for forbidden_claim in forbidden_claims:
        normalized_forbidden = normalize_text(forbidden_claim)
        if normalized_forbidden in normalized_answer:
            return 0.0

    return 1.0

def recall_at_k(
    results,
    relevant_sources,
    k: int,
) -> float:
    """
    Recall@K measures how many of the expected relevant sources
    were successfully retrieved within the first K results.

    Example:
        Expected relevant sources = 2
        Retrieved within Top 10 = 1

        Recall@10 = 1 / 2 = 0.5
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    if not relevant_sources:
        return 0.0

    matched_sources = 0

    for relevant_source in relevant_sources:

        found = any(
            source_matches_gold(
                result,
                [relevant_source],
            )
            for result in results[:k]
        )

        if found:
            matched_sources += 1

    return (
        matched_sources
        / len(relevant_sources)
    )

# ============================================================================
# ANSWER CORRECTNESS
# ============================================================================


def answer_correctness(
    *,
    actual_status: str,
    expected_status: str,
    answer: str,
    required_facts: Sequence[Sequence[str]],
    forbidden_claims: Sequence[str],
) -> dict[str, float]:
    """
    Deterministic answer-quality score.
    Components:
        40% business classification
        40% required factual coverage
        20% absence of prohibited claims
    This is intentionally auditable rather than relying exclusively on an LLM-as-judge. """

    status_score = float( actual_status == expected_status)
    fact_score = required_fact_score( answer, required_facts,)
    safety_score = forbidden_claim_score( answer, forbidden_claims,)
    overall_score = ( 0.40 * status_score + 0.40 * fact_score + 0.20 * safety_score)

    return {
        "answer_status_correct": status_score,
        "required_fact_score": round(
            fact_score,
            4,
        ),
        "forbidden_claim_score": safety_score,
        "answer_correctness": round(
            overall_score,
            4,
        ),
    }

def precision_at_k(
    results,
    relevant_sources,
    k: int,
) -> float:
    """
    Precision@K:

        relevant results in Top K
        -------------------------
                    K
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


def ndcg_at_k(
    results,
    relevant_sources,
    k: int,
) -> float:
    """
    Binary nDCG@K.

    Relevant result   -> relevance = 1
    Irrelevant result -> relevance = 0

    nDCG evaluates the quality of the complete ranking,
    not only the first relevant result.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    # ------------------------------------------------------------------
    # DCG
    # ------------------------------------------------------------------

    dcg = 0.0

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):
        relevance = 1.0 if source_matches_gold(
            result,
            relevant_sources,
        ) else 0.0

        if relevance > 0:
            dcg += relevance / math.log2(
                rank + 1
            )

    # ------------------------------------------------------------------
    # IDEAL DCG
    #
    # If there are 2 gold sources, ideal ranking would place:
    #
    # Rank 1 -> relevant
    # Rank 2 -> relevant
    # ------------------------------------------------------------------

    ideal_relevant_count = min(
        len(relevant_sources),
        k,
    )

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1,
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg