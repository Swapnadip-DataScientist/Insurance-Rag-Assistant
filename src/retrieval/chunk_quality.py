from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkQualityResult:
    is_valid: bool
    reasons: tuple[str, ...]
    raw_length: int
    stripped_length: int
    word_count: int
    printable_ratio: float
    alphanumeric_ratio: float

def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0

def validate_chunk_text(
    text: object,
    *,
    min_stripped_length: int = 20,
    min_word_count: int = 3,
    min_printable_ratio: float = 0.85,
    min_alphanumeric_ratio: float = 0.20,
) -> ChunkQualityResult:
    """
    Validate whether retrieved chunk text is usable policy content.

    This is intentionally conservative. It rejects clearly empty, control-only
    and severely garbled text, while allowing short but meaningful clauses.
    """

    reasons: list[str] = []

    if not isinstance(text, str):
        reasons.append("text_not_string")
        text = "" if text is None else str(text)

    stripped_text = text.strip()

    raw_length = len(text)
    stripped_length = len(stripped_text)
    words = re.findall(r"\b\w+\b", stripped_text, flags=re.UNICODE)
    word_count = len(words)

    printable_count = sum(
        character.isprintable()
        for character in stripped_text
    )

    alphanumeric_count = sum(
        character.isalnum()
        for character in stripped_text
    )

    printable_ratio = _safe_ratio(printable_count, stripped_length)
    alphanumeric_ratio = _safe_ratio(
        alphanumeric_count,
        stripped_length,
    )

    control_characters = [
        character
        for character in text
        if unicodedata.category(character) == "Cc"
        and character not in {"\n", "\r", "\t"}
    ]

    if not stripped_text:
        reasons.append("empty_or_whitespace_only")

    if control_characters:
        reasons.append("contains_control_characters")

    if stripped_length < min_stripped_length:
        reasons.append("text_too_short")

    if word_count < min_word_count:
        reasons.append("too_few_words")

    if printable_ratio < min_printable_ratio:
        reasons.append("low_printable_ratio")

    if alphanumeric_ratio < min_alphanumeric_ratio:
        reasons.append("low_alphanumeric_ratio")

    return ChunkQualityResult(
        is_valid=not reasons,
        reasons=tuple(reasons),
        raw_length=raw_length,
        stripped_length=stripped_length,
        word_count=word_count,
        printable_ratio=printable_ratio,
        alphanumeric_ratio=alphanumeric_ratio,
    )