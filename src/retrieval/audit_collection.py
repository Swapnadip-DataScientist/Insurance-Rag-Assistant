from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from src.retrieval.chunk_quality import validate_chunk_text


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Qdrant payload text quality."
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="Qdrant host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant HTTP port.",
    )
    parser.add_argument(
        "--collection",
        default="insurance_policy_chunks_bge_m3_v1",
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Payload field containing chunk text.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of points read per scroll request.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/reports/qdrant_chunk_quality_audit.json"),
        help="Output JSON report.",
    )

    return parser.parse_args()


def safe_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the metadata needed to identify a bad point."""

    metadata_fields = (
        "document_id",
        "source_file",
        "page_number",
        "page_chunk_index",
        "product",
        "product_type",
        "document_type",
    )

    return {
        field: payload.get(field)
        for field in metadata_fields
        if field in payload
    }


def audit_collection(
    client: QdrantClient,
    collection_name: str,
    text_field: str,
    batch_size: int,
) -> dict[str, Any]:
    total_points = 0
    valid_points = 0
    invalid_points = 0

    reason_counts: Counter[str] = Counter()
    invalid_by_source: Counter[str] = Counter()
    invalid_examples: list[dict[str, Any]] = []

    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            total_points += 1

            payload = point.payload or {}
            text = payload.get(text_field)
            quality = validate_chunk_text(text)

            if quality.is_valid:
                valid_points += 1
                continue

            invalid_points += 1
            reason_counts.update(quality.reasons)

            source_file = str(payload.get("source_file") or "UNKNOWN")
            invalid_by_source[source_file] += 1

            invalid_examples.append(
                {
                    "point_id": str(point.id),
                    "metadata": safe_metadata(payload),
                    "text_repr": repr(text),
                    "quality": asdict(quality),
                }
            )

        if next_offset is None:
            break

        offset = next_offset

    invalid_rate = (
        invalid_points / total_points
        if total_points
        else 0.0
    )

    return {
        "status": "PASS" if invalid_points == 0 else "QUALITY_ISSUES_FOUND",
        "collection_name": collection_name,
        "text_field": text_field,
        "total_points": total_points,
        "valid_points": valid_points,
        "invalid_points": invalid_points,
        "invalid_rate": invalid_rate,
        "invalid_rate_percent": round(invalid_rate * 100, 4),
        "reason_counts": dict(reason_counts.most_common()),
        "invalid_by_source": dict(invalid_by_source.most_common()),
        "invalid_examples": invalid_examples,
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    client = QdrantClient(
        host=args.host,
        port=args.port,
        timeout=30,
    )

    report = audit_collection(
        client=client,
        collection_name=args.collection,
        text_field=args.text_field,
        batch_size=args.batch_size,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    LOGGER.info("Collection: %s", report["collection_name"])
    LOGGER.info("Total points: %d", report["total_points"])
    LOGGER.info("Valid points: %d", report["valid_points"])
    LOGGER.info("Invalid points: %d", report["invalid_points"])
    LOGGER.info(
        "Invalid rate: %.4f%%",
        report["invalid_rate_percent"],
    )
    LOGGER.info("Reason counts: %s", report["reason_counts"])
    LOGGER.info("Report saved: %s", args.report)


if __name__ == "__main__":
    main()