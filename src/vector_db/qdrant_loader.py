from __future__ import annotations

import argparse #accepts values such as embedding directory and batch size from the command line.
import json
import logging
import uuid #generates stable Qdrant point IDs. Primary Key ! 
from collections.abc import Iterator #defines streaming return types.
from pathlib import Path
from typing import Any
import numpy as np
from qdrant_client import QdrantClient, models

############################# Configuration #########################

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME ="insurance_policy_chunks_bge_m3_v1" #collection where insurance chunks are stored.
VECTOR_NAME= "dense" #name assigned to the BGE-M3 dense vector.
BGE_M3_VECTOR_SIZE = 1024
DEFAULT_BATCH_SIZE = 64
EMBEDDING_MODEL ='BAAI/bge-m3'
LOGGER = logging.getLogger(__name__)
# Keep this value unchanged after the first production load.

#This fixed namespace is used with UUID5 to generate deterministic point IDs.document_id + chunk_id = always produces the same UUID
POINT_ID_NAMESPACE = uuid.UUID("86d19103-f4ea-4ae8-b78f-7a69addf1848") 

LOGGER = logging.getLogger(__name__)


#############################
# Qdrant connection -Creates the Python connection to the Qdrant server running in Docker. 
#The client communicates with: Python loader → localhost:6333 → Qdrant container
#############################

def create_qdrant_client() -> QdrantClient:
    """
    Connect to the locally running Qdrant Docker container.
    """

    client = QdrantClient(
        url=QDRANT_URL,
        timeout=60,
    )

    # This call fails immediately if Qdrant cannot be reached.
    client.get_collections()

    LOGGER.info("Connected to Qdrant at %s", QDRANT_URL)

    return client

#############################
#Collection -Creates the collection only when it does not already exist.
#############################

def create_collection_if_missing(client : QdrantClient, ) -> None : 

    """
    Create a Qdrant collection for BGE-M3 dense embeddings. Existing collections are preserved. This function never deletes data.
    """

    if client.collection_exists(COLLECTION_NAME): #This prevents accidental deletion or recreation of an existing collection.
        LOGGER.info("Collection already exists", COLLECTION_NAME)
        return

    """ Vector Configuration 
        Vector name: dense, 
        Dimension:   1024, 
        Distance:    Cosine
    """

    client.create_collection(
        collection_name =COLLECTION_NAME,
        vectors_config ={VECTOR_NAME : models.VectorParams(size = BGE_M3_VECTOR_SIZE , distance = models.Distance.COSINE,)},
        hnsw_config = models.HnswConfigDiff(m=16,ef_construct=128, full_scan_threshold=10_000),
         #16 is a reasonable baseline.
         #Controls how thoroughly Qdrant searches while constructing the HNSW graph.
         #For smaller segments, Qdrant may use direct scanning rather than the HNSW index because scanning can be more efficient at small scale
         
        optimizers_config = models.OptimizersConfigDiff(indexing_threshold= 20_000,)
        )

#Checks that an existing collection is compatible with the embeddings being uploaded.
def validate_existing_collection(client:QdrantClient,)-> None:
    """
    Ensure an existing collection is configured for a named dense vector.
    """

    collection_info = client.get_collection(COLLECTION_NAME)

    vectors_config = collection_info.config.params.vectors

    if not isinstance(vectors_config, dict): 
        raise RuntimeError(
            "The existing collection does not use named vectors. "
            f"Expected named vector '{VECTOR_NAME}'."
        )

    if VECTOR_NAME not in vectors_config:
        raise RuntimeError(f"Existing collection is missing vector '{VECTOR_NAME}'.")

    configured_size = int(vectors_config[VECTOR_NAME].size)
    expected_size = int(BGE_M3_VECTOR_SIZE)

    if configured_size != expected_size:
        raise RuntimeError("Collection vector-size mismatch: " 
                           f"expected :{expected_size},"
                           f"found{configured_size}.")

    LOGGER.info(
        "Collection configuration validated: "
        "vector=%s, dimension=%d, distance=Cosine",
        VECTOR_NAME,
        configured_size,
        )

# ---------------------------------------------------------------------------
# Payload indexes Creates indexes for metadata fields that may be used as filters during retrieval.
# ---------------------------------------------------------------------------

def create_payload_indexes(
    client: QdrantClient,
) -> None:
    """Create indexes only for metadata used in retrieval filters. Text and numeric measurements do not need keyword indexes.
    """

    index_definitions = {
        "document_id": models.PayloadSchemaType.KEYWORD,
        "source_file": models.PayloadSchemaType.KEYWORD,
        "chunk_id": models.PayloadSchemaType.KEYWORD,
        "page_number": models.PayloadSchemaType.INTEGER,
        "page_chunk_index": models.PayloadSchemaType.INTEGER,
        "chunking_strategy": models.PayloadSchemaType.KEYWORD,
    }

    collection_info = client.get_collection(COLLECTION_NAME)
    existing_indexes = collection_info.payload_schema or {}

    for field_name, field_schema in index_definitions.items():
        if field_name in existing_indexes:
            LOGGER.info(
                "Payload index already exists: %s",
                field_name,
            )
            continue

        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )

        LOGGER.info(
            "Created payload index: %s",
            field_name,
        )


# ---------------------------------------------------------------------------
# Input-file discovery - Automatically finds all embedding files:
# Automatically finds all embedding files:These three files constitute one complete embedding set.
#If either the manifest or validation file is missing, the loader stops rather than partially loading that document.
#This design lets the same loader process 10, 100 or 500 documents automatically.
# ---------------------------------------------------------------------------

def discover_embedding_sets(embeddings_directory: Path,) -> list[tuple[Path, Path, Path]]:
    """Discover matching vector, manifest and validation files. Returns:
        List of tuples containing:
        (
            dense_npy_path,
            manifest_jsonl_path,
            validation_json_path
        )
    """

    if not embeddings_directory.exists():
        raise FileNotFoundError(
            f"Embedding directory not found: {embeddings_directory}"
        )

    embedding_sets: list[tuple[Path, Path, Path]] = []

    for dense_path in sorted(
        embeddings_directory.glob("*.dense.npy")
    ):
        file_prefix = dense_path.name.removesuffix(".dense.npy")

        manifest_path = embeddings_directory / (
            f"{file_prefix}.manifest.jsonl"
        )

        validation_path = embeddings_directory / (
            f"{file_prefix}.validation.json"
        )

        missing_files = [
            str(path)
            for path in (manifest_path, validation_path)
            if not path.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Missing companion files for {dense_path.name}: "
                + ", ".join(missing_files)
            )

        embedding_sets.append(
            (
                dense_path,
                manifest_path,
                validation_path,
            )
        )

    if not embedding_sets:
        raise FileNotFoundError(
            f"No '*.dense.npy' files found in {embeddings_directory}"
        )

    return embedding_sets


# ---------------------------------------------------------------------------
# Validation - Reads the .validation.json file produced by your embedding pipeline.
# ---------------------------------------------------------------------------

def load_validation_report(
    validation_path: Path,
) -> dict[str, Any]:
    """
    Read and verify the embedding validation report.
    """

    with validation_path.open(
        "r",
        encoding="utf-8",
    ) as validation_file:
        report = json.load(validation_file)

    status = str(report.get("status", "")).upper()

    if status != "PASS":
        raise ValueError(
            f"Validation report did not pass: {validation_path}; "
            f"status={status!r}"
        )

    model_name = report.get("model_name")

    if model_name != EMBEDDING_MODEL:
        raise ValueError(
            f"Unexpected embedding model in {validation_path}: "
            f"expected {EMBEDDING_MODEL}, found {model_name}"
        )

    LOGGER.info(
        "Validation report passed: %s",
        validation_path.name,
    )

    return report

#Reads the NumPy embedding matrix.
def load_dense_embeddings(
    dense_path: Path,
) -> np.ndarray:
    """
    Memory-map the embedding matrix.

    mmap_mode='r' avoids loading the complete matrix into RAM.
    This is important when the project grows to hundreds of documents.
    """

    embeddings = np.load(
        dense_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"{dense_path.name} must contain a 2D matrix; "
            f"found shape={embeddings.shape}"
        )

    if embeddings.shape[1] != BGE_M3_VECTOR_SIZE:
        raise ValueError(
            f"Vector-dimension mismatch in {dense_path.name}: "
            f"expected {BGE_M3_VECTOR_SIZE}, "
            f"found {embeddings.shape[1]}"
        )

    if not np.issubdtype(embeddings.dtype, np.floating):
        raise ValueError(
            f"Expected floating-point embeddings in {dense_path.name}; "
            f"found dtype={embeddings.dtype}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"{dense_path.name} contains NaN or infinite values."
        )

    LOGGER.info(
        "Loaded vector matrix: %s; shape=%s; dtype=%s",
        dense_path.name,
        embeddings.shape,
        embeddings.dtype,
    )

    return embeddings

#Purpose - Counts the non-empty records in the manifest without loading the entire file into memory. Each record should correspond to one vector row.
def count_manifest_records(
    manifest_path: Path,
) -> int:
    """
    Count non-empty JSONL records without loading all records into memory.
    """

    record_count = 0

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as manifest_file:
        for line in manifest_file:
            if line.strip():
                record_count += 1

    return record_count


def validate_record_count(
    manifest_path: Path,
    embeddings: np.ndarray,
) -> None:
    """
    Ensure every manifest record has exactly one embedding vector.
    """

    manifest_count = count_manifest_records(manifest_path)
    embedding_count = embeddings.shape[0]

    if manifest_count != embedding_count:
        raise ValueError(
            f"Record-count mismatch for {manifest_path.name}: "
            f"manifest_records={manifest_count}, "
            f"embedding_rows={embedding_count}"
        )

    LOGGER.info(
        "Record alignment validated: records=%d, vectors=%d",
        manifest_count,
        embedding_count,
    )


# ---------------------------------------------------------------------------
# Manifest processing - Reads the JSONL manifest one record at a time.
# ---------------------------------------------------------------------------

def read_manifest(
    manifest_path: Path,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """
    Stream JSONL records one at a time.
    """

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as manifest_file:
        record_index = 0

        for line_number, line in enumerate(
            manifest_file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {manifest_path.name} "
                    f"at line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at "
                    f"{manifest_path.name}:{line_number}"
                )

            yield record_index, record
            record_index += 1


def require_value(
    record: dict[str, Any],
    field_name: str,
    manifest_path: Path,
    record_index: int,
) -> Any:
    """
    Retrieve a mandatory manifest field.
    """

    value = record.get(field_name)

    if value is None:
        raise ValueError(
            f"Missing '{field_name}' in {manifest_path.name}, "
            f"record={record_index}"
        )

    if isinstance(value, str) and not value.strip():
        raise ValueError(
            f"Empty '{field_name}' in {manifest_path.name}, "
            f"record={record_index}"
        )

    return value


def deterministic_point_id(
    document_id: str,
    chunk_id: str,
) -> str:
    """
    Produce the same UUID for the same document chunk.

    This provides idempotent upserts.
    """

    natural_key = f"{document_id}::{chunk_id}"

    return str(
        uuid.uuid5(
            POINT_ID_NAMESPACE,
            natural_key,
        )
    )


def create_payload(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the metadata stored with the Qdrant vector.
    """

    allowed_fields = (
        "chunk_id",
        "document_id",
        "source_file",
        "page_number",
        "page_chunk_index",
        "text",
        "character_count",
        "chunking_strategy",
        "max_chunk_chars",
        "overlap_chars",
        "embedding_row",
    )

    payload = {
        field_name: record[field_name]
        for field_name in allowed_fields
        if field_name in record
    }

    payload["embedding_model"] = EMBEDDING_MODEL
    payload["vector_name"] = VECTOR_NAME
    payload["vector_dimension"] = BGE_M3_VECTOR_SIZE

    return payload

# Combines one manifest record with the corresponding vector.

def create_qdrant_point(
    record: dict[str, Any],
    embeddings: np.ndarray,
    expected_row: int,
    manifest_path: Path,
) -> models.PointStruct:
    """
    Combine one manifest record with its corresponding embedding row.
    """

    document_id = str(
        require_value(
            record,
            "document_id",
            manifest_path,
            expected_row,
        )
    )

    chunk_id = str(
        require_value(
            record,
            "chunk_id",
            manifest_path,
            expected_row,
        )
    )

    require_value(
        record,
        "text",
        manifest_path,
        expected_row,
    )

    embedding_row = int(
        require_value(
            record,
            "embedding_row",
            manifest_path,
            expected_row,
        )
    )

    # Protect against accidental manifest/vector misalignment.
    if embedding_row != expected_row:
        raise ValueError(
            f"Embedding-row mismatch in {manifest_path.name}: "
            f"record_position={expected_row}, "
            f"embedding_row={embedding_row}"
        )

    if embedding_row < 0 or embedding_row >= embeddings.shape[0]:
        raise IndexError(
            f"Embedding row {embedding_row} is outside matrix "
            f"range 0..{embeddings.shape[0] - 1}"
        )

    vector = np.asarray(
        embeddings[embedding_row],
        dtype=np.float32,
    ).tolist()

    return models.PointStruct(
        id=deterministic_point_id(
            document_id=document_id,
            chunk_id=chunk_id,
        ),
        vector={
            VECTOR_NAME: vector,
        },
        payload=create_payload(record),
    )


# ---------------------------------------------------------------------------
# Batched upsert - Processes one complete document.
# ---------------------------------------------------------------------------

def upsert_embedding_set(
    client: QdrantClient,
    dense_path: Path,
    manifest_path: Path,
    validation_path: Path,
    batch_size: int,
) -> int:
    """
    Validate and upload one document's embeddings.
    """

    load_validation_report(validation_path)

    embeddings = load_dense_embeddings(dense_path)

    validate_record_count(
        manifest_path=manifest_path,
        embeddings=embeddings,
    )

    batch: list[models.PointStruct] = []
    uploaded_count = 0

    for record_index, record in read_manifest(manifest_path):
        point = create_qdrant_point(
            record=record,
            embeddings=embeddings,
            expected_row=record_index,
            manifest_path=manifest_path,
        )

        batch.append(point)

        if len(batch) >= batch_size:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True,
            )

            uploaded_count += len(batch)

            LOGGER.info(
                "%s: uploaded=%d",
                manifest_path.name,
                uploaded_count,
            )

            batch = []

    if batch:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True,
        )

        uploaded_count += len(batch)

    LOGGER.info(
        "Completed document: %s; points=%d",
        manifest_path.name,
        uploaded_count,
    )

    return uploaded_count

#Coordinates loading across every discovered document.

def load_all_embeddings(
    client: QdrantClient,
    embeddings_directory: Path,
    batch_size: int,
) -> int:
    """
    Discover and upload all valid embedding sets.
    """

    embedding_sets = discover_embedding_sets(
        embeddings_directory
    )

    LOGGER.info(
        "Discovered %d embedding sets",
        len(embedding_sets),
    )

    total_processed = 0

    for set_number, (
        dense_path,
        manifest_path,
        validation_path,
    ) in enumerate(embedding_sets, start=1):

        LOGGER.info(
            "Processing document %d/%d: %s",
            set_number,
            len(embedding_sets),
            dense_path.name,
        )

        processed = upsert_embedding_set(
            client=client,
            dense_path=dense_path,
            manifest_path=manifest_path,
            validation_path=validation_path,
            batch_size=batch_size,
        )

        total_processed += processed

    return total_processed


# ---------------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------------

def validate_qdrant_count(
    client: QdrantClient,
    expected_count: int,
) -> None:
    """
    Compare processed points with the current Qdrant collection count.
    """

    count_result = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )

    stored_count = count_result.count

    LOGGER.info(
        "Qdrant count validation: "
        "processed_this_run=%d, stored_collection_points=%d",
        expected_count,
        stored_count,
    )

    if stored_count < expected_count:
        raise RuntimeError(
            "Qdrant contains fewer points than were processed: "
            f"processed={expected_count}, stored={stored_count}"
        )


# ---------------------------------------------------------------------------
# Command line - Lets you configure the loader without modifying its source code.
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load BGE-M3 dense insurance-policy embeddings "
            "into Qdrant."
        )
    )

    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("data/embeddings/dense"),
        help="Directory containing dense, manifest and validation files.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of points per Qdrant upsert request.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    arguments = parse_arguments()

    if arguments.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
        )

    client = create_qdrant_client()

    create_collection_if_missing(client)

    validate_existing_collection(client)

    create_payload_indexes(client)

    total_processed = load_all_embeddings(
        client=client,
        embeddings_directory=arguments.embeddings_dir,
        batch_size=arguments.batch_size,
    )

    validate_qdrant_count(
        client=client,
        expected_count=total_processed,
    )

    LOGGER.info(
        "Qdrant loading completed successfully. "
        "Total processed points: %d",
        total_processed,
    )


if __name__ == "__main__":
    main()