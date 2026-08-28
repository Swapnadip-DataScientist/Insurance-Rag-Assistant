from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from FlagEmbedding import BGEM3FlagModel

######################################
#CONSTANTS
######################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = (PROJECT_ROOT /"data" /"embeddings" /"dense")

BGE_M3_DIMENSION = 1024
BGE_M3_MODEL_MAX_TOKENS = 8192

######################################
#LOGGING
######################################

logging.basicConfig(level= logging.INFO,
                    format=(
                            "%(asctime)s | " 
                            "%(levelname)s | "
                            "%(name)s | " 
                            "%(message)s"
                           ),
                    )

logger = logging.getLogger(__name__)

######################################
#LOGGING

@dataclass (frozen = True)
class EmbeddingConfig:

    #Configurations for BGE-M3 dense embedding generations
    model_name : str = "BAAI/bge-m3"

    #Current machine configuration
    device : str = "cpu"
    use_fp16 : bool =False

    #Conservative CPU batch size
    batch_size : int =4

    #Max no of token for chunk
    max_length : int = 512

    #Normalize vectors for Cosine Similarity
    normalize_embeddings : bool = True

    #BGE-M3 dense vector dimension
    expected_dimension : int = BGE_M3_DIMENSION

    norm_tolerance: float = 1e-2

# ============================================================
# BGE-M3 EMBEDDER
# ============================================================

class BgeM3DenseEmbedder:

    """
    Dense embedding generator using BAAI/bge-m3.
    Baseline configuration:
        Dense   = enabled
        Sparse  = disabled
        ColBERT = disabled
    """

    def __init__(self, config : EmbeddingConfig) -> None:

        self.config = config

        self._validate_config()

        logger.info( "Loading BGE-M3 model |" " model =%s | device =%s | " "fp16=%s | batch_size =%d | " 
                     "max_length = %d",
                     config.model_name,
                     config.device,
                     config.use_fp16,
                     config.batch_size,
                     config.max_length,
                    )
        

        self.model = BGEM3FlagModel( self.config.model_name, 
                                     devices = self.config.device,
                                     use_fp16 =self.config.use_fp16,
                                     pooling_method = "cls",
                                     normalize_embeddings=(self.config.normalize_embeddings),
                                     batch_size = self.config.batch_size,
                                     passage_max_length = self.config.max_length,
                                     return_dense = True,
                                     return_sparse = False,
                                     return_colbert_vecs = False,)

        logger.info("BGE-M3 model loaded successfully")

    def _validate_config(self) -> None:

        if self.config.batch_size <=0:
            raise ValueError("batch size must be greater than 0")

        if not( 1 <= self.config.max_length <= BGE_M3_MODEL_MAX_TOKENS):
            raise ValueError("max_length must be between"
                              f"1 and {BGE_M3_MODEL_MAX_TOKENS}."
                            )

    
    # ============================================================
    # TOKEN LENGTH CHECK
    # ============================================================
    
    def get_token_lengths(self, texts: list[str], tokenizer_batch_size : int = 256) -> np.ndarray:

        """ Determine real BGE-M3 tokenizer lengths.  We do NOT truncate here because we want to detect oversized chunks.
        """

        token_lengths : list[int] = []

        for start in range(
            0,
            len(texts),
            tokenizer_batch_size
        ):
            batch = texts[start: start + tokenizer_batch_size]

            encoded = self.model.tokenizer(
                batch,
                add_special_tokens=True,
                truncation = False,
                padding = False,
                return_attention_mask= False
            )

            token_lengths.extend(
                len(input_ids) 
                for input_ids 
                in encoded["input_ids"]
                )

        return np.asarray(token_lengths, dtype=np.int32)

    def validate_token_lengths(self, texts: list[str]) -> np.ndarray:

        token_lengths = self.get_token_lengths(texts)

        oversized= np.flatnonzero(token_lengths > self.config.max_length)

        if oversized.size>0:
            examples = [ 
                        {  "row" : int(index), 
                            "tokens" : int(token_lengths[index]
                                        )
                        }
                        for index in oversized[:10]
                        ]

            raise ValueError(  
                        "Chunks would be truncated by BGE-M3. "
                        f"max_length={self.config.max_length}. "
                        f"Examples={examples}")

        return token_lengths
    # ============================================================
    # EMBEDDING
    # ============================================================

    def encode_corpus(self, texts: list[str],) -> np.ndarray:
        """
        Generate dense corpus embeddings.
        Expected output:  [number_of_chunks, 1024]
        """

        if not texts:
            raise ValueError("No texts supplied for embedding.")

        output = self.model.encode_corpus(
            texts,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        if "dense_vecs" not in output:
            raise RuntimeError( "BGE-M3 did not return dense_vecs." )

        embeddings = np.asarray( output["dense_vecs"], dtype=np.float32,)

        # Defensive handling for one chunk
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1,-1,)
        return embeddings


# ============================================================
# JSONL LOADING
# ============================================================

def load_jsonl_chunks(jsonl_path: Path,) -> tuple[
    list[dict[str, Any]],
    list[str],]:
    """
    Load saved chunks from JSONL. Complete records are preserved for later  Qdrant payload creation.
    """

    if not jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    records: list[dict[str, Any]] = []
    texts: list[str] = []

    with jsonl_path.open("r", encoding="utf-8",) as file:

        for line_number, raw_line in enumerate(file, start=1,):

            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {jsonl_path}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} "
                    "must contain a JSON object."
                )

            # Your chunker should normally use "text".
            # These fallbacks make the loader defensive.
            text = (
                record.get("text")
                or record.get("chunk_text")
                or record.get("content")
            )

            if not isinstance(text, str):

                raise ValueError(
                    f"No valid text field at "
                    f"line {line_number}. "
                    f"Available fields: "
                    f"{list(record.keys())}"
                )

            text = text.strip()

            if not text:

                raise ValueError(
                    f"Empty chunk text at "
                    f"line {line_number}."
                )

            records.append(record)
            texts.append(text)

    if not records:

        raise ValueError(
            f"No chunks found in {jsonl_path}"
        )

    return records, texts


# ============================================================
# EMBEDDING VALIDATION
# ============================================================

def validate_embeddings(
    records: list[dict[str, Any]],
    texts: list[str],
    embeddings: np.ndarray,
    token_lengths: np.ndarray,
    config: EmbeddingConfig,
) -> dict[str, Any]:

    chunk_count = len(records)

    # --------------------------------------------------------
    # 1. Chunk / text count
    # --------------------------------------------------------

    if chunk_count != len(texts):
        raise ValueError(
            "Chunk/text count mismatch."
        )

    # --------------------------------------------------------
    # 2. Embeddings must be matrix [N, D]
    # --------------------------------------------------------

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must have shape [N, D]. "
            f"Received {embeddings.shape}"
        )

    # --------------------------------------------------------
    # 3. Number chunks == number vectors
    # --------------------------------------------------------

    if embeddings.shape[0] != chunk_count:
        raise ValueError(
            "Chunk/embedding count mismatch: "
            f"chunks={chunk_count}, "
            f"embeddings={embeddings.shape[0]}"
        )

    # --------------------------------------------------------
    # 4. BGE-M3 dimension
    # --------------------------------------------------------

    if ( embeddings.shape[1]!= config.expected_dimension):

        raise ValueError(
            "Unexpected embedding dimension: "
            f"{embeddings.shape[1]}. "
            f"Expected "
            f"{config.expected_dimension}."
        )

    # --------------------------------------------------------
    # 5. NaN / Inf
    # --------------------------------------------------------

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or Inf.")

    # --------------------------------------------------------
    # 6. Token count consistency
    # --------------------------------------------------------

    if len(token_lengths) != chunk_count:
        raise ValueError("Token length count mismatch.")

    # --------------------------------------------------------
    # 7. Vector norms
    # --------------------------------------------------------

    norms = np.linalg.norm(
        embeddings.astype(
            np.float64,
            copy=False,
        ),
        axis=1,
    )

    if not np.isfinite(norms).all():
        raise ValueError("Vector norms contain NaN or Inf.")

    if np.any(norms == 0):
        raise ValueError( "Zero-norm embedding detected." )

    max_norm_deviation = float(np.max( np.abs(norms - 1.0)))

    if (config.normalize_embeddings and max_norm_deviation > config.norm_tolerance):
        raise ValueError(
            "Normalized embeddings have "
            "unexpected vector norms. "
            f"Maximum deviation="
            f"{max_norm_deviation:.6f}"
        )

    # --------------------------------------------------------
    # Validation report
    # --------------------------------------------------------

    return {"status": "PASS",
        "model_name": config.model_name,
        "device":config.device,
        "use_fp16":config.use_fp16,
        "batch_size": config.batch_size,
        "max_length":config.max_length,
        "normalized":config.normalize_embeddings,
        "chunk_count":chunk_count,
        "embedding_count": int(embeddings.shape[0]),
        "embedding_shape":list(embeddings.shape),
        "embedding_dtype":str(embeddings.dtype),
        "embedding_dimension":int(embeddings.shape[1]),
        "nan_inf_check":"PASS",
        "empty_text_check": "PASS",
        "token_lengths":{
            "minimum":int(token_lengths.min()),
            "mean":float(token_lengths.mean()),
            "p95":float(np.percentile(token_lengths,95,)),
            "maximum":int(token_lengths.max()),
        },
        "vector_norms":{
            "minimum": float(norms.min()),
            "mean":float(norms.mean()),
            "maximum":float(norms.max()),
            "max_deviation_from_1": max_norm_deviation,
        },
    }


# ============================================================
# FILE HELPERS
# ============================================================

def calculate_sha256(
    path: Path,
) -> str:

    sha256 = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()


def save_npy_atomic(
    path: Path,
    embeddings: np.ndarray,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        path.name + ".tmp"
    )

    with temp_path.open("wb") as file:

        np.save(
            file,
            embeddings,
            allow_pickle=False,
        )

    os.replace(
        temp_path,
        path,
    )


def save_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        path.name + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    os.replace(
        temp_path,
        path,
    )


def save_manifest_atomic(
    path: Path,
    records: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        path.name + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for index, record in enumerate(
            records
        ):

            manifest_record = {
                **record,
                "embedding_row": index,
            }

            file.write(
                json.dumps(
                    manifest_record,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    os.replace(
        temp_path,
        path,
    )


# ============================================================
# PROCESS ONE JSONL
# ============================================================

def process_chunk_file(
    input_path: Path,
    output_dir: Path,
    embedder: BgeM3DenseEmbedder,
) -> dict[str, Any]:

    logger.info(
        "Processing chunk file: %s",
        input_path,
    )

    # -----------------------------
    # Load chunks
    # -----------------------------

    records, texts = load_jsonl_chunks(
        input_path
    )

    logger.info(
        "Loaded %d chunks.",
        len(records),
    )

    # -----------------------------
    # Check tokenizer lengths
    # -----------------------------

    token_lengths = (
        embedder.validate_token_lengths(
            texts
        )
    )

    logger.info(
        "Token validation passed | "
        "max_tokens=%d",
        int(token_lengths.max()),
    )

    # -----------------------------
    # Generate embeddings
    # -----------------------------

    embeddings = (
        embedder.encode_corpus(
            texts
        )
    )

    # -----------------------------
    # Validate embeddings
    # -----------------------------

    report = validate_embeddings(
        records=records,
        texts=texts,
        embeddings=embeddings,
        token_lengths=token_lengths,
        config=embedder.config,
    )

    report["source_file"] = str(
        input_path
    )

    report["source_sha256"] = (
        calculate_sha256(
            input_path
        )
    )

    report["created_at_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    # -----------------------------
    # Save only after PASS
    # -----------------------------

    stem = input_path.stem

    vector_path = (
        output_dir
        / f"{stem}.dense.npy"
    )

    manifest_path = (
        output_dir
        / f"{stem}.manifest.jsonl"
    )

    validation_path = (
        output_dir
        / f"{stem}.validation.json"
    )

    save_npy_atomic(
        vector_path,
        embeddings,
    )

    save_manifest_atomic(
        manifest_path,
        records,
    )

    save_json_atomic(
        validation_path,
        report,
    )

    logger.info(
        "Embedding PASS | "
        "shape=%s | "
        "norm_mean=%.6f",
        embeddings.shape,
        report["vector_norms"]["mean"],
    )

    logger.info(
        "Vectors saved: %s",
        vector_path,
    )

    logger.info(
        "Manifest saved: %s",
        manifest_path,
    )

    logger.info(
        "Validation saved: %s",
        validation_path,
    )

    return report


# ============================================================
# DISCOVER JSONL FILES
# ============================================================

def discover_jsonl_files(
    input_path: Path,
) -> list[Path]:

    if input_path.is_file():

        if (
            input_path.suffix.lower()
            != ".jsonl"
        ):
            raise ValueError(
                "Input file must be .jsonl"
            )

        return [input_path]

    if input_path.is_dir():

        files = sorted(
            input_path.rglob("*.jsonl")
        )

        if not files:
            raise FileNotFoundError(
                "No JSONL files found under "
                f"{input_path}"
            )

        return files

    raise FileNotFoundError(
        f"Input path not found: "
        f"{input_path}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate validated dense BGE-M3 "
            "embeddings from JSONL chunks."
        )
    )

    parser.add_argument("--input", type=Path, required=True, help=("JSONL chunk file or directory."),)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help=( "Embedding output directory." ),)
    parser.add_argument("--batch-size",type=int, default=4,)
    parser.add_argument( "--max-length", type=int, default=512,)
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    config = EmbeddingConfig(batch_size=args.batch_size, max_length=args.max_length, )

    # Load BGE-M3 only once.
    embedder = BgeM3DenseEmbedder(config)
    files = discover_jsonl_files(args.input)
    logger.info("Found %d JSONL file(s).", len(files),)

    for file_path in files:

        process_chunk_file( input_path=file_path, output_dir=args.output, embedder=embedder, )


if __name__ == "__main__":
    main()
