# Technical Specification

## 1. Document Ingestion

Native PDF extraction currently uses PyMuPDF.

Baseline flow:

```text
PDF
 |
 v
Native Text Extraction
 |
 v
Text Cleaning
 |
 v
Chunking
 |
 v
Metadata Enrichment
 |
 v
Embedding
 |
 v
Qdrant
```

The baseline does not blindly OCR every page. OCR and table-aware extraction are planned for the next phase.

## 2. Chunking

The chunker uses a hierarchy:

```text
Paragraph-first
    |
Sentence fallback
    |
Character fallback
```

Typical baseline settings:

```text
max_chunk_chars = 500
overlap_chars   = 200
```

Metadata carried with each chunk includes:

- `document_id`
- `source_file`
- `page_number`
- `page_chunk_index`
- `chunk_id`
- `chunking_strategy`

## 3. Embeddings

Model:

```text
BAAI/bge-m3
```

Configuration:

```text
dimension     = 1024
normalization = enabled
distance      = cosine
```

Embedding validation includes dimension checks and normalized-vector checks before loading data into Qdrant.

## 4. Qdrant

Collection:

```text
insurance_policy_chunks_bge_m3_v1
```

Named vector:

```text
dense
```

Current verified state during Docker validation:

```text
status                  = green
optimizer_status        = ok
points_count            = 1949
indexed_vectors_count   = 0
indexing_threshold      = 20000
```

`indexed_vectors_count = 0` is expected at the current corpus size because the collection is below the configured indexing threshold.

Deterministic point IDs are used to support idempotent loading.

## 5. Retrieval

The retriever supports:

- vector candidate retrieval
- document include/exclude filters
- chunk-quality checks
- duplicate suppression
- diagnostics and rejection counts

The retriever intentionally returns more candidates than are eventually passed to generation.

## 6. Reranking

Model:

```text
BAAI/bge-reranker-v2-m3
```

Architecture:

```text
Qdrant Top-N Candidates
         |
         v
Cross-Encoder Reranker
         |
         v
Top-K Evidence
```

The cross-encoder is more expensive than vector similarity because it evaluates the query and candidate together. It is therefore applied only to a smaller candidate set.

## 7. Retrieval Evaluation

Measured retrieval performance:

```text
Baseline Hit@3  = 0.6923
Reranked Hit@3  = 0.8462

Baseline MRR@3  = 0.4615
Reranked MRR@3  = 0.7692
```

The evaluation is designed to answer a practical engineering question:

> Did reranking actually improve evidence retrieval?

## 8. Generation

Current model:

```text
qwen3.5:4b-q4_K_M
```

served using Ollama.

The generator returns a structured contract including:

- answer status
- answer
- citations
- conditions
- limitations
- source metadata

Generation errors are explicitly mapped for:

- Ollama unavailable
- requested model unavailable
- invalid structured output
- timeout
- unexpected backend failure

## 9. FastAPI Runtime

The RAG runtime is created during the FastAPI lifespan.

Heavy objects are created once before requests are accepted and are not recreated inside `POST /ask`.

CPU-heavy inference is executed using a thread pool so blocking work does not run directly on the async event loop.

Current local capacity:

```text
MAX_CONCURRENT_REQUESTS=1
```

This is an intentional CPU-safe development setting.

## 10. Docker Configuration

The Dockerized stack contains:

- Qdrant
- FastAPI
- Gradio

Ollama currently remains on the host.

Runtime values are supplied through environment variables rather than being hard-coded.

Important examples:

```text
QDRANT_HOST
QDRANT_PORT
QDRANT_COLLECTION
QDRANT_VECTOR_NAME

OLLAMA_BASE_URL
OLLAMA_MODEL
OLLAMA_NUM_CTX

RETRIEVAL_TOP_K
RERANK_TOP_K
RERANKER_MODEL

MAX_CONCURRENT_REQUESTS
```

## 11. Ollama Docker Host Connectivity

When FastAPI ran directly on Windows, Ollama was reachable using:

```text
127.0.0.1:11434
```

Inside Docker, `127.0.0.1` refers to the FastAPI container itself.

Docker Desktop therefore uses:

```text
host.docker.internal
```

The generator retained its localhost protection and extended the allowed host list to:

```text
127.0.0.1
localhost
::1
host.docker.internal
```

The local-host validation was extended rather than removed.

## 12. Qdrant Volume Reuse Across Machines

The development machines used different pre-existing Docker volume names.

The Compose file therefore supports:

```text
QDRANT_VOLUME_NAME
```

Example office configuration:

```text
QDRANT_VOLUME_NAME=insurance_qdrant_storage
```

The Compose definition resolves the actual external volume name at runtime.

This avoids modifying source-controlled infrastructure configuration for each machine.

## 13. Persistence Test

The following lifecycle test was performed:

```powershell
docker compose down
docker compose up -d
```

After container recreation, Qdrant returned:

```text
status       = green
points_count = 1949
```

This validates the design principle:

> Containers are disposable; persistent data is stored outside them.

## 14. Current Docker Image Observation

The API image is currently large because the generic Linux PyTorch installation brought CUDA/NVIDIA dependencies into a CPU-oriented deployment.

Observed API image size during development was approximately:

```text
9.21 GB
```

This is functional but not an optimized production image.

A future optimization should use a CPU-specific PyTorch build or deployment-specific base image.
