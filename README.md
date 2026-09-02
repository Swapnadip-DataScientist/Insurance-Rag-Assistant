# Insurance RAG Assistant

A Retrieval-Augmented Generation (RAG) application for answering questions from insurance policy documents with source-grounded responses.

The focus of this project is not simply connecting an LLM to a vector database. The main engineering work is in building a reliable retrieval pipeline around complex insurance wording, validating retrieval quality, reranking candidate evidence, exposing the pipeline through an API, and packaging the services for repeatable deployment.

## Current Stack

- **PDF extraction:** PyMuPDF
- **Embeddings:** BAAI/bge-m3
- **Vector database:** Qdrant
- **Reranker:** BAAI/bge-reranker-v2-m3
- **Generation:** Qwen 3.5 through Ollama
- **Backend:** FastAPI
- **UI:** Gradio
- **Containerization:** Docker
- **Orchestration:** Docker Compose

## High-Level Flow

```text
User
  |
  v
Gradio UI
  |
  | POST /ask
  v
FastAPI
  |
  +--> BGE-M3 query embedding
  |
  +--> Qdrant candidate retrieval
  |
  +--> BGE cross-encoder reranking
  |
  +--> Top evidence
  |
  +--> Qwen / Ollama generation
  |
  v
Grounded answer + citations + conditions + limitations + sources
```

The LLM is deliberately kept at the end of the pipeline. Retrieval quality is treated as an engineering problem of its own.

## Why Reranking?

Insurance documents contain clauses that can be semantically similar while having very different contractual meaning. Vector search is useful for candidate generation, but the highest-similarity chunk is not always the clause that governs the answer.

The retrieval design therefore uses:

```text
Vector Search -> Candidate Set -> Cross-Encoder Reranking -> Final Evidence
```

This gives vector retrieval responsibility for recall and the reranker responsibility for improving precision.

## Retrieval Evaluation

A golden-query evaluation set is used to measure retrieval quality rather than relying only on manual inspection.

Observed retrieval metrics:

| Metric | Baseline | After Reranking |
|---|---:|---:|
| Hit@3 | 0.6923 | 0.8462 |
| MRR@3 | 0.4615 | 0.7692 |

The objective is to verify that architectural changes actually improve evidence retrieval.

## API

FastAPI exposes the RAG pipeline through:

```text
POST /ask
GET  /health/live
GET  /health/ready
```

Heavy ML and database objects are initialized once during application startup rather than being recreated for every request.

## Docker Deployment

The current Docker Compose stack contains:

```text
Gradio
FastAPI
Qdrant
```

Ollama currently runs on the Windows host and is reached from the FastAPI container through:

```text
host.docker.internal
```

Persistent Docker volumes are used for:

- Qdrant storage
- Hugging Face model cache

This allows containers to be recreated without rebuilding the vector database or repeatedly downloading model files.

## Run with Docker

Validate the Compose configuration:

```powershell
docker compose config
```

Build:

```powershell
docker compose build
```

Start:

```powershell
docker compose up -d
```

Check status:

```powershell
docker compose ps
```

Open:

```text
Gradio:  http://127.0.0.1:7860
FastAPI: http://127.0.0.1:8000/docs
Qdrant:  http://127.0.0.1:6333
```

Stop and remove the running containers while preserving named volumes:

```powershell
docker compose down
```

Do not casually use:

```powershell
docker compose down -v
```

because `-v` removes volumes and can remove persistent data.

## Project Structure

```text
insurance-rag-assistant/
|
+-- src/
|   +-- api/
|   +-- embeddings/
|   +-- evaluation/
|   +-- generation/
|   +-- ingestion/
|   +-- retrieval/
|   +-- ui/
|   +-- vector_db/
|
+-- docs/
|   +-- ARCHITECTURE.md
|   +-- DOCKER_GUIDE.md
|   +-- TECHNICAL_SPEC.md
|   +-- TROUBLESHOOTING.md
|   +-- INTERVIEW_GUIDE.md
|   +-- PHASE2_ROADMAP.md
|
+-- Dockerfile.api
+-- Dockerfile.ui
+-- compose.yaml
+-- requirements.txt
+-- requirements-ui.txt
+-- .dockerignore
+-- .gitignore
+-- .env.example
+-- README.md
```

## Documentation

See:

- `docs/ARCHITECTURE.md` — component design and request flow
- `docs/TECHNICAL_SPEC.md` — implementation details and design decisions
- `docs/DOCKER_GUIDE.md` — Docker and Compose commands with explanations
- `docs/TROUBLESHOOTING.md` — issues encountered and how they were resolved
- `docs/INTERVIEW_GUIDE.md` — concise interview-ready explanations
- `docs/PHASE2_ROADMAP.md` — planned improvements

## Current Limitations

The current baseline intentionally leaves several areas for the next phase:

- OCR and table-aware extraction
- configurable LLM provider abstraction
- concurrency/load testing
- Qdrant HNSW/index tuning for a larger corpus
- CPU-specific Docker image optimization
- stronger observability and production deployment controls

These are documented separately in the Phase 2 roadmap.
