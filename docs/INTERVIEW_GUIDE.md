# Interview Guide

These are concise explanations for discussing the project in an interview.

## Explain the Project in 60–90 Seconds

> I designed the system around retrieval reliability rather than treating the LLM as the main component. Insurance policies contain exclusions, conditions and similar-looking clauses, so vector similarity alone was not enough. I use BGE-M3 to retrieve candidates from Qdrant and a BGE cross-encoder to rerank those candidates before generation. I built a golden-query evaluation pipeline to measure whether reranking actually improved retrieval, rather than assuming it did. The backend is exposed through FastAPI, Gradio is kept as a separate client, and the application is containerized with Docker Compose using persistent Qdrant and Hugging Face cache volumes. Qwen currently runs locally through Ollama.

## Why Use RAG?

> Insurance answers need to be grounded in policy wording. I don't want the model relying on general knowledge for contractual questions. RAG lets the model answer from retrieved source evidence and allows the response to carry page-level citations.

## Why BGE-M3?

> I wanted a strong embedding model suitable for semantic retrieval and longer text while remaining practical for local experimentation. The model produces 1024-dimensional normalized vectors, which are stored in Qdrant and searched using cosine distance.

## Why Add a Reranker?

> Vector search is good for recall, but the nearest semantic match is not always the governing insurance clause. A cross-encoder evaluates the query and candidate together, so I retrieve a broader candidate set first and use reranking to improve precision before the LLM sees the evidence.

## What Evidence Shows Reranking Helped?

> I built a golden-query retrieval evaluation. Baseline Hit@3 was 0.6923 and improved to 0.8462 after reranking. MRR@3 improved from 0.4615 to 0.7692. That gave me measurable evidence that reranking improved ranking quality.

## Why Not Pass Many Chunks to the LLM?

> More context is not automatically better. Extra chunks can introduce noise and conflicting clauses. I prefer broad candidate retrieval, reranking, and then passing only the strongest evidence to generation.

## Why FastAPI?

> FastAPI acts as the backend contract around the RAG system. It keeps the UI decoupled from retrieval and generation, provides request validation, health endpoints, lifecycle management, and a clean boundary for future frontends or enterprise integrations.

## Why Separate Gradio and FastAPI?

> Gradio is only a presentation layer. It talks to FastAPI over HTTP. It doesn't know about Qdrant, embeddings, reranking, or Ollama. That means I can replace Gradio later without changing the RAG backend.

## What Is `/health/live` vs `/health/ready`?

> Liveness answers whether the API process is alive. Readiness answers whether the RAG runtime has finished initializing and is ready to accept real traffic. Production orchestrators need that distinction so they don't route requests to a process that has started but is not ready.

## Why Docker?

> Docker makes the runtime reproducible. Instead of manually recreating Python dependencies and service configuration on every machine, I package the API and UI into images and use Compose to define networking, ports, volumes, environment variables, dependencies, and health checks.

## What Does Docker Compose Do?

> Docker builds and runs containers. Docker Compose describes how multiple containers work together. In this project Compose orchestrates Gradio, FastAPI, and Qdrant and defines their networking, storage, health checks, and startup dependencies.

## Why Persistent Volumes?

> Containers are disposable, but Qdrant data and model caches are not. I store them in Docker volumes so I can remove and recreate containers without rebuilding the vector database or re-downloading models.

## What Is `host.docker.internal`?

> Once FastAPI runs inside Docker, `localhost` refers to the container itself. Ollama is running on the Windows host, so Docker Desktop provides `host.docker.internal` as the host address. I kept my existing local-Ollama security validation and explicitly whitelisted that Docker host alias rather than removing the restriction.

## What Docker Issue Did You Encounter?

> One useful issue was Qdrant being marked unhealthy even though its API and data were fine. The health check used `wget`, but the Qdrant image didn't include `wget`. I inspected the container health state, confirmed the failure was the check itself, and replaced it with a Bash TCP health check. It was a good example of separating application health from health-check implementation.

## How Did You Protect Qdrant Data?

> I verified the existing Docker volume before replacing the standalone Qdrant container. Compose then attached the same external volume. I also tested `docker compose down` followed by `docker compose up -d` and confirmed that all 1949 points remained available.

## Why Is `indexed_vectors_count` Zero?

> The current collection is below Qdrant's configured indexing threshold of 20,000 points, so zero indexed vectors is expected at this scale. It isn't data loss. HNSW/index settings are something I would benchmark when the corpus becomes larger.

## Why Is Concurrency Set to 1?

> The current environment is CPU-constrained and reranking plus local LLM inference are expensive. I deliberately protect the machine with one concurrent inference request. In production I would tune concurrency using hardware capacity, model server throughput, latency targets, and load testing.

## What Would You Improve Next?

> The next priorities are selective OCR and table-aware extraction, configurable LLM providers, concurrency/load testing, Qdrant index tuning for larger corpora, smaller CPU-specific Docker images, and stronger observability.
