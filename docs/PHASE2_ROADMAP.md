# Phase 2 Roadmap

Phase 2 intentionally improves the system without discarding the working baseline.

## 1. OCR and Table-Aware Ingestion

Add detection for:

- scanned/image-only pages
- empty native text extraction
- low-quality extracted text
- table-heavy pages
- garbled text

Route only problematic pages through OCR or table-aware processing.

Avoid blanket OCR across every document.

## 2. LLM Provider Abstraction

The current generator uses Ollama.

Introduce a provider interface so the RAG pipeline can switch between:

- Ollama
- OpenAI
- Azure OpenAI
- AWS Bedrock
- vLLM

without changing retrieval logic.

Conceptually:

```text
RAG Service
    |
    v
LLM Provider Interface
    |
    +--> Ollama
    +--> OpenAI
    +--> Azure OpenAI
    +--> Bedrock
    +--> vLLM
```

## 3. Concurrency and Load Testing

Current development setting:

```text
MAX_CONCURRENT_REQUESTS=1
```

Revisit based on:

- CPU/GPU capacity
- embedding throughput
- reranker throughput
- LLM serving throughput
- latency targets
- queue depth
- concurrent-user testing

Do not simply increase the value without load measurements.

## 4. Qdrant Index / HNSW Tuning

Current state:

```text
points_count          = 1949
indexed_vectors_count = 0
indexing_threshold    = 20000
```

When the corpus grows, benchmark:

- `indexing_threshold`
- HNSW `m`
- `ef_construct`
- search-time parameters
- RAM usage
- disk usage
- retrieval latency
- retrieval quality

## 5. Docker Image Optimization

Current API image is large due to generic PyTorch CUDA dependencies.

Investigate:

- CPU-specific PyTorch installation
- dependency pinning
- smaller base images where practical
- multi-stage builds
- image scanning
- build cache strategy

## 6. Observability

Add structured metrics for:

- request ID
- total latency
- embedding latency
- retrieval latency
- reranking latency
- generation latency
- retrieved source IDs
- answer status
- error category
- model/provider
- request queue time

Potential production tooling:

- OpenTelemetry
- Prometheus
- Grafana
- centralized logging

## 7. Security and Configuration

Improve:

- provider secrets through secret managers
- environment separation
- non-root runtime user where practical
- stricter port exposure for production
- API authentication/authorization
- rate limiting
- audit logging

## 8. Production Deployment

Move from local Compose toward a production orchestrator only when justified.

Possible target environments:

- Azure AKS
- AWS EKS
- AWS ECS
- Kubernetes
- OpenShift

Production deployment should handle:

- replicas
- rolling upgrades
- readiness/liveness
- secrets
- autoscaling
- ingress
- monitoring
- persistent storage
