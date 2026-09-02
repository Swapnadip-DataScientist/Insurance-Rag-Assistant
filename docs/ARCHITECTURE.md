# Architecture

## 1. Objective

The application is designed to answer insurance policy questions using evidence retrieved from policy documents.

The primary architectural principle is:

> Retrieval quality is handled before generation. The LLM is not used as a substitute for evidence retrieval.

## 2. Logical Architecture

```text
                         User
                           |
                           v
                    +-------------+
                    |   Gradio    |
                    |    :7860    |
                    +------+------+
                           |
                           | HTTP POST /ask
                           v
                    +-------------+
                    |   FastAPI   |
                    |    :8000    |
                    +------+------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
       BGE-M3 Embedding              Retrieval
                                         |
                                         v
                                      Qdrant
                                         |
                                         v
                                  Candidate Chunks
                                         |
                                         v
                                  BGE Reranker
                                         |
                                         v
                                    Top Evidence
                                         |
                                         v
                                     Generator
                                         |
                                         v
                             Ollama / Qwen 3.5
                                         |
                                         v
                           Structured Grounded Answer
```

## 3. Request Flow

1. The user enters an insurance question in Gradio.
2. Gradio sends the question to `POST /ask`.
3. FastAPI passes the question to the RAG service.
4. BGE-M3 creates the query embedding.
5. Qdrant retrieves candidate chunks.
6. Retrieval filters remove poor-quality or duplicate candidates.
7. BGE reranker scores candidates against the original query.
8. Only the strongest evidence is supplied to the generator.
9. Qwen produces a structured answer.
10. FastAPI validates and returns the response.
11. Gradio displays the answer and source evidence.

## 4. Why UI and API Are Separate

Gradio is only a client.

It does not directly access:

- Qdrant
- embeddings
- reranking
- Ollama
- retrieval logic

This keeps the application boundary clean:

```text
UI -> API -> RAG Services
```

The UI can later be replaced by React, Angular, mobile, or another enterprise channel without redesigning the backend.

## 5. Docker Architecture

```text
                    Docker Compose Network

        +----------------+      +----------------+
        |     Gradio     | ---> |    FastAPI     |
        |     :7860      |      |     :8000      |
        +----------------+      +-------+--------+
                                        |
                              +---------+----------+
                              |                    |
                              v                    v
                         Qdrant :6333     host.docker.internal
                                                   |
                                                   v
                                           Windows Ollama
                                              :11434
```

Qdrant and FastAPI communicate through Docker service-name DNS:

```text
http://qdrant:6333
```

Gradio reaches FastAPI using:

```text
http://api:8000/ask
```

FastAPI reaches host-based Ollama using:

```text
http://host.docker.internal:11434
```

## 6. Persistent Storage

Qdrant stores its persistent database outside the disposable container filesystem:

```text
Qdrant Container
      |
      v
/qdrant/storage
      |
      v
Named Docker Volume
```

The Hugging Face model cache is also stored in a named volume.

This separates runtime containers from persistent state.

## 7. Health and Readiness

FastAPI provides:

```text
/health/live
/health/ready
```

Liveness answers:

> Is the API process alive?

Readiness answers:

> Has the RAG runtime initialized and is the application ready for traffic?

Qdrant uses a container-level TCP health check on port `6333`.

## 8. Production Evolution

The local architecture can evolve toward:

```text
Users
  |
HTTPS / WAF
  |
API Gateway / Load Balancer
  |
FastAPI Replicas
  |
  +--> Vector Database
  +--> Embedding Service
  +--> Reranking Service
  +--> LLM Inference Service
```

Potential orchestration platforms include Kubernetes, AKS, EKS, ECS, or OpenShift.

The current Compose deployment intentionally avoids unnecessary microservice decomposition while the project remains local and portfolio-oriented.
