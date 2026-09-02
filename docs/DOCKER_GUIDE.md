# Docker and Docker Compose Guide

This document records the commands used in the Insurance RAG Assistant and what each one means.

## Core Concepts

```text
Dockerfile
= recipe for building one image

Docker Image
= packaged application template

Container
= running instance of an image

compose.yaml
= configuration describing how multiple services run together

Volume
= persistent data outside disposable containers

Network
= communication path between containers
```

## Validate Compose

```powershell
docker compose config
```

Reads and validates `compose.yaml`.

## Build Images

Build all custom images:

```powershell
docker compose build
```

Build only FastAPI:

```powershell
docker compose build api
```

Build only Gradio:

```powershell
docker compose build ui
```

Force a rebuild without cache:

```powershell
docker compose build --no-cache api
```

Use `--no-cache` only when needed because the API dependency layer is large.

## Start Services

Start the entire stack:

```powershell
docker compose up -d
```

Meaning:

```text
docker   = Docker CLI
compose  = use Docker Compose
up       = create/start services
-d       = detached mode; run in background
```

Start only the UI service:

```powershell
docker compose up -d ui
```

`ui` is the service name from `compose.yaml`.

Because services have dependencies, Compose can also ensure required dependent services are running.

## Process Status

```powershell
docker compose ps
```

`ps` means **process status**, not properties.

It shows the current state of Compose services.

## Logs

All logs:

```powershell
docker compose logs
```

Follow live logs:

```powershell
docker compose logs -f
```

FastAPI only:

```powershell
docker compose logs -f api
```

Qdrant only:

```powershell
docker compose logs -f qdrant
```

Gradio only:

```powershell
docker compose logs -f ui
```

`Ctrl+C` stops following logs; it does not stop the running containers.

## Container Lifecycle

Stop running containers without removing them:

```powershell
docker compose stop
```

Start the same stopped containers:

```powershell
docker compose start
```

Remove Compose containers and the Compose network while preserving named volumes:

```powershell
docker compose down
```

Recreate/start the stack:

```powershell
docker compose up -d
```

Dangerous around persistent data:

```powershell
docker compose down -v
```

`-v` removes volumes. Do not use it casually with Qdrant data.

## Inspect Containers

Running containers:

```powershell
docker ps
```

All containers:

```powershell
docker ps -a
```

Inspect mounts:

```powershell
docker inspect <container-name> --format '{{json .Mounts}}'
```

Inspect health:

```powershell
docker inspect <container-name> --format '{{json .State.Health}}'
```

## Images

```powershell
docker images
```

Project-specific:

```powershell
docker images | Select-String "insurance-rag"
```

## Volumes

```powershell
docker volume ls
```

Inspect a volume:

```powershell
docker volume inspect <volume-name>
```

The Qdrant volume contains persistent vector database data and should not be deleted casually.

## Networks

```powershell
docker network ls
```

Inspect:

```powershell
docker network inspect <network-name>
```

Compose service names act as internal DNS names.

Examples:

```text
Gradio -> http://api:8000
FastAPI -> http://qdrant:6333
```

## Health Checks

FastAPI:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Qdrant collection:

```powershell
Invoke-RestMethod http://127.0.0.1:6333/collections/insurance_policy_chunks_bge_m3_v1 |
ConvertTo-Json -Depth 3
```

## Enter a Container

FastAPI:

```powershell
docker compose exec api bash
```

UI:

```powershell
docker compose exec ui bash
```

Exit:

```text
exit
```

## Disk Usage and Cleanup

Docker disk usage:

```powershell
docker system df
```

Remove unused stopped containers:

```powershell
docker container prune
```

Remove unused images:

```powershell
docker image prune
```

Remove build cache:

```powershell
docker builder prune
```

Be careful with:

```powershell
docker volume prune
docker system prune --volumes
```

because volumes may contain persistent application data.

## Typical Daily Workflow

```powershell
git pull origin main
docker compose config
docker compose build
docker compose up -d
docker compose ps
```

Then use:

```text
http://127.0.0.1:7860
http://127.0.0.1:8000/docs
```

At the end:

```powershell
docker compose down
```
