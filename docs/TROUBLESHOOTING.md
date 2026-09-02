# Troubleshooting Notes

This file documents real issues encountered during Dockerization and the reasoning behind each fix.

## 1. Dockerfile Not Found

Error:

```text
failed to read dockerfile: open Dockerfile.api: no such file or directory
```

Cause:

The file was named:

```text
DockerFile.api
```

while Compose expected:

```text
Dockerfile.api
```

Fix:

Rename the file so the name matches the Compose configuration exactly.

## 2. UI Requirements File Not Found

Error:

```text
COPY requirements-ui.txt .
"/requirements-ui.txt": not found
```

Cause:

The file had been created as:

```text
requirements-ui.text
```

instead of:

```text
requirements-ui.txt
```

Fix:

Rename the file and rebuild the UI image.

## 3. Existing Qdrant Volume Name Differed Between Machines

One machine used:

```text
insurance_rag_qdrant_storage
```

Another used:

```text
insurance_qdrant_storage
```

The collection contained important existing data, so recreating the database was not desirable.

Fix:

Make the external volume name configurable:

```text
QDRANT_VOLUME_NAME
```

and keep the machine-specific value in `.env`.

## 4. Qdrant Reported Unhealthy Even Though the API Worked

Symptoms:

- direct Qdrant REST calls succeeded
- collection status was green
- 1949 points were present
- Docker still marked Qdrant as unhealthy

Health inspection showed:

```text
exec: "wget": executable file not found in $PATH
```

Cause:

The health check attempted to use `wget`, but the Qdrant image did not include it.

Important lesson:

```text
Application healthy != health-check command healthy
```

Fix:

Replace the `wget` health check with a Bash TCP check against port `6333`.

## 5. FastAPI Failed When Using Host Ollama

Error:

```text
ValueError: Only a local Ollama endpoint is permitted.
Use 127.0.0.1, localhost or ::1.
```

Cause:

The application had intentionally restricted Ollama to localhost addresses.

Inside Docker:

```text
127.0.0.1
```

means the FastAPI container itself, not the Windows host.

The Compose configuration therefore used:

```text
host.docker.internal
```

but the application security whitelist rejected that hostname.

Fix:

Keep the localhost restriction and add:

```text
host.docker.internal
```

to the allowed host list.

The validation was extended rather than disabled.

## 6. Why `host.docker.internal` Was Needed

Normal local execution:

```text
FastAPI -> 127.0.0.1:11434 -> Ollama
```

Docker execution:

```text
FastAPI container
      |
      v
host.docker.internal:11434
      |
      v
Windows Ollama
```

The hostname is used by Docker Desktop to allow a container to reach a service running on the host.

## 7. `.env` Safety Lesson

An existing `.env` should never be recreated blindly with a force option.

Secrets and machine-specific configuration belong in `.env`, and `.env` should be Git-ignored.

Safer workflow:

1. Check whether `.env` already exists.
2. Open and edit the existing file.
3. Never overwrite it just to add one configuration value.
4. Keep `.env.example` in Git with placeholders only.

## 8. Qdrant Persistence Test

The stack was stopped and recreated:

```powershell
docker compose down
docker compose up -d
```

After recreation:

```text
status       = green
points_count = 1949
```

This confirmed that the Qdrant named volume was being reused correctly.

## 9. Large FastAPI Docker Image

Observed image size:

```text
~9.21 GB
```

Cause:

The generic PyTorch package installed CUDA/NVIDIA libraries even though development inference was CPU-oriented.

This is not a functional failure, but it is a production optimization target.

Planned improvement:

- CPU-only PyTorch build
- more tightly pinned dependencies
- potentially multi-stage build
- image vulnerability/size review
