from __future__ import annotations

import asyncio
import logging
import uuid

from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from starlette.concurrency import (
    run_in_threadpool,
)

from src.api.config import ApiSettings
from src.api.runtime import build_rag_service
from src.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
)

from src.generation.generator import (
    InvalidModelResponseError,
    ModelNotAvailableError,
    OllamaUnavailableError,
)


LOGGER = logging.getLogger(__name__)


# =============================================================
# CONFIGURATION
# =============================================================

settings = ApiSettings()
settings.validate()


# =============================================================
# APPLICATION LIFESPAN
# =============================================================
#
# Heavy ML/database objects are constructed once before
# requests are accepted and released during shutdown.
#
# They must NOT be reconstructed inside POST /ask.
# =============================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    runtime = build_rag_service(
        settings
    )

    service = runtime.__enter__()

    app.state.rag_service = service

    app.state.inference_semaphore = (
        asyncio.Semaphore(
            settings.max_concurrent_requests
        )
    )

    LOGGER.info(
        "Insurance RAG API ready."
    )

    try:

        yield

    finally:

        runtime.__exit__(
            None,
            None,
            None,
        )

        LOGGER.info(
            "Insurance RAG API stopped."
        )


# =============================================================
# FASTAPI APPLICATION
# =============================================================


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Grounded insurance policy question-answering API "
        "using BGE-M3, Qdrant, BGE reranking and Qwen/Ollama."
    ),
    lifespan=lifespan,
)


# =============================================================
# REQUEST ID MIDDLEWARE
# =============================================================


@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next,
):

    request_id = (
        request.headers.get(
            "X-Request-ID"
        )
        or str(uuid.uuid4())
    )

    response = await call_next(
        request
    )

    response.headers[
        "X-Request-ID"
    ] = request_id

    return response


# =============================================================
# LIVENESS
# =============================================================


@app.get(
    "/health/live",
    response_model=HealthResponse,
)
def health_live() -> HealthResponse:

    return HealthResponse(
        status="ok"
    )


# =============================================================
# READINESS
# =============================================================


@app.get(
    "/health/ready",
    response_model=HealthResponse,
)
def health_ready(
    request: Request,
) -> HealthResponse:

    if not hasattr(
        request.app.state,
        "rag_service",
    ):
        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready.",
        )

    return HealthResponse(
        status="ready"
    )


# =============================================================
# ASK
# =============================================================


@app.post(
    "/ask",
    response_model=AskResponse,
)
async def ask(
    payload: AskRequest,
    request: Request,
) -> AskResponse:

    service = (
        request.app.state.rag_service
    )

    semaphore = (
        request.app.state.inference_semaphore
    )

    try:

        # -----------------------------------------------------
        # CPU inference is blocking work.
        #
        # Do not execute it directly on FastAPI's async
        # event loop.
        # -----------------------------------------------------

        async with semaphore:

            result = await run_in_threadpool(
                service.ask,
                payload.query.strip(),
            )

        return AskResponse.model_validate(
            result
        )

    except OllamaUnavailableError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ModelNotAvailableError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except InvalidModelResponseError as exc:

        # Upstream model responded, but the structured
        # response violated our application contract.
        raise HTTPException(
            status_code=502,
            detail=(
                "The generation model returned "
                "an invalid structured response."
            ),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:

        LOGGER.exception(
            "Unhandled RAG request failure."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The insurance RAG request failed."
            ),
        ) from exc