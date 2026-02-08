from fastapi.responses import JSONResponse


def embedding_unavailable() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Embedding service unavailable"},
        headers={"Retry-After": "5"},
    )
