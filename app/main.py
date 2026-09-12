import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.FRAME_DIR, exist_ok=True)
    if settings.QDRANT_MODE == "embedded":
        os.makedirs(settings.QDRANT_PATH, exist_ok=True)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Multi-Modal Educational Intelligence Copilot: Team 4a Data Ingestion Subsystem.\n"
            "Ingests PDFs, MP4 lecture videos, and YouTube URLs into normalized text with dense embeddings "
            "and stores them in Qdrant for downstream RAG retrieval."
        ),
        lifespan=lifespan,
    )

    # Enable CORS for team frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "qdrant_mode": settings.QDRANT_MODE,
            "celery_mode": "eager" if settings.CELERY_ALWAYS_EAGER else "distributed",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
