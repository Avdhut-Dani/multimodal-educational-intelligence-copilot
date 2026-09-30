import os
from functools import lru_cache
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Multi-Modal Educational Data Ingestion Subsystem (Team 4a)"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on", "debug")
        return bool(v)

    # Qdrant Vector DB Configuration
    # Modes: "embedded" (stored on disk/memory, ideal for local lightweight dev) or "server" (connects to Qdrant service)
    QDRANT_MODE: str = "embedded"
    QDRANT_PATH: str = "./data/qdrant_storage"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "education_materials"

    # Celery & Redis Task Broker
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_BACKEND_URL: str = "redis://localhost:6379/1"
    CELERY_ALWAYS_EAGER: bool = True  # True runs tasks synchronously in process for lightweight local dev without Redis daemon
    CELERY_VISIBILITY_TIMEOUT: int = 60
    CELERY_WORKER_CONCURRENCY: int = 10

    # Chunking & Embedding Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_VERSION: str = "v2.2.0"
    EMBEDDING_DIM: int = 384
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    EMBEDDING_BATCH_SIZE: int = 32

    # Publisher & Retry
    PUBLISH_BATCH_SIZE: int = 100
    PUBLISH_MAX_RETRIES: int = 3
    PUBLISH_INITIAL_BACKOFF: float = 2.0

    # Video & Keyframe Settings
    KEYFRAME_INTERVAL: int = 30  # seconds (Requirement 2.3: default 30s)
    WHISPER_MODEL_SIZE: str = "tiny"  # "tiny", "base", "small" for local fast inference
    GROQ_API_KEY: Optional[str] = None  # Optional Groq Cloud Whisper API key for high-speed transcription
    GROQ_WHISPER_MODEL: str = "whisper-large-v3"
    MAX_VIDEO_SIZE_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB (Requirement 2.7)
    MAX_VIDEO_DURATION_HOURS: float = 4.0               # 4 hours (Requirement 2.7)

    # PDF Settings
    MAX_PDF_SIZE_BYTES: int = 200 * 1024 * 1024  # 200 MB
    MAX_PDF_PAGES: int = 5000

    # Feature Flags
    SLIDE_OCR_ENABLED: bool = False              # Set to False to focus strictly on company 30s keyframe requirement
    PDF_TABLE_EXTRACTION_ENABLED: bool = True    # Academic Table-to-Markdown preservation
    SCENE_DETECTION_ENABLED: bool = False        # Set to False for strict 30s interval frame capture

    # Storage Paths
    UPLOAD_DIR: str = "./storage/uploads"
    FRAME_DIR: str = "./storage/frames"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.FRAME_DIR, exist_ok=True)
    if settings.QDRANT_MODE == "embedded":
        os.makedirs(settings.QDRANT_PATH, exist_ok=True)
    return settings

