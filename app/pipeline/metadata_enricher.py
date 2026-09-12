import uuid
from datetime import datetime, timezone
from typing import Any, Dict
from app.config.settings import get_settings


class MetadataEnricher:
    """Enriches vectorized chunks with common provenance and source-specific locators."""

    def __init__(self, model_name: str = None, model_version: str = None, settings=None):
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.EMBEDDING_MODEL_NAME
        self.model_version = model_version or self.settings.EMBEDDING_MODEL_VERSION

    def enrich(self, chunk_text: str, source_type: str, job_id: str, source_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Produces a standardized enriched record matching Team 4b's contract."""
        base_metadata = {
            "chunk_id": str(uuid.uuid4()),
            "source_type": source_type,  # "pdf", "mp4", "youtube"
            "job_id": job_id,
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "embedding_model": self.model_name,
            "embedding_model_version": self.model_version,
        }

        if source_type == "pdf":
            base_metadata.update({
                "page_number": int(source_meta.get("page_number", 1)),
                "headings": source_meta.get("headings", []),
                "filename": str(source_meta.get("filename", "")),
                "chunk_position": int(source_meta.get("chunk_position", 0)),
                "has_tables": bool(source_meta.get("has_tables", False)),
            })
        elif source_type == "mp4":
            base_metadata.update({
                "start_timestamp": float(source_meta.get("start_time", 0.0)),
                "end_timestamp": float(source_meta.get("end_time", 0.0)),
                "filename": str(source_meta.get("filename", "")),
                "frame_reference": source_meta.get("frame_reference"),
                "visual_slide_text": source_meta.get("visual_text"),
            })
        elif source_type == "youtube":
            base_metadata.update({
                "start_timestamp": float(source_meta.get("start_time", 0.0)),
                "duration": float(source_meta.get("duration", 0.0)),
                "video_title": str(source_meta.get("title", "")),
                "channel_name": str(source_meta.get("channel", "")),
                "source_url": str(source_meta.get("url", "")),
                "publication_date": str(source_meta.get("publish_date", "")),
            })

        return {
            "text": chunk_text,
            "metadata": base_metadata,
        }

