from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PUBLISH_FAILED = "publish_failed"


class YouTubeRequest(BaseModel):
    url: HttpUrl = Field(..., description="Valid YouTube video URL")
    language: str = Field(default="en", description="Transcript language code (default: en)")


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str = "Job accepted and queued for processing"


class IngestionJob(BaseModel):
    job_id: str
    source_type: str  # "pdf", "mp4", "youtube"
    status: JobStatus
    progress: int = Field(default=0, ge=0, le=100)
    chunk_count: Optional[int] = None
    error_details: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JobListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    jobs: List[IngestionJob]


class Chunk(BaseModel):
    text: str
    chunk_index: int
    total_chunks: int


class PDFHeading(BaseModel):
    text: str
    level: int


class PDFSegment(BaseModel):
    text: str
    page_number: int  # 1-indexed
    headings: List[Dict[str, Any]] = Field(default_factory=list)  # [{"text": str, "level": int}]
    is_image_only: bool = False
    tables: List[str] = Field(default_factory=list)  # Novelty 3: Markdown formatted tables


class VideoSegment(BaseModel):
    text: str
    start_time: float
    end_time: float
    frame_reference: Optional[str] = None
    visual_text: Optional[str] = None  # Novelty 1: Slide OCR text
    status: str = "ok"  # "ok" or "transcription_failed"


class YouTubeSegment(BaseModel):
    text: str
    start_time: float
    duration: float


class KeyframeInfo(BaseModel):
    timestamp: float
    frame_path: str
    is_scene_change: bool = False
    ocr_text: Optional[str] = None
