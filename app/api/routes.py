import os
import uuid
from typing import Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from app.config.settings import get_settings
from app.models.schemas import (
    IngestionJob,
    JobListResponse,
    JobResponse,
    JobStatus,
    YouTubeRequest,
)
from app.tasks import process_pdf, process_video, process_youtube
from app.utils.progress import get_progress_tracker

router = APIRouter()
settings = get_settings()
tracker = get_progress_tracker()


async def save_uploaded_file(file: UploadFile, prefix: str = "upload") -> str:
    """Saves uploaded multipart file to local storage directory."""
    unique_name = f"{prefix}_{uuid.uuid4().hex[:8]}_{file.filename}"
    target_path = os.path.join(settings.UPLOAD_DIR, unique_name)

    with open(target_path, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
            out_file.write(chunk)

    return target_path


@router.post(
    "/ingest/pdf",
    response_model=JobResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit PDF document for ingestion",
)
async def ingest_pdf(file: UploadFile = File(...)):
    # Property 28: Content-type validation
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid content type '{file.content_type}'. Expected 'application/pdf'.",
        )

    file_path = await save_uploaded_file(file, prefix="pdf")
    job_id = str(uuid.uuid4())

    # Register in progress tracker
    tracker.create_job(job_id=job_id, source_type="pdf", metadata={"filename": file.filename})

    # Dispatch to Celery worker (or eager execution in lightweight mode)
    process_pdf.delay(file_path=file_path, job_id=job_id)

    return JobResponse(job_id=job_id, status=JobStatus.QUEUED)


@router.post(
    "/ingest/video",
    response_model=JobResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit MP4 video for ingestion",
)
async def ingest_video(file: UploadFile = File(...)):
    # Property 28: Content-type validation
    if file.content_type != "video/mp4":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid content type '{file.content_type}'. Expected 'video/mp4'.",
        )

    file_path = await save_uploaded_file(file, prefix="video")
    job_id = str(uuid.uuid4())

    tracker.create_job(job_id=job_id, source_type="mp4", metadata={"filename": file.filename})
    process_video.delay(file_path=file_path, job_id=job_id)

    return JobResponse(job_id=job_id, status=JobStatus.QUEUED)


@router.post(
    "/ingest/youtube",
    response_model=JobResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit YouTube URL for ingestion",
)
async def ingest_youtube(request: YouTubeRequest):
    url_str = str(request.url)
    job_id = str(uuid.uuid4())

    tracker.create_job(job_id=job_id, source_type="youtube", metadata={"url": url_str})
    process_youtube.delay(url=url_str, language=request.language, job_id=job_id)

    return JobResponse(job_id=job_id, status=JobStatus.QUEUED)


@router.get(
    "/jobs/{job_id}",
    response_model=IngestionJob,
    summary="Get status and progress of an ingestion job",
)
async def get_job_status(job_id: str):
    job = tracker.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job '{job_id}' not found",
        )
    return job


@router.get(
    "/jobs",
    response_model=JobListResponse,
    summary="List all ingestion jobs with pagination",
)
async def list_jobs(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
):
    # Property 29: Pagination correctness
    jobs = tracker.list_jobs(page=page, page_size=page_size)
    total = tracker.count_jobs()

    return JobListResponse(
        total=total,
        page=page,
        page_size=page_size,
        jobs=jobs,
    )
