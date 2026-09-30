import logging
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

logger = logging.getLogger(__name__)
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
    try:
        process_pdf.delay(file_path=file_path, job_id=job_id)
    except Exception as e:
        logger.warning(f"Task execution for PDF job {job_id} encountered error: {e}")

    job = tracker.get_job(job_id)
    current_status = job.status if job else JobStatus.QUEUED
    return JobResponse(job_id=job_id, status=current_status)


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
    try:
        process_video.delay(file_path=file_path, job_id=job_id)
    except Exception as e:
        logger.warning(f"Task execution for video job {job_id} encountered error: {e}")

    job = tracker.get_job(job_id)
    current_status = job.status if job else JobStatus.QUEUED
    return JobResponse(job_id=job_id, status=current_status)


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
    try:
        process_youtube.delay(url=url_str, language=request.language, job_id=job_id)
    except Exception as e:
        logger.warning(f"Task execution for YouTube job {job_id} encountered error: {e}")

    job = tracker.get_job(job_id)
    current_status = job.status if job else JobStatus.QUEUED
    return JobResponse(job_id=job_id, status=current_status)


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


@router.get(
    "/api/catalog/points",
    summary="Inspect vectorized points and metadata stored in Qdrant",
)
async def get_catalog_points(
    job_id: Optional[str] = Query(default=None, description="Optional job ID filter"),
    limit: int = Query(default=50, ge=1, le=200, description="Max points to return"),
    with_vectors: bool = Query(default=True, description="Include raw 384-d vector embeddings in response"),
):
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    try:
        if settings.QDRANT_MODE == "embedded":
            client = QdrantClient(path=settings.QDRANT_PATH)
        else:
            client = QdrantClient(url=settings.QDRANT_URL)

        q_filter = None
        if job_id:
            q_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.job_id",
                        match=qmodels.MatchValue(value=job_id),
                    )
                ]
            )

        scroll_res = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=q_filter,
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
        )
        col_info = client.get_collection(settings.QDRANT_COLLECTION)

        points_data = []
        for pt in scroll_res[0]:
            pt_dict = {
                "id": str(pt.id),
                "text": pt.payload.get("text", "") if pt.payload else "",
                "metadata": pt.payload.get("metadata", {}) if pt.payload else {},
            }
            if with_vectors and pt.vector is not None:
                vec = pt.vector if isinstance(pt.vector, list) else list(pt.vector)
                pt_dict["vector_dim"] = len(vec)
                pt_dict["vector"] = [round(v, 4) for v in vec]
            points_data.append(pt_dict)

        return {
            "collection": settings.QDRANT_COLLECTION,
            "total_points": col_info.points_count,
            "vector_dim": col_info.config.params.vectors.size,
            "returned_count": len(points_data),
            "points": points_data,
        }
    except Exception as e:
        return {
            "collection": settings.QDRANT_COLLECTION,
            "total_points": 0,
            "vector_dim": settings.EMBEDDING_DIM,
            "returned_count": 0,
            "points": [],
            "error": str(e),
        }


@router.get(
    "/frames/{filename}",
    summary="Serve video keyframe snapshot images",
)
async def get_frame_image(filename: str):
    from fastapi.responses import FileResponse
    safe_name = os.path.basename(filename)
    frame_path = os.path.join(settings.FRAME_DIR, safe_name)
    if not os.path.exists(frame_path):
        raise HTTPException(status_code=404, detail="Keyframe image not found")
    return FileResponse(frame_path, media_type="image/jpeg")
