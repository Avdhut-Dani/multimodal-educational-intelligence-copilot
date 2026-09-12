import os
from typing import Any, Dict, List
from app.celery_app import celery_app
from app.config.settings import get_settings
from app.models.exceptions import (
    IngestionError,
    PDFUnreadableError,
    PublishingError,
    TranscriptUnavailableError,
    VideoInaccessibleError,
)
from app.models.schemas import JobStatus
from app.pipeline.chunker import Chunker
from app.pipeline.embedding_generator import EmbeddingGenerator
from app.pipeline.metadata_enricher import MetadataEnricher
from app.pipeline.publisher import VectorDBPublisher
from app.processors.pdf_processor import PDFProcessor
from app.processors.video_processor import VideoProcessor
from app.processors.youtube_processor import YouTubeProcessor
from app.utils.progress import get_progress_tracker


@celery_app.task(bind=True, name="app.tasks.process_pdf")
def process_pdf(self, file_path: str, job_id: str = None):
    """Asynchronously processes an uploaded PDF file through the complete ingestion pipeline."""
    task_id = job_id or self.request.id or "job-pdf"
    tracker = get_progress_tracker()
    settings = get_settings()

    try:
        # Stage 0: Queued
        tracker.update_progress(task_id, 0, JobStatus.QUEUED)

        # Stage 1: Extraction (25%)
        tracker.update_progress(task_id, 25, JobStatus.PROCESSING)
        pdf_processor = PDFProcessor(settings=settings)
        segments = pdf_processor.process(file_path)

        # Stage 2: Chunking (50%)
        tracker.update_progress(task_id, 50, JobStatus.PROCESSING)
        chunker = Chunker(chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)

        all_chunks_with_meta: List[Dict[str, Any]] = []
        filename = os.path.basename(file_path)

        for seg in segments:
            if seg.is_image_only or not seg.text.strip():
                continue

            page_chunks = chunker.chunk(seg.text)
            for idx, c in enumerate(page_chunks):
                all_chunks_with_meta.append({
                    "text": c.text,
                    "source_meta": {
                        "page_number": seg.page_number,
                        "headings": seg.headings,
                        "filename": filename,
                        "chunk_position": idx,
                        "has_tables": len(seg.tables) > 0,
                    }
                })

        if not all_chunks_with_meta:
            tracker.update_progress(task_id, 100, JobStatus.COMPLETED, chunk_count=0)
            return {"job_id": task_id, "status": "completed", "chunk_count": 0}

        # Stage 3: Embedding (75%)
        tracker.update_progress(task_id, 75, JobStatus.PROCESSING)
        embedder = EmbeddingGenerator(settings=settings)
        chunk_texts = [item["text"] for item in all_chunks_with_meta]
        embeddings = embedder.generate(chunk_texts)

        # Stage 4: Metadata Enrichment & Publishing (100%)
        enricher = MetadataEnricher(settings=settings)
        publisher = VectorDBPublisher(settings=settings)

        records_to_publish: List[Dict[str, Any]] = []
        for i, item in enumerate(all_chunks_with_meta):
            enriched = enricher.enrich(
                chunk_text=item["text"],
                source_type="pdf",
                job_id=task_id,
                source_meta=item["source_meta"],
            )
            enriched["embedding"] = embeddings[i]
            records_to_publish.append(enriched)

        total_published = publisher.publish(records_to_publish)
        tracker.update_progress(task_id, 100, JobStatus.COMPLETED, chunk_count=total_published)
        return {"job_id": task_id, "status": "completed", "chunk_count": total_published}

    except PublishingError as pe:
        tracker.update_progress(task_id, 75, JobStatus.PUBLISH_FAILED, error_details=str(pe))
        raise
    except PDFUnreadableError as ue:
        tracker.update_progress(task_id, 25, JobStatus.FAILED, error_details=str(ue))
        raise
    except Exception as e:
        tracker.update_progress(task_id, 0, JobStatus.FAILED, error_details=str(e))
        raise


@celery_app.task(bind=True, name="app.tasks.process_video")
def process_video(self, file_path: str, job_id: str = None):
    """Asynchronously processes an uploaded MP4 video file through the complete ingestion pipeline."""
    task_id = job_id or self.request.id or "job-video"
    tracker = get_progress_tracker()
    settings = get_settings()

    try:
        # Stage 0: Queued
        tracker.update_progress(task_id, 0, JobStatus.QUEUED)

        # Stage 1: Extraction (25%)
        tracker.update_progress(task_id, 25, JobStatus.PROCESSING)
        video_processor = VideoProcessor(settings=settings)
        segments, video_meta, keyframes = video_processor.process(file_path, job_id=task_id)

        # Stage 2: Chunking (50%)
        tracker.update_progress(task_id, 50, JobStatus.PROCESSING)
        chunker = Chunker(chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)

        filename = os.path.basename(file_path)
        all_chunks_with_meta: List[Dict[str, Any]] = []

        for seg in segments:
            if not seg.text.strip():
                continue
            seg_chunks = chunker.chunk(seg.text)
            for c in seg_chunks:
                all_chunks_with_meta.append({
                    "text": c.text,
                    "source_meta": {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "filename": filename,
                        "frame_reference": seg.frame_reference,
                        "visual_text": seg.visual_text,
                    }
                })

        if not all_chunks_with_meta:
            tracker.update_progress(task_id, 100, JobStatus.COMPLETED, chunk_count=0)
            return {"job_id": task_id, "status": "completed", "chunk_count": 0}

        # Stage 3: Embedding (75%)
        tracker.update_progress(task_id, 75, JobStatus.PROCESSING)
        embedder = EmbeddingGenerator(settings=settings)
        embeddings = embedder.generate([item["text"] for item in all_chunks_with_meta])

        # Stage 4: Metadata Enrichment & Publishing (100%)
        enricher = MetadataEnricher(settings=settings)
        publisher = VectorDBPublisher(settings=settings)

        records_to_publish: List[Dict[str, Any]] = []
        for i, item in enumerate(all_chunks_with_meta):
            enriched = enricher.enrich(
                chunk_text=item["text"],
                source_type="mp4",
                job_id=task_id,
                source_meta=item["source_meta"],
            )
            enriched["embedding"] = embeddings[i]
            records_to_publish.append(enriched)

        total_published = publisher.publish(records_to_publish)
        tracker.update_progress(task_id, 100, JobStatus.COMPLETED, chunk_count=total_published)
        return {"job_id": task_id, "status": "completed", "chunk_count": total_published}

    except PublishingError as pe:
        tracker.update_progress(task_id, 75, JobStatus.PUBLISH_FAILED, error_details=str(pe))
        raise
    except Exception as e:
        tracker.update_progress(task_id, 0, JobStatus.FAILED, error_details=str(e))
        raise


@celery_app.task(bind=True, name="app.tasks.process_youtube")
def process_youtube(self, url: str, language: str = "en", job_id: str = None):
    """Asynchronously processes a YouTube URL through the complete ingestion pipeline."""
    task_id = job_id or self.request.id or "job-youtube"
    tracker = get_progress_tracker()
    settings = get_settings()

    try:
        # Stage 0: Queued
        tracker.update_progress(task_id, 0, JobStatus.QUEUED)

        # Stage 1: Extraction (25%)
        tracker.update_progress(task_id, 25, JobStatus.PROCESSING)
        yt_processor = YouTubeProcessor()
        segments, video_meta = yt_processor.process(url=url, language=language)

        # Stage 2: Chunking (50%)
        tracker.update_progress(task_id, 50, JobStatus.PROCESSING)
        chunker = Chunker(chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)

        # Aggregate transcript segments into chunk-sized units preserving timestamp mapping
        all_chunks_with_meta: List[Dict[str, Any]] = []

        current_group_text = []
        group_start_time = 0.0
        group_duration = 0.0

        for seg in segments:
            if not current_group_text:
                group_start_time = seg.start_time
            current_group_text.append(seg.text)
            group_duration += seg.duration

            if len(" ".join(current_group_text).split()) >= settings.CHUNK_SIZE:
                grouped_text = " ".join(current_group_text)
                sub_chunks = chunker.chunk(grouped_text)
                for sc in sub_chunks:
                    all_chunks_with_meta.append({
                        "text": sc.text,
                        "source_meta": {
                            "start_time": group_start_time,
                            "duration": round(group_duration, 2),
                            "title": video_meta.get("title", ""),
                            "channel": video_meta.get("channel", ""),
                            "url": url,
                            "publish_date": video_meta.get("publication_date", ""),
                        }
                    })
                current_group_text = []
                group_duration = 0.0

        if current_group_text:
            grouped_text = " ".join(current_group_text)
            sub_chunks = chunker.chunk(grouped_text)
            for sc in sub_chunks:
                all_chunks_with_meta.append({
                    "text": sc.text,
                    "source_meta": {
                        "start_time": group_start_time,
                        "duration": round(group_duration, 2),
                        "title": video_meta.get("title", ""),
                        "channel": video_meta.get("channel", ""),
                        "url": url,
                        "publish_date": video_meta.get("publication_date", ""),
                    }
                })

        if not all_chunks_with_meta:
            tracker.update_progress(task_id, 100, JobStatus.COMPLETED, chunk_count=0)
            return {"job_id": task_id, "status": "completed", "chunk_count": 0}

        # Stage 3: Embedding (75%)
        tracker.update_progress(task_id, 75, JobStatus.PROCESSING)
        embedder = EmbeddingGenerator(settings=settings)
        embeddings = embedder.generate([item["text"] for item in all_chunks_with_meta])

        # Stage 4: Metadata Enrichment & Publishing (100%)
        enricher = MetadataEnricher(settings=settings)
        publisher = VectorDBPublisher(settings=settings)

        records_to_publish: List[Dict[str, Any]] = []
        for i, item in enumerate(all_chunks_with_meta):
            enriched = enricher.enrich(
                chunk_text=item["text"],
                source_type="youtube",
                job_id=task_id,
                source_meta=item["source_meta"],
            )
            enriched["embedding"] = embeddings[i]
            records_to_publish.append(enriched)

        total_published = publisher.publish(records_to_publish)
        tracker.update_progress(task_id, 100, JobStatus.COMPLETED, chunk_count=total_published)
        return {"job_id": task_id, "status": "completed", "chunk_count": total_published}

    except PublishingError as pe:
        tracker.update_progress(task_id, 75, JobStatus.PUBLISH_FAILED, error_details=str(pe))
        raise
    except (TranscriptUnavailableError, VideoInaccessibleError) as ue:
        tracker.update_progress(task_id, 25, JobStatus.FAILED, error_details=str(ue))
        raise
    except Exception as e:
        tracker.update_progress(task_id, 0, JobStatus.FAILED, error_details=str(e))
        raise

