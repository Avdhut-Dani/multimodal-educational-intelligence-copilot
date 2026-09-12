import time
from unittest.mock import MagicMock, call, patch
import numpy as np
import pytest
from app.models.exceptions import PublishingError
from app.pipeline.metadata_enricher import MetadataEnricher
from app.pipeline.publisher import VectorDBPublisher


def test_property_20_and_21_metadata_completeness():
    """Properties 20 & 21: Common and source-specific metadata fields are fully populated."""
    enricher = MetadataEnricher(model_name="all-MiniLM-L6-v2", model_version="v2.2.0")

    # 1. PDF metadata check
    pdf_meta = {
        "page_number": 3,
        "headings": [{"text": "Intro", "level": 1}],
        "filename": "textbook.pdf",
        "chunk_position": 1,
        "has_tables": False,
    }
    pdf_record = enricher.enrich("PDF chunk text", "pdf", "job-101", pdf_meta)
    meta = pdf_record["metadata"]

    assert meta["source_type"] == "pdf"
    assert meta["job_id"] == "job-101"
    assert meta["page_number"] == 3
    assert meta["headings"] == [{"text": "Intro", "level": 1}]
    assert meta["filename"] == "textbook.pdf"
    assert meta["chunk_position"] == 1
    assert "chunk_id" in meta
    assert "ingestion_timestamp" in meta

    # 2. MP4 video metadata check
    mp4_meta = {
        "start_time": 45.0,
        "end_time": 65.0,
        "filename": "lecture.mp4",
        "frame_reference": "/frames/kf1.jpg",
        "visual_text": "Slide text here",
    }
    mp4_record = enricher.enrich("Video chunk text", "mp4", "job-102", mp4_meta)
    v_meta = mp4_record["metadata"]

    assert v_meta["source_type"] == "mp4"
    assert v_meta["start_timestamp"] == 45.0
    assert v_meta["end_timestamp"] == 65.0
    assert v_meta["filename"] == "lecture.mp4"
    assert v_meta["frame_reference"] == "/frames/kf1.jpg"
    assert v_meta["visual_slide_text"] == "Slide text here"

    # 3. YouTube metadata check
    yt_meta = {
        "start_time": 10.0,
        "duration": 25.5,
        "title": "Calculus 1",
        "channel": "MathAcademy",
        "url": "https://youtube.com/watch?v=123",
        "publish_date": "20260101",
    }
    yt_record = enricher.enrich("YouTube chunk text", "youtube", "job-103", yt_meta)
    y_meta = yt_record["metadata"]

    assert y_meta["source_type"] == "youtube"
    assert y_meta["start_timestamp"] == 10.0
    assert y_meta["duration"] == 25.5
    assert y_meta["video_title"] == "Calculus 1"
    assert y_meta["channel_name"] == "MathAcademy"
    assert y_meta["source_url"] == "https://youtube.com/watch?v=123"


def test_property_23_and_25_publisher_batching_and_chunk_count():
    """Properties 23 & 25: Publisher batches at most 100 records and returns exact published count."""
    mock_client = MagicMock()
    mock_collections = MagicMock()
    mock_collections.collections = []
    mock_client.get_collections.return_value = mock_collections

    publisher = VectorDBPublisher(
        client=mock_client,
        collection_name="test_col",
        batch_size=50,
    )

    # Prepare 120 dummy records
    records = []
    for i in range(120):
        records.append({
            "text": f"Chunk text {i}",
            "embedding": np.zeros(384, dtype=np.float32),
            "metadata": {"chunk_id": f"chunk-{i}", "source_type": "pdf"},
        })

    count = publisher.publish(records)
    assert count == 120
    # Batch size is 50, so 120 items should result in 3 upsert calls (50 + 50 + 20)
    assert mock_client.upsert.call_count == 3


def test_property_24_retry_exponential_backoff():
    """Property 24: Vector DB publisher retries 3 times with exponential backoff before failing."""
    mock_client = MagicMock()
    mock_collections = MagicMock()
    mock_collections.collections = []
    mock_client.get_collections.return_value = mock_collections

    # Always raise error on upsert
    mock_client.upsert.side_effect = RuntimeError("Database connection timed out")

    publisher = VectorDBPublisher(
        client=mock_client,
        collection_name="test_col",
        max_retries=3,
        initial_backoff=0.01,  # Speed up test sleep time
    )

    records = [{
        "text": "Faulty chunk",
        "embedding": np.zeros(384, dtype=np.float32),
        "metadata": {"chunk_id": "err-1"},
    }]

    with pytest.raises(PublishingError):
        publisher.publish(records)

    # Verify exactly 3 attempts were made
    assert mock_client.upsert.call_count == 3
