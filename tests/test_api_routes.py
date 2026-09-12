from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest
from app.main import app
from app.models.schemas import JobStatus
from app.utils.progress import get_progress_tracker

client = TestClient(app)


def test_property_28_api_content_type_validation():
    """Property 28: Reject file with invalid content type with HTTP 422."""
    # Send text file to PDF endpoint
    response = client.post(
        "/ingest/pdf",
        files={"file": ("test.txt", b"plain text content", "text/plain")},
    )
    assert response.status_code == 422
    assert "Expected 'application/pdf'" in response.text

    # Send text file to Video endpoint
    response_video = client.post(
        "/ingest/video",
        files={"file": ("test.txt", b"plain text content", "text/plain")},
    )
    assert response_video.status_code == 422
    assert "Expected 'video/mp4'" in response_video.text


def test_property_28_youtube_request_validation():
    """Property 28: Reject invalid YouTube request body with HTTP 422."""
    response = client.post("/ingest/youtube", json={"invalid_key": "not a url"})
    assert response.status_code == 422


def test_property_26_job_id_uniqueness():
    """Property 26: Distinct ingestion submissions generate unique non-colliding job identifiers."""
    with patch("app.tasks.process_youtube.delay"):
        r1 = client.post("/ingest/youtube", json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
        r2 = client.post("/ingest/youtube", json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})

        assert r1.status_code == 200
        assert r2.status_code == 200

        job1 = r1.json()["job_id"]
        job2 = r2.json()["job_id"]

        assert job1 != job2
        assert len(job1) > 0
        assert len(job2) > 0


def test_property_27_progress_stage_mapping():
    """Property 27: Job progress maps correctly: extraction 25%, chunking 50%, embedding 75%, publication 100%."""
    tracker = get_progress_tracker()
    test_job_id = "test-stage-job-1"
    tracker.create_job(test_job_id, "pdf")

    assert tracker.get_job(test_job_id).progress == 0

    tracker.update_progress(test_job_id, 25, JobStatus.PROCESSING)
    assert tracker.get_job(test_job_id).progress == 25

    tracker.update_progress(test_job_id, 50, JobStatus.PROCESSING)
    assert tracker.get_job(test_job_id).progress == 50

    tracker.update_progress(test_job_id, 75, JobStatus.PROCESSING)
    assert tracker.get_job(test_job_id).progress == 75

    tracker.update_progress(test_job_id, 100, JobStatus.COMPLETED, chunk_count=10)
    job = tracker.get_job(test_job_id)
    assert job.progress == 100
    assert job.status == JobStatus.COMPLETED
    assert job.chunk_count == 10


def test_property_29_pagination_correctness():
    """Property 29: GET /jobs pagination returns at most page_size items and correct total count."""
    tracker = get_progress_tracker()

    # Pre-populate 15 distinct jobs
    for i in range(15):
        tracker.create_job(f"paginated-job-{i}", "pdf")

    # Request page 1 with page_size 5
    res1 = client.get("/jobs?page=1&page_size=5")
    assert res1.status_code == 200
    data1 = res1.json()
    assert len(data1["jobs"]) == 5
    assert data1["page"] == 1
    assert data1["page_size"] == 5
    assert data1["total"] >= 15

    # Request page 2 with page_size 5
    res2 = client.get("/jobs?page=2&page_size=5")
    assert res2.status_code == 200
    data2 = res2.json()
    assert len(data2["jobs"]) == 5

    # Check no overlap between page 1 and page 2
    ids_page1 = {j["job_id"] for j in data1["jobs"]}
    ids_page2 = {j["job_id"] for j in data2["jobs"]}
    assert len(ids_page1.intersection(ids_page2)) == 0


def test_health_endpoint():
    """Verify system health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
