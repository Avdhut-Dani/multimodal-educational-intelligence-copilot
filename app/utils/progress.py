from datetime import datetime, timezone
import json
import threading
from typing import Any, Dict, List, Optional
from app.config.settings import get_settings
from app.models.schemas import IngestionJob, JobStatus

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class JobProgressTracker:
    """Tracks and persists Ingestion Job lifecycle, stages (25%, 50%, 75%, 100%), and metadata."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self._local_registry: Dict[str, Dict[str, Any]] = {}
        self._redis_client = None

    @property
    def redis_client(self):
        if self._redis_client is None and REDIS_AVAILABLE and not self.settings.CELERY_ALWAYS_EAGER:
            try:
                self._redis_client = redis.Redis.from_url(
                    self.settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                self._redis_client.ping()
            except Exception:
                self._redis_client = None
        return self._redis_client

    def create_job(self, job_id: str, source_type: str, metadata: Optional[Dict[str, Any]] = None) -> IngestionJob:
        """Registers a newly submitted ingestion job in queued status."""
        now = datetime.now(timezone.utc)
        job_data = {
            "job_id": job_id,
            "source_type": source_type,
            "status": JobStatus.QUEUED.value,
            "progress": 0,
            "chunk_count": None,
            "error_details": None,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "metadata": metadata or {},
        }

        with self._lock:
            self._local_registry[job_id] = job_data

        if self.redis_client:
            try:
                self.redis_client.set(f"job:{job_id}", json.dumps(job_data))
                self.redis_client.zadd("jobs_index", {job_id: now.timestamp()})
            except Exception:
                pass

        return self._dict_to_job(job_data)

    def update_progress(
        self,
        job_id: str,
        progress: int,
        status: JobStatus = JobStatus.PROCESSING,
        chunk_count: Optional[int] = None,
        error_details: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[IngestionJob]:
        """Updates job progress stage (25% extraction, 50% chunking, 75% embedding, 100% publication)."""
        now = datetime.now(timezone.utc)

        with self._lock:
            job_data = self._local_registry.get(job_id)
            if not job_data:
                # If not in local memory, try fetching from Redis
                if self.redis_client:
                    try:
                        raw = self.redis_client.get(f"job:{job_id}")
                        if raw:
                            job_data = json.loads(raw)
                    except Exception:
                        pass

            if not job_data:
                job_data = {
                    "job_id": job_id,
                    "source_type": "unknown",
                    "created_at": now.isoformat(),
                    "metadata": {},
                }

            job_data["progress"] = progress
            job_data["status"] = status.value if isinstance(status, JobStatus) else str(status)
            job_data["updated_at"] = now.isoformat()
            if chunk_count is not None:
                job_data["chunk_count"] = chunk_count
            if error_details is not None:
                job_data["error_details"] = error_details
            if extra_metadata:
                job_data["metadata"].update(extra_metadata)

            self._local_registry[job_id] = job_data

        if self.redis_client:
            try:
                self.redis_client.set(f"job:{job_id}", json.dumps(job_data))
            except Exception:
                pass

        return self._dict_to_job(job_data)

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        """Retrieves an IngestionJob by its unique ID."""
        with self._lock:
            job_data = self._local_registry.get(job_id)

        if not job_data and self.redis_client:
            try:
                raw = self.redis_client.get(f"job:{job_id}")
                if raw:
                    job_data = json.loads(raw)
                    with self._lock:
                        self._local_registry[job_id] = job_data
            except Exception:
                pass

        if not job_data:
            return None

        return self._dict_to_job(job_data)

    def list_jobs(self, page: int = 1, page_size: int = 20) -> List[IngestionJob]:
        """Returns paginated list of all ingestion jobs."""
        with self._lock:
            all_jobs = list(self._local_registry.values())

        # Sort descending by created_at
        all_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paged_items = all_jobs[start_idx:end_idx]

        return [self._dict_to_job(d) for d in paged_items]

    def count_jobs(self) -> int:
        with self._lock:
            return len(self._local_registry)

    def _dict_to_job(self, data: Dict[str, Any]) -> IngestionJob:
        return IngestionJob(
            job_id=data["job_id"],
            source_type=data.get("source_type", "unknown"),
            status=JobStatus(data["status"]),
            progress=data.get("progress", 0),
            chunk_count=data.get("chunk_count"),
            error_details=data.get("error_details"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


_global_tracker = None


def get_progress_tracker() -> JobProgressTracker:
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = JobProgressTracker()
    return _global_tracker
