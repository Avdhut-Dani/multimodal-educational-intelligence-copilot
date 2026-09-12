import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.models import PointStruct
from app.config.settings import get_settings
from app.models.exceptions import PublishingError


class VectorDBPublisher:
    """Publishes vectorized chunks with metadata to Qdrant with exponential backoff and batching."""

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        initial_backoff: Optional[float] = None,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.collection_name = collection_name or self.settings.QDRANT_COLLECTION
        self.batch_size = batch_size or self.settings.PUBLISH_BATCH_SIZE
        self.max_retries = max_retries or self.settings.PUBLISH_MAX_RETRIES
        self.initial_backoff = initial_backoff or self.settings.PUBLISH_INITIAL_BACKOFF

        if client is not None:
            self.client = client
        else:
            self.client = self._init_client()

        self._ensure_collection()

    def _init_client(self) -> QdrantClient:
        """Initializes Qdrant client based on configured mode (embedded vs server)."""
        if self.settings.QDRANT_MODE == "embedded":
            return QdrantClient(path=self.settings.QDRANT_PATH)
        else:
            return QdrantClient(url=self.settings.QDRANT_URL)

    def _ensure_collection(self):
        """Ensures Qdrant collection exists with proper vector size and Cosine distance."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.settings.EMBEDDING_DIM,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        except Exception:
            # Collection may already exist or client is mock
            pass

    def publish(
        self,
        records: List[Dict[str, Any]],
        on_success: Optional[Callable[[int], None]] = None,
        on_failure: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Publishes records to Qdrant in batches adhering to batch size <= 100 with exponential retry."""
        if not records:
            if on_success:
                on_success(0)
            return 0

        total_published = 0
        total_records = len(records)

        for i in range(0, total_records, self.batch_size):
            batch = records[i: i + self.batch_size]
            points: List[PointStruct] = []

            for r in batch:
                vec = r["embedding"]
                if isinstance(vec, np.ndarray):
                    vec_list = vec.tolist()
                else:
                    vec_list = list(vec)

                points.append(
                    PointStruct(
                        id=r["metadata"]["chunk_id"],
                        vector=vec_list,
                        payload={
                            "text": r["text"],
                            "metadata": r["metadata"],
                        },
                    )
                )

            try:
                self._publish_batch_with_retry(points)
                total_published += len(batch)
            except Exception as e:
                err_msg = f"Vector DB publication failed after {self.max_retries} attempts: {str(e)}"
                if on_failure:
                    on_failure(err_msg)
                raise PublishingError(err_msg)

        if on_success:
            on_success(total_published)

        return total_published

    def _publish_batch_with_retry(self, points: List[PointStruct]):
        """Retries write operation up to max_retries with exponential backoff (2s, 4s, 8s)."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
                return
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    sleep_time = self.initial_backoff * (2 ** attempt)
                    time.sleep(sleep_time)

        raise last_exception or PublishingError("Publication failed")

