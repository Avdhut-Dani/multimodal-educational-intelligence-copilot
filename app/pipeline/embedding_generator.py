from typing import List, Union
import numpy as np
from app.config.settings import get_settings

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingGenerator:
    """Generates 384-dimensional L2-normalized dense embeddings using sentence-transformers."""

    def __init__(self, model_name: str = None, batch_size: int = None, settings=None):
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.EMBEDDING_MODEL_NAME
        self.batch_size = batch_size or self.settings.EMBEDDING_BATCH_SIZE
        self.embedding_dim = self.settings.EMBEDDING_DIM
        self._model = None

    @property
    def model(self):
        if self._model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def max_tokens(self) -> int:
        if self.model is not None:
            return getattr(self.model, "max_seq_length", 256)
        return 256

    def generate(self, chunks: List[str]) -> List[np.ndarray]:
        """Generates L2-normalized embeddings for a list of text strings in batches."""
        if not chunks:
            return []

        # Truncate text that exceeds max token length (approx 4 chars per token)
        char_limit = self.max_tokens * 4
        truncated_chunks = [c[:char_limit] if len(c) > char_limit else c for c in chunks]

        # If sentence_transformers model is loaded, compute actual embeddings
        if self.model is not None:
            all_embeddings = []
            for i in range(0, len(truncated_chunks), self.batch_size):
                batch = truncated_chunks[i: i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    normalize_embeddings=True,  # Built-in L2 normalization
                )
                all_embeddings.extend([np.array(e, dtype=np.float32) for e in batch_embeddings])
            return all_embeddings

        # Deterministic fallback for environments without model loaded (e.g. testing)
        # Generates deterministic 384-dim L2-normalized vector from text hash
        fallback_embeddings = []
        for text in truncated_chunks:
            # Simple deterministic pseudo-random seed from string
            seed = sum(ord(c) for c in text) % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.embedding_dim).astype(np.float32)
            # L2 normalize: ||v|| = 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            fallback_embeddings.append(vec)

        return fallback_embeddings

