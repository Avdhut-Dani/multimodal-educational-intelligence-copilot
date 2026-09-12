import numpy as np
import pytest
from app.pipeline.embedding_generator import EmbeddingGenerator


def test_property_16_embedding_dimensionality():
    """Property 16: Embedding vector has exactly 384 dimensions."""
    generator = EmbeddingGenerator()
    chunks = ["Introduction to convolutional neural networks and computer vision."]
    embeddings = generator.generate(chunks)

    assert len(embeddings) == 1
    assert embeddings[0].shape == (384,)


def test_property_18_l2_normalization():
    """Property 18: L2 norm of each embedding vector equals 1.0 (+- 1e-5)."""
    generator = EmbeddingGenerator()
    chunks = [
        "Backpropagation uses the chain rule of calculus.",
        "Recurrent neural networks process sequential data.",
        "Transformers rely on multi-head self-attention mechanisms.",
    ]
    embeddings = generator.generate(chunks)

    assert len(embeddings) == 3
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, abs=1e-5) == 1.0


def test_property_17_batch_processing_correctness():
    """Property 17: Generates exactly N embeddings for N input chunks."""
    generator = EmbeddingGenerator(batch_size=2)
    chunks = [f"Educational concept lecture chunk number {i}" for i in range(7)]
    embeddings = generator.generate(chunks)

    assert len(embeddings) == 7
    for emb in embeddings:
        assert emb.shape == (384,)


def test_property_19_oversized_input_truncation():
    """Property 19: Extremely long text is truncated safely and yields valid normalized 384-dim vector."""
    generator = EmbeddingGenerator()
    giant_chunk = "Machine learning " * 1000  # Exceeds max tokens
    embeddings = generator.generate([giant_chunk])

    assert len(embeddings) == 1
    assert embeddings[0].shape == (384,)
    assert pytest.approx(np.linalg.norm(embeddings[0]), abs=1e-5) == 1.0

