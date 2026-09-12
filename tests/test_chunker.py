import pytest
from app.models.schemas import Chunk
from app.pipeline.chunker import Chunker


def test_property_15_normalization_idempotence():
    """Property 15: normalize(normalize(text)) == normalize(text), no control characters or excess spaces."""
    chunker = Chunker()
    raw_text = "  Hello \x00\x05 World!\t\nThis   is   an \x1b educational    lecture.   "

    norm1 = chunker.normalize(raw_text)
    norm2 = chunker.normalize(norm1)

    assert norm1 == norm2
    assert "\x00" not in norm1
    assert "  " not in norm1
    assert norm1 == "Hello World! This is an educational lecture."


def test_property_14_uniform_chunk_output_format():
    """Property 14: Uniform output format: text (non-empty), chunk_index >= 0, chunk_index < total_chunks."""
    chunker = Chunker(chunk_size=10, overlap=2)
    sample_text = (
        "Machine learning is an application of AI. "
        "It provides systems the ability to automatically learn. "
        "Neural networks form the backbone of deep learning. "
        "Supervised learning uses labeled training datasets."
    )

    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 1

    for i, c in enumerate(chunks):
        assert isinstance(c, Chunk)
        assert len(c.text.strip()) > 0
        assert c.chunk_index == i
        assert c.total_chunks == len(chunks)
        assert c.chunk_index < c.total_chunks


def test_property_13_sentence_boundary_preservation():
    """Property 13: Chunks end on terminal sentence punctuation (. ! ?) rather than mid-sentence."""
    chunker = Chunker(chunk_size=15, overlap=3)
    text = (
        "First sentence here. "
        "Second sentence is a bit longer than the first. "
        "Third sentence completes the thought! "
        "Fourth sentence asks a question?"
    )

    chunks = chunker.chunk(text)
    for c in chunks:
        # Each chunk's text should end with a terminal punctuation mark (. ! ?)
        assert c.text.rstrip()[-1] in (".", "!", "?")


def test_property_12_short_text_single_chunk():
    """Requirement 4.4 / Property 12: Short text segment is emitted as a single chunk without padding."""
    chunker = Chunker(chunk_size=512, overlap=50)
    short_text = "This is a brief syllabus announcement."

    chunks = chunker.chunk(short_text)
    assert len(chunks) == 1
    assert chunks[0].text == short_text
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1
