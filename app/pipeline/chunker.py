import re
from typing import List
from app.models.schemas import Chunk


class Chunker:
    """Normalizes text and splits into sentence-boundary-preserving chunks with token overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def normalize(self, text: str) -> str:
        """Idempotent normalization: strips control characters, collapses whitespace, ensures UTF-8."""
        if not text:
            return ""
        # Remove ASCII control characters except standard tab/newlines, then collapse all whitespace
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Ensure valid UTF-8
        return cleaned.encode("utf-8", errors="ignore").decode("utf-8")

    def chunk(self, text: str) -> List[Chunk]:
        """Splits normalized text into chunks preserving sentence boundaries and overlap."""
        normalized = self.normalize(text)
        if not normalized:
            return []

        words = normalized.split()
        if not words:
            return []

        # If total tokens are less than or equal to chunk_size, emit single chunk without padding
        if len(words) <= self.chunk_size:
            return [Chunk(text=normalized, chunk_index=0, total_chunks=1)]

        # Split into sentences preserving terminal punctuation (. ! ?)
        sentences = self._split_sentences(normalized)
        raw_chunks = self._build_sentence_chunks(sentences)

        total = len(raw_chunks)
        return [
            Chunk(text=chunk_text, chunk_index=i, total_chunks=total)
            for i, chunk_text in enumerate(raw_chunks)
        ]

    def _split_sentences(self, text: str) -> List[str]:
        """Splits on terminal punctuation (. ! ?) followed by whitespace."""
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw_sentences if s.strip()]

    def _build_sentence_chunks(self, sentences: List[str]) -> List[str]:
        """Builds chunks adhering to chunk_size and overlap while respecting sentence boundaries."""
        chunks: List[str] = []
        current_sentences: List[str] = []
        current_token_count = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = sentence.split()
            sentence_len = len(sentence_tokens)

            # If a single sentence exceeds chunk_size on its own, split it by words with overlap
            if sentence_len > self.chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_token_count = 0

                # Break the oversized sentence into sub-chunks
                step = self.chunk_size - self.overlap
                for start_idx in range(0, sentence_len, step):
                    sub_tokens = sentence_tokens[start_idx: start_idx + self.chunk_size]
                    if sub_tokens:
                        chunks.append(" ".join(sub_tokens))
                    if start_idx + self.chunk_size >= sentence_len:
                        break
                i += 1
                continue

            # Check if adding this sentence exceeds chunk_size
            if current_token_count + sentence_len > self.chunk_size and current_sentences:
                # Emit current chunk
                chunks.append(" ".join(current_sentences))

                # Backtrack to satisfy overlap tokens
                overlap_sentences: List[str] = []
                overlap_count = 0
                for s in reversed(current_sentences):
                    s_tokens = len(s.split())
                    if overlap_count + s_tokens <= self.overlap or not overlap_sentences:
                        overlap_sentences.insert(0, s)
                        overlap_count += s_tokens
                    else:
                        break

                current_sentences = list(overlap_sentences)
                current_token_count = overlap_count

            current_sentences.append(sentence)
            current_token_count += sentence_len
            i += 1

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

