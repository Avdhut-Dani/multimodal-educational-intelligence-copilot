import os
from typing import Any, Dict, List
try:
    import pymupdf as fitz
except ImportError:
    import fitz
from app.config.settings import get_settings
from app.models.exceptions import PDFUnreadableError
from app.models.schemas import PDFSegment


class PDFProcessor:
    """Extracts text, structural headings, tables (Novelty 3), and image metadata from PDFs."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.max_size_bytes = self.settings.MAX_PDF_SIZE_BYTES
        self.max_pages = self.settings.MAX_PDF_PAGES
        self.table_extraction_enabled = self.settings.PDF_TABLE_EXTRACTION_ENABLED

    def process(self, file_path: str) -> List[PDFSegment]:
        """Processes a PDF file into a list of PDFSegment objects."""
        if not os.path.exists(file_path):
            raise PDFUnreadableError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > self.max_size_bytes:
            raise PDFUnreadableError(f"File size {file_size} exceeds maximum limit {self.max_size_bytes} bytes")

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise PDFUnreadableError(f"Cannot open PDF file: {str(e)}")

        if doc.is_encrypted:
            try:
                # Attempt with empty password; if still encrypted, raise error
                if not doc.authenticate(""):
                    raise PDFUnreadableError("PDF is password-protected and encrypted")
            except Exception:
                raise PDFUnreadableError("PDF is password-protected and encrypted")

        total_pages = len(doc)
        if total_pages > self.max_pages:
            raise PDFUnreadableError(f"Page count {total_pages} exceeds maximum limit {self.max_pages}")

        segments: List[PDFSegment] = []

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-indexed
            page = doc[page_idx]

            # 1. Extract raw text
            text = page.get_text("text") or ""

            # 2. Check for image-only condition
            images = page.get_images()
            is_image_only = (len(text.strip()) == 0) and (len(images) > 0)

            # 3. Extract structural headings
            headings = self._extract_headings(page)

            # 4. Novelty 3: Extract structured tables if enabled
            markdown_tables = []
            if self.table_extraction_enabled and not is_image_only:
                markdown_tables = self._extract_tables_as_markdown(page)
                if markdown_tables:
                    # Append markdown tables to segment text so embeddings and RAG capture tabular relations
                    table_block = "\n\n### Extracted Tables\n" + "\n\n".join(markdown_tables)
                    text = f"{text}\n{table_block}".strip()

            segments.append(
                PDFSegment(
                    text=text if not is_image_only else "",
                    page_number=page_num,
                    headings=headings,
                    is_image_only=is_image_only,
                    tables=markdown_tables,
                )
            )

        doc.close()
        return segments

    def _extract_headings(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Extracts headings using font-size heuristics from text blocks."""
        headings: List[Dict[str, Any]] = []
        try:
            blocks = page.get_text("dict").get("blocks", [])
        except Exception:
            return headings

        # First pass: collect all font sizes to find body text baseline
        font_sizes = []
        for b in blocks:
            if b.get("type") == 0:  # text block
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 0)
                        if size > 0 and span.get("text", "").strip():
                            font_sizes.append(size)

        if not font_sizes:
            return headings

        # Determine body text size using most frequent font size (or min font size)
        from collections import Counter
        counts = Counter(font_sizes)
        # Most common font size; if all equal frequency, use min font size
        max_freq = max(counts.values())
        candidates = [size for size, cnt in counts.items() if cnt == max_freq]
        body_size = min(candidates)

        for b in blocks:
            if b.get("type") == 0:
                for line in b.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                    if not line_text or len(line_text) > 120:  # Headings are concise
                        continue

                    # Average font size in the line
                    sizes = [s.get("size", 0) for s in line.get("spans", []) if s.get("size", 0) > 0]
                    if not sizes:
                        continue
                    line_font_size = sum(sizes) / len(sizes)

                    # Determine heading level based on ratio to body size
                    ratio = line_font_size / body_size if body_size > 0 else 1.0
                    if ratio >= 1.5:
                        headings.append({"text": line_text, "level": 1})
                    elif ratio >= 1.25:
                        headings.append({"text": line_text, "level": 2})
                    elif ratio >= 1.1:
                        headings.append({"text": line_text, "level": 3})

        return headings

    def _extract_tables_as_markdown(self, page: fitz.Page) -> List[str]:
        """Novelty 3: Extracts tables from page and formats as standard Markdown tables."""
        md_tables: List[str] = []
        try:
            tables = page.find_tables()
            if not tables or not tables.tables:
                return md_tables

            for tab in tables:
                df_rows = tab.extract()
                if not df_rows or len(df_rows) < 2:
                    continue

                # Header
                headers = [str(col).replace("\n", " ").strip() if col else f"Col {i+1}" for i, col in enumerate(df_rows[0])]
                separator = ["---"] * len(headers)
                md_lines = [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(separator) + " |",
                ]

                # Rows
                for row in df_rows[1:]:
                    cleaned_row = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
                    # Ensure matching number of cells
                    if len(cleaned_row) < len(headers):
                        cleaned_row.extend([""] * (len(headers) - len(cleaned_row)))
                    md_lines.append("| " + " | ".join(cleaned_row[:len(headers)]) + " |")

                md_tables.append("\n".join(md_lines))
        except Exception:
            # Table extraction is best-effort and non-fatal
            pass

        return md_tables

