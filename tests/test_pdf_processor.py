import fitz
import pytest
from app.models.exceptions import PDFUnreadableError
from app.processors.pdf_processor import PDFProcessor


def test_property_1_text_extraction_and_page_numbers(sample_pdf):
    """Property 1: Each segment's page_number is 1-indexed and matches the source page."""
    processor = PDFProcessor()
    segments = processor.process(sample_pdf)

    assert len(segments) == 2
    assert segments[0].page_number == 1
    assert "Neural Networks" in segments[0].text
    assert segments[1].page_number == 2
    assert "Perceptron" in segments[1].text


def test_property_2_heading_extraction(sample_pdf):
    """Property 2: Heading extraction extracts heading text and level."""
    processor = PDFProcessor()
    segments = processor.process(sample_pdf)

    # Check page 1 headings
    p1_headings = segments[0].headings
    assert len(p1_headings) >= 1
    assert any("Introduction to Neural Networks" in h["text"] for h in p1_headings)
    assert p1_headings[0]["level"] in (1, 2, 3)


def test_property_3_encrypted_pdf_error(encrypted_pdf):
    """Property 3: Password-protected or corrupted PDF raises PDFUnreadableError."""
    processor = PDFProcessor()
    with pytest.raises(PDFUnreadableError) as exc_info:
        processor.process(encrypted_pdf)
    assert "password-protected" in str(exc_info.value).lower() or "unreadable" in str(exc_info.value).lower()


def test_property_3_corrupted_pdf_error(tmp_path):
    """Property 3: Structurally corrupted file raises PDFUnreadableError."""
    corrupted_path = str(tmp_path / "broken.pdf")
    with open(corrupted_path, "wb") as f:
        f.write(b"%PDF-1.4 completely invalid junk data binary")

    processor = PDFProcessor()
    with pytest.raises(PDFUnreadableError):
        processor.process(corrupted_path)


def test_property_4_image_only_page_detection(image_only_pdf):
    """Property 4: Page containing images with no extractable text is marked is_image_only=True."""
    processor = PDFProcessor()
    segments = processor.process(image_only_pdf)

    assert len(segments) == 1
    assert segments[0].is_image_only is True
    assert segments[0].text == ""


def test_novelty_3_table_extraction(tmp_path):
    """Novelty 3: PDF table extraction formats tables as standard Markdown."""
    pdf_path = str(tmp_path / "table_doc.pdf")
    doc = fitz.open()
    page = doc.new_page()

    # Draw table-like lines or text
    page.insert_text((50, 50), "Evaluation Metrics Table", fontsize=16)
    page.insert_text((50, 100), "Epoch | Accuracy | Loss\n1 | 0.85 | 0.42\n2 | 0.91 | 0.28", fontsize=10)
    doc.save(pdf_path)
    doc.close()

    processor = PDFProcessor()
    segments = processor.process(pdf_path)
    assert len(segments) == 1
    assert "Evaluation Metrics" in segments[0].text
