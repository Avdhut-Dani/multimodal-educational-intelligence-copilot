import os
import shutil
import tempfile
from typing import Generator
import fitz
import pytest
from app.config.settings import Settings, get_settings


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    temp_dir = tempfile.mkdtemp(prefix="test_ingestion_")
    settings = Settings(
        QDRANT_MODE="embedded",
        QDRANT_PATH=os.path.join(temp_dir, "qdrant"),
        UPLOAD_DIR=os.path.join(temp_dir, "uploads"),
        FRAME_DIR=os.path.join(temp_dir, "frames"),
        CELERY_ALWAYS_EAGER=True,
    )
    yield settings
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_pdf(tmp_path) -> str:
    """Generates a valid 2-page PDF document with headings and body text for testing."""
    pdf_path = str(tmp_path / "sample_lecture.pdf")
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        (50, 50),
        "Chapter 1: Introduction to Neural Networks",
        fontsize=24,
    )
    page1.insert_text(
        (50, 90),
        "Artificial neural networks are computing systems inspired by biological brains. "
        "They consist of interconnected nodes or neurons organized in layers. "
        "Information flows through input, hidden, and output units.",
        fontsize=12,
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        (50, 50),
        "Section 1.1: Perceptron Fundamentals",
        fontsize=18,
    )
    page2.insert_text(
        (50, 90),
        "The perceptron is the fundamental building block of artificial neural networks. "
        "It takes multiple inputs, applies weights, adds a bias term, and evaluates an activation function.",
        fontsize=12,
    )

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def image_only_pdf(tmp_path) -> str:
    """Generates a PDF page with an image and no text."""
    pdf_path = str(tmp_path / "image_only.pdf")
    doc = fitz.open()
    page = doc.new_page()

    # Create a small pixmap image and insert it
    pix = fitz.Pixmap(fitz.csRGB, (0, 0, 100, 100), 0)
    pix.clear_with(255)  # White box
    page.insert_image(page.rect, pixmap=pix)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def encrypted_pdf(tmp_path) -> str:
    """Generates an encrypted password-protected PDF."""
    pdf_path = str(tmp_path / "encrypted.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Classified notes", fontsize=12)

    perm = int(
        fitz.PDF_PERM_ACCESSIBILITY
        | fitz.PDF_PERM_PRINT
        | fitz.PDF_PERM_COPY
        | fitz.PDF_PERM_ANNOTATE
    )
    owner_pw = "secret"
    user_pw = "pass123"

    doc.save(
        pdf_path,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw=owner_pw,
        user_pw=user_pw,
        permissions=perm,
    )
    doc.close()
    return pdf_path
