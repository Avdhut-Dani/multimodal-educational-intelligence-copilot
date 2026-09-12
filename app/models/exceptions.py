class IngestionError(Exception):
    """Base exception for all ingestion subsystem errors."""
    pass


class PDFUnreadableError(IngestionError):
    """Raised when a PDF cannot be opened, is corrupted, or is password-protected."""
    def __init__(self, message: str = "PDF is unreadable, corrupted, or password-protected"):
        super().__init__(message)
        self.message = message


class TranscriptUnavailableError(IngestionError):
    """Raised when a YouTube video has no captions or transcripts available."""
    def __init__(self, message: str = "No transcript is available for this video"):
        super().__init__(message)
        self.message = message


class VideoInaccessibleError(IngestionError):
    """Raised when a YouTube URL is invalid, private, or removed."""
    def __init__(self, message: str = "Video is inaccessible, private, or invalid URL"):
        super().__init__(message)
        self.message = message


class TranscriptionError(IngestionError):
    """Raised when audio transcription fails."""
    def __init__(self, message: str = "Audio transcription failed"):
        super().__init__(message)
        self.message = message


class PublishingError(IngestionError):
    """Raised when Vector DB publishing exhausts all retry attempts."""
    def __init__(self, message: str = "Vector DB publication failed after maximum retries"):
        super().__init__(message)
        self.message = message

