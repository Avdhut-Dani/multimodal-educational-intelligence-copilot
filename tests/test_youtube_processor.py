from unittest.mock import MagicMock, patch
import pytest
from app.models.exceptions import TranscriptUnavailableError, VideoInaccessibleError
from app.processors.youtube_processor import YouTubeProcessor


def test_property_10_invalid_url_error():
    """Property 10: Invalid or unparseable YouTube URL raises VideoInaccessibleError."""
    processor = YouTubeProcessor()
    with pytest.raises(VideoInaccessibleError):
        processor.process("not-a-valid-url")


def test_property_8_and_9_transcript_and_metadata_preservation():
    """Properties 8 & 9: Transcripts preserve start and duration; metadata contains title, channel, duration."""
    processor = YouTubeProcessor()

    mock_entries = [
        {"text": "Welcome to deep learning lecture.", "start": 0.0, "duration": 4.5},
        {"text": "Today we cover activation functions.", "start": 4.5, "duration": 5.0},
    ]

    mock_transcript = MagicMock()
    mock_transcript.fetch.return_value = mock_entries

    mock_list = MagicMock()
    mock_list.find_transcript.return_value = mock_transcript

    mock_meta = {
        "title": "Deep Learning Chapter 1",
        "channel": "MIT OpenCourseWare",
        "duration": 600.0,
        "publication_date": "20260115",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    }

    with patch("app.processors.youtube_processor.YouTubeTranscriptApi.list_transcripts", return_value=mock_list), \
         patch.object(processor, "_fetch_metadata", return_value=mock_meta):

        segments, meta = processor.process("https://www.youtube.com/watch?v=dQw4w9WgXcQ", language="en")

        assert len(segments) == 2
        assert segments[0].start_time == 0.0
        assert segments[0].duration == 4.5
        assert segments[1].start_time == 4.5
        assert segments[1].duration == 5.0

        assert meta["title"] == "Deep Learning Chapter 1"
        assert meta["channel"] == "MIT OpenCourseWare"
        assert meta["duration"] == 600.0


def test_property_10_transcripts_disabled_error():
    """Property 10: Video without captions raises TranscriptUnavailableError."""
    processor = YouTubeProcessor()
    with patch("app.processors.youtube_processor.YouTubeTranscriptApi.list_transcripts") as mock_list, \
         patch.object(processor, "_fetch_metadata", return_value={"title": "Test", "channel": "C", "duration": 10}):

        from youtube_transcript_api import TranscriptsDisabled
        mock_list.side_effect = TranscriptsDisabled("Video captions disabled")

        with pytest.raises(TranscriptUnavailableError):
            processor.process("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
