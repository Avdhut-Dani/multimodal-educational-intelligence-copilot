from unittest.mock import MagicMock, patch
import pytest
from app.models.schemas import KeyframeInfo
from app.processors.video_processor import VideoMetadata, VideoProcessor


def test_property_5_keyframe_interval_count():
    """Property 5: Extracts floor(duration / interval) keyframes at k * interval timestamps."""
    processor = VideoProcessor()
    duration = 95.0  # 95 / 30 = 3 keyframes (at 0s, 30s, 60s)

    mock_meta = VideoMetadata(duration=duration, has_audio=False, size_bytes=1000)

    with patch.object(processor, "_probe_video", return_value=mock_meta), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1000):

        segments, meta, keyframes = processor.process("dummy_lecture.mp4", job_id="job-v1")

        assert len(keyframes) == 3
        for k, kf in enumerate(keyframes):
            expected_ts = k * 30
            assert abs(kf.timestamp - expected_ts) <= 1.0


def test_property_6_audio_absent_continuation():
    """Property 6: Audio-absent video proceeds with frame analysis and marks audio_absent=True."""
    processor = VideoProcessor()
    mock_meta = VideoMetadata(duration=60.0, has_audio=False, size_bytes=1000)

    with patch.object(processor, "_probe_video", return_value=mock_meta), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1000):

        segments, meta, keyframes = processor.process("silent_lecture.mp4", job_id="job-v2")

        assert meta["audio_absent"] is True
        assert len(segments) > 0  # Frame-based synthesized segments produced
        assert segments[0].status == "ok"


def test_property_7_transcription_failure_isolation():
    """Property 7: Audio transcription failure marks segment as transcription_failed and does not crash."""
    processor = VideoProcessor()
    mock_meta = VideoMetadata(duration=30.0, has_audio=True, size_bytes=1000)

    with patch.object(processor, "_probe_video", return_value=mock_meta), \
         patch.object(processor, "_extract_audio", return_value="/tmp/audio.wav"), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1000), \
         patch.object(processor, "_get_whisper_model") as mock_whisper:

        # Whisper transcribe raises an error
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("Unsupported audio codec")
        mock_whisper.return_value = mock_model

        segments, meta, keyframes = processor.process("corrupted_audio.mp4", job_id="job-v3")

        assert len(segments) >= 1
        assert segments[0].status == "transcription_failed"

