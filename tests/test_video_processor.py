from unittest.mock import MagicMock, patch
import pytest
from app.models.schemas import KeyframeInfo, VideoSegment
from app.processors.video_processor import VideoMetadata, VideoProcessor


def test_property_5_keyframe_interval_count():
    """Property 5: Keyframes are extracted at 30-second intervals (0s, 30s, 60s)."""
    processor = VideoProcessor()
    duration = 95.0  # 95 / 30 = 3 keyframes (at 0s, 30s, 60s)
    mock_meta = VideoMetadata(duration=duration, has_audio=False, size_bytes=1000)

    with patch.object(processor, "_probe_video", return_value=mock_meta), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1000):

        segments, meta, keyframes = processor.process("dummy_lecture.mp4", job_id="job-v1")

        assert len(keyframes) == 3
        assert meta["keyframe_count"] == 3
        assert abs(keyframes[0].timestamp - 0.0) <= 1.0
        assert abs(keyframes[1].timestamp - 30.0) <= 1.0
        assert abs(keyframes[2].timestamp - 60.0) <= 1.0


def test_property_6_audio_absent_continuation():
    """Property 6: Audio-absent video marks audio_absent=True and creates keyframe segments."""
    processor = VideoProcessor()
    mock_meta = VideoMetadata(duration=60.0, has_audio=False, size_bytes=1000)

    with patch.object(processor, "_probe_video", return_value=mock_meta), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1000):

        segments, meta, keyframes = processor.process("silent_lecture.mp4", job_id="job-v2")

        assert meta["audio_absent"] is True
        assert len(segments) > 0
        assert segments[0].status == "audio_absent"
        assert segments[0].frame_reference is not None


def test_property_7_transcription_failure_isolation():
    """Property 7: Audio transcription failure marks segment as transcription_failed and does not crash."""
    processor = VideoProcessor()
    mock_meta = VideoMetadata(duration=30.0, has_audio=True, size_bytes=1000)

    with patch.object(processor, "_probe_video", return_value=mock_meta), \
         patch.object(processor, "_extract_audio", return_value="dummy_job.mp3"), \
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


def test_sentence_merging_algorithm():
    """Verifies that _merge_into_sentences stitches sentences and interpolates timestamps."""
    processor = VideoProcessor()
    raw = [
        {"text": "Hello students. Today we", "start": 0.0, "end": 2.0},
        {"text": "will learn Python.", "start": 2.0, "end": 4.0},
    ]
    merged = processor._merge_into_sentences(raw)
    assert len(merged) == 2
    assert merged[0].text == "Hello students."
    assert merged[1].text == "Today we will learn Python."
    assert merged[0].start_time == 0.0
    assert merged[1].end_time >= 3.5


def test_keyframe_segment_association():
    """Verifies that only segments covering the 30s timestamps receive frame references,
    while intermediate segments remain text-only with frame_reference=None.
    """
    processor = VideoProcessor()
    segments = [
        VideoSegment(text="First sentence.", start_time=0.0, end_time=9.5, status="ok"),
        VideoSegment(text="Middle sentence A.", start_time=10.0, end_time=19.5, status="ok"),
        VideoSegment(text="Middle sentence B.", start_time=20.0, end_time=28.0, status="ok"),
        VideoSegment(text="Thirtieth second sentence.", start_time=28.5, end_time=36.0, status="ok"),
    ]
    keyframes = [
        KeyframeInfo(timestamp=0.0, frame_path="frames/job_frame_0s.jpg"),
        KeyframeInfo(timestamp=30.0, frame_path="frames/job_frame_30s.jpg"),
    ]

    processor._associate_keyframes_with_segments(segments, keyframes)

    # Segment 0 spans 0s -> should have frame
    assert segments[0].frame_reference == "frames/job_frame_0s.jpg"
    assert segments[0].visual_text == "Keyframe captured at 0s"

    # Segment 1 & 2 do NOT span any 30s milestone -> frame_reference must be None
    assert segments[1].frame_reference is None
    assert segments[1].visual_text is None
    assert segments[2].frame_reference is None
    assert segments[2].visual_text is None

    # Segment 3 spans 30s -> should have frame
    assert segments[3].frame_reference == "frames/job_frame_30s.jpg"
    assert segments[3].visual_text == "Keyframe captured at 30s"
