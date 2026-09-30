import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Ensure virtual environment Scripts is in PATH for ffmpeg and whisper
scripts_dir = os.path.dirname(sys.executable)
if scripts_dir and scripts_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")

from app.config.settings import get_settings
from app.models.exceptions import IngestionError
from app.models.schemas import KeyframeInfo, VideoSegment
from app.utils.scene_detector import SceneDetector

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


@dataclass
class VideoMetadata:
    duration: float
    has_audio: bool
    size_bytes: int


class VideoProcessor:
    """Processes MP4 video files into timestamped audio transcripts and 30-second interval keyframe snapshots.
    Complies strictly with Requirement 2 (Acceptance Criteria 1 through 7).
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.keyframe_interval = self.settings.KEYFRAME_INTERVAL  # Default: 30 seconds
        self.max_size_bytes = self.settings.MAX_VIDEO_SIZE_BYTES  # 2 GB (Requirement 2.7)
        self.max_duration_seconds = self.settings.MAX_VIDEO_DURATION_HOURS * 3600  # 4 Hours (Requirement 2.7)
        self.whisper_model_name = self.settings.WHISPER_MODEL_SIZE
        self.scene_detector = SceneDetector(
            keyframe_interval=self.keyframe_interval,
            output_dir=self.settings.FRAME_DIR,
        )
        self._whisper_model = None

    def _get_whisper_model(self):
        if self._whisper_model is None and WHISPER_AVAILABLE:
            self._whisper_model = whisper.load_model(self.whisper_model_name)
        return self._whisper_model

    def process(self, file_path: str, job_id: str) -> Tuple[List[VideoSegment], Dict[str, Any], List[KeyframeInfo]]:
        """Processes an MP4 file into segments, metadata dict, and keyframes."""
        if not os.path.exists(file_path):
            raise IngestionError(f"Video file not found: {file_path}")

        # Requirement 2.7: Support up to 2GB in size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_size_bytes:
            raise IngestionError(f"Video file size {file_size} exceeds maximum limit {self.max_size_bytes} bytes (2GB)")

        metadata = self._probe_video(file_path)
        # Requirement 2.7: Support up to 4 hours in duration
        if metadata.duration > self.max_duration_seconds:
            raise IngestionError(f"Video duration {metadata.duration}s exceeds maximum limit {self.max_duration_seconds}s (4h)")

        # Requirement 2.3 & 2.4: Extract keyframes at configurable interval (default: 30s)
        keyframes = self.scene_detector.extract_keyframes(
            video_path=file_path,
            duration=metadata.duration,
            job_id=job_id,
            dynamic_scene_detection=False,
        )

        segments: List[VideoSegment] = []
        extra_meta: Dict[str, Any] = {
            "duration": metadata.duration,
            "has_audio": metadata.has_audio,
            "audio_absent": not metadata.has_audio,
            "keyframe_count": len(keyframes),
        }

        # Requirement 2.1 & 2.2: Audio Extraction & Transcription
        if metadata.has_audio:
            audio_path = self._extract_audio(file_path, job_id)
            if audio_path and os.path.exists(audio_path):
                transcription_segments = self._transcribe_audio(audio_path)
                segments.extend(transcription_segments)
                # Clean up extracted wav file
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
            else:
                # Audio extraction failed
                segments.append(
                    VideoSegment(
                        text="[Audio Extraction Failed: unable to demux audio stream]",
                        start_time=0.0,
                        end_time=0.0,
                        status="transcription_failed",
                    )
                )
        else:
            # Requirement 2.5: audio_absent condition -> proceed with frames only
            extra_meta["audio_absent"] = True
            for kf in keyframes:
                segments.append(
                    VideoSegment(
                        text=f"[Video Keyframe at {int(kf.timestamp)}s]",
                        start_time=kf.timestamp,
                        end_time=kf.timestamp + self.keyframe_interval,
                        frame_reference=kf.frame_path,
                        visual_text=f"Keyframe captured at {int(kf.timestamp)}s",
                        status="ok",
                    )
                )

        # Requirement 2.4: Associate each keyframe with its corresponding timestamp
        for seg in segments:
            closest_kf = self._find_closest_keyframe(seg.start_time, keyframes)
            if closest_kf:
                seg.frame_reference = closest_kf.frame_path
                if not seg.visual_text:
                    seg.visual_text = f"Keyframe captured at {int(closest_kf.timestamp)}s"

        return segments, extra_meta, keyframes

    def _get_ffmpeg_bin(self) -> Optional[str]:
        """Locates ffmpeg executable from PATH or imageio-ffmpeg."""
        bin_path = shutil.which("ffmpeg")
        if not bin_path:
            try:
                import imageio_ffmpeg
                bin_path = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                pass
        return bin_path

    def _probe_video(self, file_path: str) -> VideoMetadata:
        """Probes video duration and audio track presence using ffmpeg or OpenCV."""
        duration = 0.0
        has_audio = False
        file_size = os.path.getsize(file_path)

        # 1. Try probing via ffmpeg
        ffmpeg_bin = self._get_ffmpeg_bin()
        if ffmpeg_bin:
            try:
                cmd = [ffmpeg_bin, "-i", file_path]
                res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                for line in res.stderr.splitlines():
                    if "Duration:" in line:
                        parts = line.split("Duration:")[1].split(",")[0].strip().split(":")
                        if len(parts) == 3:
                            duration = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                    if "Audio:" in line:
                        has_audio = True
                if duration > 0:
                    return VideoMetadata(duration=duration, has_audio=has_audio, size_bytes=file_size)
            except Exception:
                pass

        # 2. Fallback probe via OpenCV if available
        try:
            import cv2
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                if fps > 0 and frame_count > 0:
                    duration = frame_count / fps
                cap.release()
        except Exception:
            pass

        if duration <= 0:
            duration = 60.0

        return VideoMetadata(duration=duration, has_audio=has_audio, size_bytes=file_size)

    def _extract_audio(self, video_path: str, job_id: str) -> Optional[str]:
        """Extracts 16kHz mono audio track using ffmpeg."""
        ffmpeg_bin = self._get_ffmpeg_bin()
        if not ffmpeg_bin:
            return None

        audio_output = os.path.join(self.settings.UPLOAD_DIR, f"{job_id}_audio.wav")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_output,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return audio_output
        except Exception:
            return None

    def _transcribe_audio(self, audio_path: str) -> List[VideoSegment]:
        """Transcribes audio using Groq Whisper API (if key provided) or local Whisper, capturing timestamps."""
        segments: List[VideoSegment] = []

        # 1. Check for Groq API Key
        groq_api_key = self.settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            try:
                groq_segments = self._transcribe_with_groq(audio_path, groq_api_key)
                if groq_segments:
                    return groq_segments
            except Exception as e:
                print(f"[VideoProcessor] Groq transcription warning: {e}. Falling back to local Whisper...")

        # 2. Local Whisper model fallback
        model = self._get_whisper_model()
        if model is None:
            segments.append(
                VideoSegment(
                    text="[Audio Transcription Failed: Whisper model unavailable]",
                    start_time=0.0,
                    end_time=0.0,
                    status="transcription_failed",
                )
            )
            return segments

        try:
            # fp16=False ensures CPU compatibility on Windows
            result = model.transcribe(audio_path, fp16=False, verbose=False)
            raw_segments = result.get("segments", [])
            for r in raw_segments:
                text = r.get("text", "").strip()
                if not text:
                    continue
                segments.append(
                    VideoSegment(
                        text=text,
                        start_time=round(float(r.get("start", 0.0)), 2),
                        end_time=round(float(r.get("end", 0.0)), 2),
                        status="ok",
                    )
                )
        except Exception as e:
            # Requirement 2.6: Transcription failure isolation
            segments.append(
                VideoSegment(
                    text=f"[Audio Transcription Failed: {str(e)}]",
                    start_time=0.0,
                    end_time=0.0,
                    status="transcription_failed",
                )
            )

        return segments

    def _transcribe_with_groq(self, audio_path: str, api_key: str) -> List[VideoSegment]:
        """Calls Groq's high-speed Whisper cloud API for audio transcription."""
        import requests

        groq_url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {
            "model": self.settings.GROQ_WHISPER_MODEL,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
            "temperature": 0.2,
        }

        filename = os.path.basename(audio_path)
        mime_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"

        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, mime_type)}
            response = requests.post(groq_url, headers=headers, data=data, files=files, timeout=300)

        if response.status_code != 200:
            raise RuntimeError(f"Groq API returned status {response.status_code}: {response.text}")

        res_json = response.json()
        segments: List[VideoSegment] = []
        raw_segments = res_json.get("segments", [])
        for r in raw_segments:
            text = r.get("text", "").strip()
            if not text:
                continue
            segments.append(
                VideoSegment(
                    text=text,
                    start_time=round(float(r.get("start", 0.0)), 2),
                    end_time=round(float(r.get("end", 0.0)), 2),
                    status="ok",
                )
            )

        if not segments and res_json.get("text"):
            segments.append(
                VideoSegment(
                    text=res_json["text"].strip(),
                    start_time=0.0,
                    end_time=0.0,
                    status="ok",
                )
            )

        return segments

    def _find_closest_keyframe(self, timestamp: float, keyframes: List[KeyframeInfo]) -> Optional[KeyframeInfo]:
        """Finds keyframe closest to given timestamp."""
        if not keyframes:
            return None
        return min(keyframes, key=lambda k: abs(k.timestamp - timestamp))
