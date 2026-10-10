import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure virtual environment Scripts is in PATH for ffmpeg and whisper
scripts_dir = os.path.dirname(sys.executable)
if scripts_dir and scripts_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")

from app.config.settings import get_settings
from app.models.exceptions import IngestionError
from app.models.schemas import KeyframeInfo, VideoSegment
from app.utils.scene_detector import SceneDetector

logger = logging.getLogger(__name__)

# Matches sentence-ending punctuation for both English and Hindi (Devanagari "।")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

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
    title: Optional[str] = None
    subject: Optional[str] = None


class VideoProcessor:
    """Processes MP4 video files into high-accuracy, sentence-reconstructed timestamped audio transcripts
    and extracts 30-second interval keyframe screenshots linked to their specific timeframe.
    Implements core audio extraction and transcription algorithms from Multimodel Copilot, combined
    with 30-second keyframe snapshot associations.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.max_size_bytes = self.settings.MAX_VIDEO_SIZE_BYTES  # 2 GB
        self.max_duration_seconds = self.settings.MAX_VIDEO_DURATION_HOURS * 3600  # 4 Hours
        self.keyframe_interval = self.settings.KEYFRAME_INTERVAL  # Default: 30 seconds
        self.whisper_model_name = self.settings.WHISPER_MODEL_SIZE
        self.groq_model = self.settings.GROQ_WHISPER_MODEL
        self.scene_detector = SceneDetector(
            keyframe_interval=self.keyframe_interval,
            output_dir=self.settings.FRAME_DIR,
        )
        self._whisper_model = None

    def _get_whisper_model(self):
        if self._whisper_model is None and WHISPER_AVAILABLE:
            self._whisper_model = whisper.load_model(self.whisper_model_name)
        return self._whisper_model

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
                    return VideoMetadata(
                        duration=duration,
                        has_audio=has_audio,
                        size_bytes=file_size,
                        title=Path(file_path).stem,
                    )
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

        return VideoMetadata(
            duration=duration,
            has_audio=has_audio,
            size_bytes=file_size,
            title=Path(file_path).stem,
        )

    def _build_prompt(self, subject: Optional[str] = None) -> str:
        """Builds an educational transcription prompt hint if subject is provided."""
        if subject:
            return f"This is an educational lecture on {subject}. Use correct terminology and punctuation."
        return ""

    def _extract_audio(self, video_path: str, job_id: str) -> Optional[str]:
        """Extracts 16kHz mono MP3 audio using ffmpeg with loudnorm normalization.
        Adopts the exact audio extraction flags from Multimodel Copilot (audio_extractor.py):
        - 16000 Hz, mono
        - loudnorm filter (I=-16:TP=-1.5:LRA=11) to normalize volume and amplify quiet speech
        - libmp3lame 64k compression (~5x smaller than WAV to stay under Groq 25MB limit)
        """
        ffmpeg_bin = self._get_ffmpeg_bin()
        if not ffmpeg_bin:
            return None

        audio_output = os.path.join(self.settings.UPLOAD_DIR, f"{job_id}.mp3")
        cmd = [
            ffmpeg_bin,
            "-i", str(video_path),
            "-ar", "16000",
            "-ac", "1",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-codec:a", "libmp3lame",
            "-b:a", "64k",
            "-vn",
            "-y",
            str(audio_output),
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if os.path.exists(audio_output) and os.path.getsize(audio_output) > 0:
                return audio_output
            return None
        except Exception as e:
            logger.warning(f"ffmpeg extraction failed: {e}")
            return None

    def _parse_groq_response(self, result: dict) -> List[Dict[str, Any]]:
        """Extracts raw speech segments from Groq API response."""
        segments: List[Dict[str, Any]] = []
        raw_segments = result.get("segments", [])
        if raw_segments:
            for seg in raw_segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                segments.append({
                    "text": text,
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                })
        else:
            text = result.get("text", "").strip()
            if text:
                segments.append({"text": text, "start": 0.0, "end": 0.0})
        return segments

    def _merge_into_sentences(self, raw_segments: List[Dict[str, Any]]) -> List[VideoSegment]:
        """Whisper-family models decode audio in fixed internal chunks, often splitting sentences
        at chunk boundaries. This algorithm stitches segment text into a continuous stream,
        re-splits on real sentence boundaries (. ! ? and Devanagari ।), and assigns each reconstructed
        sentence an interpolated start and end timestamp based on character position.
        From Multimodel Copilot transcriber.py.
        """
        if not raw_segments:
            return []

        full_text = ""
        char_times: List[float] = []

        for seg in raw_segments:
            piece = seg["text"].strip()
            if not piece:
                continue
            if full_text:
                full_text += " "
                char_times.append(seg["start"])
            full_text += piece
            duration = max(seg["end"] - seg["start"], 0.01)
            for i in range(len(piece)):
                frac = i / max(len(piece) - 1, 1)
                char_times.append(seg["start"] + frac * duration)

        raw_sentences = _SENTENCE_END.split(full_text)

        merged: List[VideoSegment] = []
        cursor = 0
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            start_idx = full_text.find(sentence, cursor)
            if start_idx == -1:
                start_idx = cursor
            end_idx = start_idx + len(sentence) - 1
            cursor = end_idx + 1

            start_idx = min(start_idx, len(char_times) - 1)
            end_idx = min(end_idx, len(char_times) - 1)

            start_t = round(char_times[start_idx], 2)
            end_t = round(char_times[end_idx], 2)
            if end_t <= start_t:
                end_t = round(start_t + 1.0, 2)

            merged.append(
                VideoSegment(
                    text=sentence,
                    start_time=start_t,
                    end_time=end_t,
                    status="ok",
                )
            )

        return merged

    def _associate_keyframes_with_segments(
        self,
        segments: List[VideoSegment],
        keyframes: List[KeyframeInfo],
    ) -> None:
        """Associates each 30-second keyframe screenshot with exactly one corresponding
        transcript segment where that timestamp falls (or nearest).
        Transcript segments in between remain text-only with frame_reference=None.
        """
        if not segments or not keyframes:
            return

        # Initialize all segments with no frame reference
        for seg in segments:
            seg.frame_reference = None
            seg.visual_text = None

        used_segment_indices = set()

        for kf in keyframes:
            kf_ts = kf.timestamp
            matched_idx = None

            # 1. Look for an unused segment whose start_time <= kf_ts <= end_time
            candidates = [
                i for i, s in enumerate(segments)
                if s.start_time <= kf_ts <= s.end_time and i not in used_segment_indices
            ]
            if candidates:
                matched_idx = candidates[0]
            else:
                # 2. If no segment directly covers kf_ts (e.g. pause or silence),
                # pick the closest unused segment
                unused_indices = [i for i in range(len(segments)) if i not in used_segment_indices]
                if unused_indices:
                    matched_idx = min(unused_indices, key=lambda i: abs(segments[i].start_time - kf_ts))
                else:
                    # Fallback to closest overall if all segments already have a frame
                    matched_idx = min(range(len(segments)), key=lambda i: abs(segments[i].start_time - kf_ts))

            if matched_idx is not None:
                used_segment_indices.add(matched_idx)
                seg = segments[matched_idx]
                seg.frame_reference = kf.frame_path
                seg.visual_text = f"Keyframe captured at {int(kf_ts)}s"

    def _save_transcript(self, segments: List[VideoSegment], job_id: str) -> str:
        """Saves transcript as JSON in storage/transcripts/{job_id}.json."""
        out_dir = self.settings.TRANSCRIPT_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{job_id}.json")

        data = [
            {
                "text": s.text,
                "start": s.start_time,
                "end": s.end_time,
                "source": "audio",
                "frame_reference": s.frame_reference,
                "visual_text": s.visual_text,
            }
            for s in segments
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return out_path

    def _transcribe_with_groq(
        self,
        audio_path: str,
        api_key: str,
        job_id: str,
        subject: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[VideoSegment]:
        """Calls Groq's high-speed Whisper cloud API with sentence reconstruction."""
        import requests

        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if size_mb > 25:
            raise ValueError(f"Audio file is {size_mb:.1f}MB, exceeding Groq's 25MB limit.")

        headers = {"Authorization": f"Bearer {api_key}"}
        data = {
            "model": self.groq_model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
            "temperature": 0.2,
        }
        prompt = self._build_prompt(subject)
        if prompt:
            data["prompt"] = prompt
        if language:
            data["language"] = language

        filename = os.path.basename(audio_path)
        mime_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"

        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, mime_type)}
            response = requests.post(GROQ_URL, headers=headers, data=data, files=files, timeout=300)

        if response.status_code != 200:
            raise RuntimeError(f"Groq API returned status {response.status_code}: {response.text}")

        res_json = response.json()
        raw_segments = self._parse_groq_response(res_json)
        segments = self._merge_into_sentences(raw_segments)
        return segments

    def _transcribe_with_local_whisper(self, audio_path: str, job_id: str) -> List[VideoSegment]:
        """Fallback transcription using local Whisper model when Groq key is absent or offline."""
        model = self._get_whisper_model()
        if model is None:
            return [
                VideoSegment(
                    text="[Audio Transcription Failed: Whisper model unavailable]",
                    start_time=0.0,
                    end_time=0.0,
                    status="transcription_failed",
                )
            ]

        try:
            result = model.transcribe(audio_path, fp16=False, verbose=False)
            raw = [
                {
                    "text": r.get("text", "").strip(),
                    "start": float(r.get("start", 0.0)),
                    "end": float(r.get("end", 0.0)),
                }
                for r in result.get("segments", [])
                if r.get("text", "").strip()
            ]
            segments = self._merge_into_sentences(raw)
            if not segments and result.get("text"):
                segments = [
                    VideoSegment(
                        text=result["text"].strip(),
                        start_time=0.0,
                        end_time=0.0,
                        status="ok",
                    )
                ]
            return segments
        except Exception as e:
            return [
                VideoSegment(
                    text=f"[Audio Transcription Failed: {str(e)}]",
                    start_time=0.0,
                    end_time=0.0,
                    status="transcription_failed",
                )
            ]

    def _transcribe_audio(
        self,
        audio_path: str,
        job_id: str,
        subject: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[VideoSegment]:
        """Transcribes audio using Groq Whisper API (primary) with local Whisper fallback."""
        groq_api_key = self.settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            try:
                segments = self._transcribe_with_groq(
                    audio_path=audio_path,
                    api_key=groq_api_key,
                    job_id=job_id,
                    subject=subject,
                    language=language,
                )
                if segments:
                    return segments
            except Exception as e:
                logger.warning(f"Groq transcription warning: {e}. Falling back to local Whisper...")

        return self._transcribe_with_local_whisper(audio_path, job_id)

    def process(
        self,
        file_path: str,
        job_id: str,
        subject: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Tuple[List[VideoSegment], Dict[str, Any], List[KeyframeInfo]]:
        """Main processing entry point.
        Performs 30-second keyframe snapshot extraction, audio extraction, and sentence-level transcription.
        Associates each 30s keyframe screenshot with the specific timeframe segment where it occurs.
        """
        if not os.path.exists(file_path):
            raise IngestionError(f"Video file not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > self.max_size_bytes:
            raise IngestionError(f"Video file size {file_size} exceeds maximum limit {self.max_size_bytes} bytes (2GB)")

        metadata = self._probe_video(file_path)
        if metadata.duration > self.max_duration_seconds:
            raise IngestionError(f"Video duration {metadata.duration}s exceeds maximum limit {self.max_duration_seconds}s (4h)")

        # 1. Extract keyframes at 30-second intervals (0s, 30s, 60s, ...)
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
            "title": metadata.title,
        }

        # Handle Audio Absent videos: each 30s keyframe is a segment
        if not metadata.has_audio:
            extra_meta["audio_absent"] = True
            for kf in keyframes:
                segments.append(
                    VideoSegment(
                        text=f"[Video Keyframe at {int(kf.timestamp)}s]",
                        start_time=kf.timestamp,
                        end_time=kf.timestamp + self.keyframe_interval,
                        frame_reference=kf.frame_path,
                        visual_text=f"Keyframe captured at {int(kf.timestamp)}s",
                        status="audio_absent",
                    )
                )
            if not segments:
                segments.append(
                    VideoSegment(
                        text="[Audio absent in video: visual lecture only]",
                        start_time=0.0,
                        end_time=metadata.duration,
                        status="audio_absent",
                    )
                )
            self._save_transcript(segments, job_id)
            return segments, extra_meta, keyframes

        # Extract normalized MP3 audio
        audio_path = self._extract_audio(file_path, job_id)
        if not audio_path or not os.path.exists(audio_path):
            segments.append(
                VideoSegment(
                    text="[Audio Extraction Failed: unable to demux audio stream]",
                    start_time=0.0,
                    end_time=0.0,
                    status="transcription_failed",
                )
            )
            self._save_transcript(segments, job_id)
            return segments, extra_meta, keyframes

        try:
            # Transcribe with sentence boundary stitching
            segments = self._transcribe_audio(
                audio_path=audio_path,
                job_id=job_id,
                subject=subject,
                language=language,
            )

            # Associate each 30-second keyframe to its exact corresponding sentence segment
            self._associate_keyframes_with_segments(segments, keyframes)
            self._save_transcript(segments, job_id)

        finally:
            # Clean up extracted temp audio file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception:
                pass

        return segments, extra_meta, keyframes
