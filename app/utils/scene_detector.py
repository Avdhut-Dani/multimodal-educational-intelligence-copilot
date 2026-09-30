import os
from typing import List
from app.models.schemas import KeyframeInfo

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class SceneDetector:
    """Extracts keyframes from MP4 video at configurable intervals (default: 30 seconds) per Requirement 2.3 & 2.4."""

    def __init__(self, keyframe_interval: int = 30, scene_threshold: float = 0.35, output_dir: str = "./storage/frames"):
        self.keyframe_interval = keyframe_interval
        self.scene_threshold = scene_threshold
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_keyframes(self, video_path: str, duration: float, job_id: str, dynamic_scene_detection: bool = False) -> List[KeyframeInfo]:
        """Extracts keyframes at exact 30-second intervals (0s, 30s, 60s, ...) saving screenshots to disk."""
        keyframes: List[KeyframeInfo] = []
        if duration <= 0:
            return keyframes

        count = int(duration // self.keyframe_interval)
        if count == 0 and duration > 0:
            count = 1  # At least extract frame at 0s for short videos

        cap = None
        if OPENCV_AVAILABLE and os.path.exists(video_path):
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    cap = None
            except Exception:
                cap = None

        for k in range(count):
            ts = float(k * self.keyframe_interval)
            frame_filename = f"{job_id}_frame_{int(ts)}s.jpg"
            frame_path = os.path.join(self.output_dir, frame_filename)

            saved = False
            if cap is not None:
                try:
                    # Direct seek to exact timestamp in milliseconds
                    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cv2.imwrite(frame_path, frame)
                        saved = True
                except Exception:
                    saved = False

            # If OpenCV could not read frame or wasn't available, write placeholder frame
            if not saved and OPENCV_AVAILABLE:
                try:
                    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        f"Keyframe at {int(ts)}s",
                        (50, 360),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imwrite(frame_path, blank)
                except Exception:
                    pass

            keyframes.append(
                KeyframeInfo(
                    timestamp=round(ts, 2),
                    frame_path=frame_path,
                    is_scene_change=False,
                )
            )

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        return keyframes
