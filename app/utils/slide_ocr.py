import os
from typing import Optional

try:
    from rapidocr_onnxruntime import RapidOCR
    _ocr_engine = None

    def get_ocr_engine():
        global _ocr_engine
        if _ocr_engine is None:
            _ocr_engine = RapidOCR()
        return _ocr_engine

    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False


class SlideOCR:
    """Novelty 1: Extracts text and visual diagram annotations from lecture keyframe slides."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and RAPIDOCR_AVAILABLE

    def extract_text_from_frame(self, frame_path: str) -> Optional[str]:
        """Extracts text lines from a saved keyframe image."""
        if not self.enabled or not os.path.exists(frame_path):
            return None

        try:
            ocr = get_ocr_engine()
            result, elapse_list = ocr(frame_path)
            if not result:
                return None

            # result format: [[box, text, score], ...]
            lines = [item[1].strip() for item in result if len(item) > 1 and item[1].strip()]
            if not lines:
                return None

            return " ".join(lines)
        except Exception:
            return None

