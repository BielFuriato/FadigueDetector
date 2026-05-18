"""Deteccao de landmarks faciais com MediaPipe Face Mesh."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import cv2
import numpy as np


DEFAULT_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_TASK_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "models" / "face_landmarker.task"


@dataclass
class FaceLandmarks:
    """Landmarks de uma face em coordenadas normalizadas e de pixel."""

    normalized: np.ndarray
    pixels: np.ndarray
    frame_width: int
    frame_height: int


class FaceLandmarkDetector:
    """Detector baseado no Face Mesh classico ou no MediaPipe Tasks."""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp

        self.mp = mp
        self.backend = "solutions" if hasattr(mp, "solutions") else "tasks"
        if self.backend == "solutions":
            self.mp_face_mesh = mp.solutions.face_mesh
            self.detector = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            model_path = self._ensure_task_model()
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> FaceLandmarks | None:
        """Detecta a primeira face e retorna seus landmarks."""
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "solutions":
            return self._detect_with_solutions(frame_rgb, width, height)
        return self._detect_with_tasks(frame_rgb, width, height)

    def _detect_with_solutions(self, frame_rgb: np.ndarray, width: int, height: int) -> FaceLandmarks | None:
        results = self.detector.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        return self._to_face_landmarks(landmarks, width, height)

    def _detect_with_tasks(self, frame_rgb: np.ndarray, width: int, height: int) -> FaceLandmarks | None:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        results = self.detector.detect(image)
        if not results.face_landmarks:
            return None

        landmarks = results.face_landmarks[0]
        return self._to_face_landmarks(landmarks, width, height)

    def _to_face_landmarks(self, landmarks, width: int, height: int) -> FaceLandmarks:
        normalized = np.array([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)
        pixels = np.column_stack(
            (
                np.clip(normalized[:, 0] * width, 0, width - 1),
                np.clip(normalized[:, 1] * height, 0, height - 1),
                normalized[:, 2] * width,
            )
        ).astype(np.float32)

        return FaceLandmarks(
            normalized=normalized,
            pixels=pixels,
            frame_width=width,
            frame_height=height,
        )

    def _ensure_task_model(self) -> Path:
        model_path = Path(os.getenv("MEDIAPIPE_FACE_LANDMARKER_MODEL", DEFAULT_TASK_MODEL_PATH))
        if model_path.exists():
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urlretrieve(DEFAULT_TASK_MODEL_URL, model_path)
        except (OSError, URLError) as exc:
            raise RuntimeError(
                "MediaPipe Tasks precisa do modelo face_landmarker.task. "
                f"Baixe de {DEFAULT_TASK_MODEL_URL} e salve em {model_path}, "
                "ou defina MEDIAPIPE_FACE_LANDMARKER_MODEL para o caminho do arquivo."
            ) from exc
        return model_path

    def close(self) -> None:
        """Fecha recursos internos do MediaPipe."""
        self.detector.close()
