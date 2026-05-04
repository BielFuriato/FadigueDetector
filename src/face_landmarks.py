"""Deteccao de landmarks faciais com MediaPipe Face Mesh."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceLandmarks:
    """Landmarks de uma face em coordenadas normalizadas e de pixel."""

    normalized: np.ndarray
    pixels: np.ndarray
    frame_width: int
    frame_height: int


class FaceLandmarkDetector:
    """Detector baseado em mediapipe.solutions.face_mesh."""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> FaceLandmarks | None:
        """Detecta a primeira face e retorna seus landmarks."""
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
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

    def close(self) -> None:
        """Fecha recursos internos do MediaPipe."""
        self.detector.close()
