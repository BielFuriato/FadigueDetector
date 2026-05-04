"""Captura de video por webcam ou arquivo."""

from __future__ import annotations

from typing import Any

import cv2

from .utils import parse_camera_source


class CameraStream:
    """Wrapper simples para cv2.VideoCapture com tratamento de erros."""

    def __init__(
        self,
        source: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> None:
        self.source = parse_camera_source(source)
        self.width = width
        self.height = height
        self.fps = fps
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Abre a camera ou arquivo de video."""
        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {self.source}")

        if isinstance(self.source, int):
            if self.width:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.fps:
                self.capture.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> tuple[bool, Any]:
        """Le um frame."""
        if self.capture is None:
            raise RuntimeError("CameraStream.read() chamado antes de open().")
        return self.capture.read()

    def get_fps(self) -> float:
        """Retorna FPS nominal quando disponivel."""
        if self.capture is None:
            return 0.0
        fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if fps > 1 else 0.0

    def release(self) -> None:
        """Libera o recurso de captura."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
