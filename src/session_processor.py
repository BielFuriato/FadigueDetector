"""Processamento reutilizavel de uma sessao de fadiga."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from .alerts import AlertManager
from .face_landmarks import FaceLandmarkDetector
from .metrics import (
    calculate_eye_aspect_ratio,
    calculate_mouth_aspect_ratio,
    calculate_pupil_position,
    estimate_head_pose_2d,
)
from .risk_model import RiskModel, RiskResult
from .temporal_buffer import TemporalBuffer
from .visualization import draw_landmarks


def empty_metrics() -> dict[str, Any]:
    """Metricas neutras usadas quando nenhuma face e detectada."""
    return {
        "ear_left": 0.0,
        "ear_right": 0.0,
        "ear_avg": 0.0,
        "mar": 0.0,
        "eye_closed": False,
        "mouth_open": False,
        "pupil_x": None,
        "pupil_y": None,
        "head_tilt_score": 0.0,
        "looking_down": False,
        "looking_side": False,
        "face_frontal": False,
    }


class FatigueSessionProcessor:
    """Processa frames mantendo estado temporal isolado por sessao."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        sound_enabled: bool = False,
        draw_landmarks_enabled: bool = True,
    ) -> None:
        self.config = config
        self.sound_enabled = sound_enabled
        self.draw_landmarks_enabled = draw_landmarks_enabled
        self.thresholds = config.get("thresholds", {})
        self.temporal_config = config.get("temporal", {})
        self.alert_config = config.get("alerts", {})

        self.detector: FaceLandmarkDetector | None = None
        self.buffer: TemporalBuffer | None = None
        self.risk_model: RiskModel | None = None
        self.alerts: AlertManager | None = None
        self.reset()

    def reset(self) -> None:
        """Limpa o estado temporal para iniciar uma nova sessao."""
        if self.detector is None:
            self.detector = FaceLandmarkDetector()

        self.buffer = TemporalBuffer(
            window_sec=float(
                self.temporal_config.get(
                    "risk_window_sec",
                    self.temporal_config.get("perclos_window_sec", 60),
                )
            ),
            yawn_window_sec=float(self.temporal_config.get("yawn_window_sec", 120)),
            long_eye_closure_sec=float(self.thresholds.get("long_eye_closure_sec", 2.0)),
            yawn_min_duration_sec=float(self.thresholds.get("yawn_min_duration_sec", 1.5)),
            face_missing_sec=float(self.thresholds.get("face_missing_sec", 3.0)),
        )
        self.risk_model = RiskModel(self.thresholds, self.config.get("scores", {}))
        self.alerts = AlertManager(
            sound_enabled=self.sound_enabled and bool(self.alert_config.get("sound_enabled", True)),
            cooldown_sec=float(self.alert_config.get("cooldown_sec", 5)),
        )
        self.frame_id = 0
        self.fps = 0.0
        self.previous_perf_time: float | None = None
        self.previous_total_yawn_count = 0
        self.last_result: dict[str, Any] | None = None

    def close(self) -> None:
        """Fecha recursos internos do MediaPipe."""
        if self.detector is not None:
            self.detector.close()
            self.detector = None

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float | None = None,
        *,
        input_color: str = "bgr",
        fps: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Processa um frame e retorna imagem anotada e metricas atuais.

        `timestamp` deve vir da linha do tempo do video para arquivos e do
        relogio real para webcam. O frame de saida preserva o formato de cor
        informado em `input_color`.
        """
        if self.detector is None or self.buffer is None or self.risk_model is None or self.alerts is None:
            self.reset()

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Frame vazio ou invalido.")

        timestamp_value = float(time.time() if timestamp is None else timestamp)
        self.frame_id += 1
        display_fps = self._update_fps(fps)

        color_format = input_color.lower()
        if color_format == "rgb":
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif color_format == "bgr":
            frame_bgr = frame.copy()
        else:
            raise ValueError("input_color deve ser 'bgr' ou 'rgb'.")

        face = self.detector.detect(frame_bgr)
        metrics = empty_metrics()
        landmarks_px = None

        if face is not None:
            landmarks_px = face.pixels
            ear = calculate_eye_aspect_ratio(landmarks_px)
            mar = calculate_mouth_aspect_ratio(landmarks_px)
            pupil = calculate_pupil_position(landmarks_px)
            head = estimate_head_pose_2d(landmarks_px)
            metrics.update(
                {
                    "ear_left": ear.left,
                    "ear_right": ear.right,
                    "ear_avg": ear.average,
                    "mar": mar,
                    "eye_closed": ear.average < float(self.thresholds.get("ear_closed", 0.21)),
                    "mouth_open": mar > float(self.thresholds.get("mar_open", 0.65)),
                    "pupil_x": pupil.x if pupil is not None else None,
                    "pupil_y": pupil.y if pupil is not None else None,
                    "head_tilt_score": head.head_tilt_score,
                    "looking_down": head.looking_down,
                    "looking_side": head.looking_side,
                    "face_frontal": head.face_frontal,
                }
            )

        record = {
            "timestamp": timestamp_value,
            "ear_avg": metrics["ear_avg"],
            "mar": metrics["mar"],
            "eye_closed": metrics["eye_closed"],
            "mouth_open": metrics["mouth_open"],
            "pupil_x": metrics["pupil_x"],
            "pupil_y": metrics["pupil_y"],
            "head_tilt_score": metrics["head_tilt_score"],
            "looking_down": metrics["looking_down"],
            "looking_side": metrics["looking_side"],
            "face_frontal": metrics["face_frontal"],
            "face_detected": face is not None,
        }
        temporal = self.buffer.add(record)
        yawn_detected_recently = temporal["total_yawn_count"] > self.previous_total_yawn_count
        self.previous_total_yawn_count = temporal["total_yawn_count"]

        risk = self.risk_model.evaluate(temporal, metrics["head_tilt_score"])
        alert = self.alerts.evaluate(
            risk,
            temporal,
            {
                **metrics,
                "long_eye_closure_sec": float(self.thresholds.get("long_eye_closure_sec", 2.0)),
                "yawn_detected_recently": yawn_detected_recently,
            },
        )

        annotated_bgr = frame_bgr.copy()
        if self.draw_landmarks_enabled:
            draw_landmarks(annotated_bgr, landmarks_px)

        result = self._build_result(
            timestamp=timestamp_value,
            metrics=metrics,
            temporal=temporal,
            risk=risk,
            alert_message=alert.message,
            alert_level=alert.level,
            face_detected=face is not None,
            fps=display_fps,
        )
        self.last_result = result

        if color_format == "rgb":
            return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), result
        return annotated_bgr, result

    def _update_fps(self, external_fps: float | None) -> float:
        if external_fps is not None and external_fps > 0:
            self.fps = float(external_fps)
            return self.fps

        now = time.perf_counter()
        if self.previous_perf_time is None:
            self.previous_perf_time = now
            return self.fps

        instant_fps = 1.0 / max(now - self.previous_perf_time, 1e-6)
        self.fps = instant_fps if self.fps == 0.0 else (0.90 * self.fps + 0.10 * instant_fps)
        self.previous_perf_time = now
        return self.fps

    def _build_result(
        self,
        *,
        timestamp: float,
        metrics: dict[str, Any],
        temporal: dict[str, Any],
        risk: RiskResult,
        alert_message: str,
        alert_level: str,
        face_detected: bool,
        fps: float,
    ) -> dict[str, Any]:
        log_row = {
            "timestamp": timestamp,
            "frame_id": self.frame_id,
            "face_detected": face_detected,
            "ear_left": metrics["ear_left"],
            "ear_right": metrics["ear_right"],
            "ear_avg": metrics["ear_avg"],
            "mar": metrics["mar"],
            "eye_closed": metrics["eye_closed"],
            "mouth_open": metrics["mouth_open"],
            "perclos": temporal["perclos"],
            "yawn_count_window": temporal["yawn_count_window"],
            "long_eye_closure_count": temporal["long_eye_closure_count"],
            "avg_blink_duration": temporal["avg_blink_duration"],
            "avg_pupil_movement": temporal["avg_pupil_movement"],
            "pupil_movement_samples": temporal["pupil_movement_samples"],
            "head_tilt_score": metrics["head_tilt_score"],
            "looking_down": metrics["looking_down"],
            "looking_side": metrics["looking_side"],
            "fatigue_score": risk.fatigue_score,
            "final_state": risk.final_state,
            "alert_message": alert_message,
            "fps": fps,
        }
        return {
            "timestamp": timestamp,
            "frame_id": self.frame_id,
            "face_detected": face_detected,
            "metrics": metrics,
            "temporal": temporal,
            "risk": {
                "fatigue_score": risk.fatigue_score,
                "fatigue_level": risk.fatigue_level,
                "final_state": risk.final_state,
            },
            "alert": {
                "message": alert_message,
                "level": alert_level,
            },
            "fps": fps,
            "log_row": log_row,
        }
