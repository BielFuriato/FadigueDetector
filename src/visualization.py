"""Desenho da interface visual em OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from .metrics import DRAW_LANDMARKS


COLORS = {
    "NORMAL": (46, 204, 113),
    "ATENCAO": (0, 215, 255),
    "RISCO_MODERADO": (0, 140, 255),
    "RISCO_ALTO": (0, 0, 255),
}


def color_for_state(state: str) -> tuple[int, int, int]:
    normalized = state.upper()
    if "CRITICO" in normalized or "ALTO" in normalized:
        return COLORS["RISCO_ALTO"]
    if "MODERADO" in normalized:
        return COLORS["RISCO_MODERADO"]
    if "ATEN" in normalized:
        return COLORS["ATENCAO"]
    if "RISCO" in normalized:
        return COLORS["RISCO_ALTO"]
    return COLORS["NORMAL"]


def draw_landmarks(frame: np.ndarray, landmarks_px: np.ndarray | None) -> None:
    """Desenha apenas pontos principais para manter a tela limpa."""
    if landmarks_px is None:
        return
    for index in DRAW_LANDMARKS:
        x, y = landmarks_px[index][:2].astype(int)
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)


def draw_status_panel(
    frame: np.ndarray,
    metrics: dict,
    temporal: dict,
    risk,
    alert,
    fps: float,
) -> np.ndarray:
    """Desenha painel com metricas e estado final."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (430, 343), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    state_color = color_for_state(risk.final_state)
    lines = [
        ("EAR", f"{metrics.get('ear_avg', 0.0):.3f}"),
        ("MAR", f"{metrics.get('mar', 0.0):.3f}"),
        ("PERCLOS", f"{temporal.get('perclos', 0.0) * 100:.1f}%"),
        ("Bocejos janela", str(temporal.get("yawn_count_window", 0))),
        ("Piscada media", f"{temporal.get('avg_blink_duration', 0.0):.2f}s"),
        ("Mov. pupila", f"{temporal.get('avg_pupil_movement', 0.0):.3f}"),
        ("Fadiga", f"{risk.fatigue_score:.0f} ({risk.fatigue_level})"),
        ("Distracao", f"{risk.distraction_score:.0f} ({risk.distraction_level})"),
        ("Estado", risk.final_state),
        ("FPS", f"{fps:.1f}"),
    ]

    y = 42
    for label, value in lines:
        color = state_color if label == "Estado" else (230, 230, 230)
        cv2.putText(frame, f"{label}: {value}", (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        y += 29

    if alert.message:
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, height - 58), (width, height), state_color, -1)
        cv2.putText(
            frame,
            alert.message,
            (24, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.86,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame
