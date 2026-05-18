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
    """Compact HUD overlay — glanceable essentials only. Detailed metrics live in the side cards."""
    state_color = color_for_state(risk.final_state)

    # -- Compact corner HUD box (top-left) --
    lines = [
        ("Estado", risk.final_state),
        ("Fadiga", f"{risk.fatigue_score:.0f}"),
        ("PERCLOS", f"{temporal.get('perclos', 0.0) * 100:.1f}%"),
        ("FPS", f"{fps:.1f}"),
    ]

    box_x, box_y = 8, 8
    box_w, box_h = 186, 108
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.46
    line_spacing = 23
    text_x = box_x + 12
    text_y = box_y + 24

    for label, value in lines:
        color = state_color if label == "Estado" else (220, 220, 220)
        cv2.putText(frame, f"{label}: {value}", (text_x, text_y), font, font_scale, color, 1, cv2.LINE_AA)
        text_y += line_spacing

    # -- Slim alert strip (bottom, only when active) --
    if alert.message:
        h, w = frame.shape[:2]
        bar_h = 28
        cv2.rectangle(frame, (0, h - bar_h), (w, h), state_color, -1)
        cv2.putText(
            frame,
            alert.message,
            (14, h - 9),
            font,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame
