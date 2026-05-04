"""Metricas faciais: EAR, MAR e postura simplificada da cabeca."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Indices estaveis do MediaPipe Face Mesh usados para medidas geometricas 2D.
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
MOUTH_MAR = {
    "left": 61,
    "right": 291,
    "upper_inner": 13,
    "lower_inner": 14,
    "upper_left": 81,
    "lower_left": 178,
    "upper_right": 311,
    "lower_right": 402,
}
HEAD_POINTS = {
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "nose_tip": 1,
    "chin": 152,
    "face_left": 234,
    "face_right": 454,
}
DRAW_LANDMARKS = sorted(
    set(LEFT_EYE_EAR + RIGHT_EYE_EAR + LEFT_IRIS + RIGHT_IRIS + list(MOUTH_MAR.values()) + list(HEAD_POINTS.values()))
)


@dataclass
class EarResult:
    left: float
    right: float
    average: float


@dataclass
class HeadPoseResult:
    head_tilt_score: float
    looking_down: bool
    looking_side: bool
    face_frontal: bool
    eye_slope_deg: float
    nose_offset_ratio: float
    nose_vertical_ratio: float


@dataclass
class PupilPositionResult:
    x: float
    y: float


def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a[:2] - point_b[:2]))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-6 else 0.0


def calculate_eye_aspect_ratio(landmarks_px: np.ndarray) -> EarResult:
    """Calcula EAR para os dois olhos.

    EAR menor indica menor abertura ocular. Os limiares sao calibrados no YAML.
    """
    def ear(indices: list[int]) -> float:
        p1, p2, p3, p4, p5, p6 = [landmarks_px[index] for index in indices]
        vertical_1 = _distance(p2, p6)
        vertical_2 = _distance(p3, p5)
        horizontal = _distance(p1, p4)
        return _safe_ratio(vertical_1 + vertical_2, 2.0 * horizontal)

    left = ear(LEFT_EYE_EAR)
    right = ear(RIGHT_EYE_EAR)
    return EarResult(left=left, right=right, average=(left + right) / 2.0)


def calculate_mouth_aspect_ratio(landmarks_px: np.ndarray) -> float:
    """Calcula MAR usando abertura vertical e largura da boca."""
    left = landmarks_px[MOUTH_MAR["left"]]
    right = landmarks_px[MOUTH_MAR["right"]]
    upper_inner = landmarks_px[MOUTH_MAR["upper_inner"]]
    lower_inner = landmarks_px[MOUTH_MAR["lower_inner"]]
    upper_left = landmarks_px[MOUTH_MAR["upper_left"]]
    lower_left = landmarks_px[MOUTH_MAR["lower_left"]]
    upper_right = landmarks_px[MOUTH_MAR["upper_right"]]
    lower_right = landmarks_px[MOUTH_MAR["lower_right"]]

    vertical = (
        _distance(upper_inner, lower_inner)
        + _distance(upper_left, lower_left)
        + _distance(upper_right, lower_right)
    ) / 3.0
    horizontal = _distance(left, right)
    return _safe_ratio(vertical, horizontal)


def calculate_pupil_position(landmarks_px: np.ndarray) -> PupilPositionResult | None:
    """Estima posicao media das pupilas normalizada pela largura dos olhos."""
    if len(landmarks_px) <= max(LEFT_IRIS + RIGHT_IRIS):
        return None

    def eye_relative_position(iris_indices: list[int], left_corner: int, right_corner: int) -> np.ndarray:
        iris_center = np.mean([landmarks_px[index] for index in iris_indices], axis=0)
        corner_left = landmarks_px[left_corner]
        corner_right = landmarks_px[right_corner]
        eye_center = (corner_left + corner_right) / 2.0
        eye_width = max(_distance(corner_left, corner_right), 1.0)
        return (iris_center[:2] - eye_center[:2]) / eye_width

    left = eye_relative_position(LEFT_IRIS, 33, 133)
    right = eye_relative_position(RIGHT_IRIS, 362, 263)
    average = (left + right) / 2.0
    return PupilPositionResult(x=float(average[0]), y=float(average[1]))


def estimate_head_pose_2d(landmarks_px: np.ndarray) -> HeadPoseResult:
    """Estima postura da cabeca por geometria 2D simples.

    Esta aproximacao evita solvePnP para manter o prototipo facil de executar.
    Ela e sensivel a camera, iluminacao e posicao inicial do usuario.
    """
    left_eye = landmarks_px[HEAD_POINTS["left_eye_outer"]]
    right_eye = landmarks_px[HEAD_POINTS["right_eye_outer"]]
    nose = landmarks_px[HEAD_POINTS["nose_tip"]]
    chin = landmarks_px[HEAD_POINTS["chin"]]
    face_left = landmarks_px[HEAD_POINTS["face_left"]]
    face_right = landmarks_px[HEAD_POINTS["face_right"]]

    eye_center = (left_eye + right_eye) / 2.0
    face_center = (face_left + face_right) / 2.0
    face_width = max(_distance(face_left, face_right), 1.0)
    eye_chin_vertical = max(abs(chin[1] - eye_center[1]), 1.0)

    eye_delta = right_eye[:2] - left_eye[:2]
    eye_slope_deg = float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0])))
    nose_offset_ratio = float((nose[0] - face_center[0]) / face_width)
    nose_vertical_ratio = float((nose[1] - eye_center[1]) / eye_chin_vertical)

    tilt_component = min(abs(eye_slope_deg) / 25.0, 1.0)
    side_component = min(abs(nose_offset_ratio) / 0.22, 1.0)
    down_component = min(max(nose_vertical_ratio - 0.50, 0.0) / 0.25, 1.0)
    head_tilt_score = float(np.clip((0.45 * tilt_component + 0.35 * side_component + 0.20 * down_component) * 100, 0, 100))

    looking_side = abs(nose_offset_ratio) > 0.16
    looking_down = nose_vertical_ratio > 0.66
    face_frontal = not looking_side and not looking_down and abs(eye_slope_deg) < 12

    return HeadPoseResult(
        head_tilt_score=head_tilt_score,
        looking_down=looking_down,
        looking_side=looking_side,
        face_frontal=face_frontal,
        eye_slope_deg=eye_slope_deg,
        nose_offset_ratio=nose_offset_ratio,
        nose_vertical_ratio=nose_vertical_ratio,
    )
