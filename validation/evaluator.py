"""Funcoes de avaliacao dos resultados por video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_COLUMNS = [
    "video_id",
    "video_path",
    "dataset",
    "expected_class",
    "expected_event",
    "duration_sec",
    "valid_face_ratio",
    "mean_perclos",
    "max_perclos",
    "mean_fatigue_score",
    "max_fatigue_score",
    "mean_distraction_score",
    "max_distraction_score",
    "first_attention_time_sec",
    "first_moderate_risk_time_sec",
    "first_high_risk_time_sec",
    "first_alert_time_sec",
    "first_low_vigilance_time_sec",
    "first_drowsiness_time_sec",
    "detected_yawns",
    "detected_long_eye_closures",
    "predicted_class",
    "is_correct_basic",
    "is_preventive_detection",
    "false_positive_flag",
    "false_negative_flag",
    "processing_error",
    "notes",
]

FRAME_COLUMNS = [
    "video_id",
    "timestamp_sec",
    "frame_id",
    "face_detected",
    "ear_avg",
    "mar",
    "eye_closed",
    "mouth_open",
    "perclos",
    "blink_count",
    "long_eye_closure_count",
    "avg_blink_duration",
    "avg_pupil_movement",
    "pupil_movement_samples",
    "yawn_count",
    "head_tilt_score",
    "looking_down",
    "looking_side",
    "fatigue_score",
    "distraction_score",
    "final_state",
    "alert_message",
]


DEFAULT_VALIDATION_CONFIG = {
    "classification": {
        "warmup_sec": 10,
        "low_vigilance_min_fatigue": 35,
        "low_vigilance_min_duration_sec": 5,
        "drowsy_min_fatigue": 50,
        "drowsy_min_duration_sec": 2,
    },
    "event_detection": {
        "early_drowsiness_min_fatigue": 31,
        "drowsiness_min_fatigue": 55,
        "microsleep_min_perclos": 0.25,
        "distraction_event_min_score": 31,
        "no_face_max_valid_face_ratio": 0.75,
        "none_max_fatigue_below": 31,
    },
    "false_positive": {
        "strong_fatigue_min": 31,
        "strong_distraction_min": 81,
    },
}

DEFAULT_SCORE_LEVELS = {
    "attention_min": 31,
    "moderate_min": 61,
    "high_min": 81,
}


def _section(config: dict[str, Any] | None, key: str) -> dict[str, Any]:
    merged = dict(DEFAULT_VALIDATION_CONFIG[key])
    if config:
        merged.update(config.get(key, {}) or {})
    return merged


def _levels(score_levels: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_SCORE_LEVELS)
    if score_levels:
        merged.update(score_levels)
    return merged


def sustained_event_time(
    df: pd.DataFrame,
    threshold: float,
    warmup_sec: float,
    min_duration_sec: float,
) -> tuple[bool, float | str, float]:
    """Detecta se o score ficou acima do limiar por uma duracao minima."""
    data = df[df["timestamp_sec"] >= warmup_sec]
    if data.empty:
        return False, "", 0.0

    step = float(data["timestamp_sec"].diff().median() or 1.0)
    best_duration = 0.0
    current_duration = 0.0
    current_start: float | None = None
    best_start: float | None = None

    for _, row in data.iterrows():
        if float(row["fatigue_score"]) >= threshold:
            if current_duration == 0.0:
                current_start = float(row["timestamp_sec"])
            current_duration += step
            if current_duration > best_duration:
                best_duration = current_duration
                best_start = current_start
        else:
            current_duration = 0.0
            current_start = None

    detected = best_duration >= min_duration_sec
    return detected, best_start if detected and best_start is not None else "", best_duration


def predict_class_from_timeline(
    df: pd.DataFrame,
    validation_config: dict[str, Any] | None = None,
) -> tuple[str, float | str, float | str, float, float]:
    """Classifica o video como uma simulacao temporal com alerta sustentado."""
    classification = _section(validation_config, "classification")
    warmup_sec = float(classification["warmup_sec"])
    low_vigilance_min = float(classification["low_vigilance_min_fatigue"])
    low_vigilance_duration = float(classification["low_vigilance_min_duration_sec"])
    drowsy_min = float(classification["drowsy_min_fatigue"])
    drowsy_duration = float(classification["drowsy_min_duration_sec"])

    drowsy_detected, drowsy_start, drowsy_best_duration = sustained_event_time(
        df, drowsy_min, warmup_sec, drowsy_duration
    )
    low_detected, low_start, low_best_duration = sustained_event_time(
        df, low_vigilance_min, warmup_sec, low_vigilance_duration
    )

    if drowsy_detected:
        return "drowsy", low_start, drowsy_start, low_best_duration, drowsy_best_duration
    if low_detected:
        return "low_vigilance", low_start, "", low_best_duration, drowsy_best_duration
    return "alert", "", "", low_best_duration, drowsy_best_duration


def predict_class(max_fatigue_score: float, validation_config: dict[str, Any] | None = None) -> str:
    """Compatibilidade para usos simples: classifica por pico."""
    classification = _section(validation_config, "classification")
    if max_fatigue_score >= float(classification["drowsy_min_fatigue"]):
        return "drowsy"
    if max_fatigue_score >= float(classification["low_vigilance_min_fatigue"]):
        return "low_vigilance"
    return "alert"


def first_time(df: pd.DataFrame, condition) -> float | str:
    matches = df.loc[condition(df)]
    if matches.empty:
        return ""
    return float(matches.iloc[0]["timestamp_sec"])


def bool_text(value: bool) -> bool:
    return bool(value)


def evaluate_expected_event(
    row: dict[str, Any],
    metrics: dict[str, Any],
    predicted_class: str,
    validation_config: dict[str, Any] | None = None,
) -> bool:
    """Avalia se o evento esperado foi detectado por regras simples."""
    event_config = _section(validation_config, "event_detection")
    expected_event = str(row.get("expected_event", "none")).lower()
    expected_class = str(row.get("expected_class", "")).lower()
    max_fatigue = float(metrics["max_fatigue_score"])
    max_distraction = float(metrics["max_distraction_score"])
    detected_yawns = int(metrics["detected_yawns"])
    detected_long_closures = int(metrics["detected_long_eye_closures"])
    max_perclos = float(metrics["max_perclos"])

    if expected_event == "yawn":
        return detected_yawns > 0
    if expected_event in {"microsleep", "eyes_closed"}:
        return (
            detected_long_closures > 0
            or max_perclos >= float(event_config["microsleep_min_perclos"])
            or max_fatigue >= float(event_config["drowsiness_min_fatigue"])
        )
    if expected_event == "early_drowsiness":
        return predicted_class in {"low_vigilance", "drowsy"}
    if expected_event == "drowsiness":
        return predicted_class == "drowsy"
    if expected_event == "looking_down":
        return max_distraction >= float(event_config["distraction_event_min_score"])
    if expected_event == "looking_side":
        return max_distraction >= float(event_config["distraction_event_min_score"])
    if expected_event == "no_face":
        return (
            max_distraction >= float(event_config["distraction_event_min_score"])
            and float(metrics["valid_face_ratio"]) < float(event_config["no_face_max_valid_face_ratio"])
        )
    if expected_event in {"none", "talking"} or expected_class in {"alert", "normal"}:
        return predicted_class == "alert"
    return (
        max_fatigue >= float(event_config["early_drowsiness_min_fatigue"])
        or max_distraction >= float(event_config["distraction_event_min_score"])
    )


def preventive_detection(
    row: dict[str, Any],
    metrics: dict[str, Any],
    predicted_class: str,
    validation_config: dict[str, Any] | None = None,
) -> bool:
    expected_class = str(row.get("expected_class", "")).lower()
    expected_event = str(row.get("expected_event", "")).lower()

    if expected_class == "low_vigilance" or expected_event == "early_drowsiness":
        return predicted_class in {"low_vigilance", "drowsy"}
    if expected_class == "drowsy" or expected_event in {"drowsiness", "microsleep"}:
        return predicted_class == "drowsy"
    return False


def is_false_positive(
    row: dict[str, Any],
    metrics: dict[str, Any],
    predicted_class: str,
    validation_config: dict[str, Any] | None = None,
) -> bool:
    expected_class = str(row.get("expected_class", "")).lower()
    expected_event = str(row.get("expected_event", "")).lower()
    max_distraction = float(metrics["max_distraction_score"])
    detected_yawns = int(metrics["detected_yawns"])

    benign = expected_class in {"alert", "normal"} and expected_event in {"none", "talking"}
    if not benign:
        return False
    false_positive_config = _section(validation_config, "false_positive")
    return predicted_class != "alert" or max_distraction >= float(false_positive_config["strong_distraction_min"]) or (
        expected_event == "talking" and detected_yawns > 0
    )


def summarize_video(
    row: dict[str, Any],
    frame_csv: Path,
    duration_sec: float,
    validation_config: dict[str, Any] | None = None,
    score_levels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cria uma linha consolidada de avaliacao para um video processado."""
    df = pd.read_csv(frame_csv)
    if df.empty:
        return error_summary(row, "Video processado sem frames validos.", duration_sec=duration_sec)

    levels = _levels(score_levels)
    attention_min = float(levels["attention_min"])
    moderate_min = float(levels["moderate_min"])
    high_min = float(levels["high_min"])
    max_yawns = int(df["yawn_count"].max()) if "yawn_count" in df else 0
    max_long_closures = int(df["long_eye_closure_count"].max()) if "long_eye_closure_count" in df else 0
    validation_classification = _section(validation_config, "classification")
    predicted, low_start, drowsy_start, low_best_duration, drowsy_best_duration = predict_class_from_timeline(
        df, validation_config
    )
    metrics = {
        "duration_sec": duration_sec,
        "valid_face_ratio": float(df["face_detected"].astype(bool).mean()),
        "mean_perclos": float(df["perclos"].mean()),
        "max_perclos": float(df["perclos"].max()),
        "mean_fatigue_score": float(df["fatigue_score"].mean()),
        "max_fatigue_score": float(df["fatigue_score"].max()),
        "mean_distraction_score": float(df["distraction_score"].mean()),
        "max_distraction_score": float(df["distraction_score"].max()),
        "first_attention_time_sec": first_time(
            df, lambda data: (data["fatigue_score"] >= attention_min) | (data["distraction_score"] >= attention_min)
        ),
        "first_moderate_risk_time_sec": first_time(
            df, lambda data: (data["fatigue_score"] >= moderate_min) | (data["distraction_score"] >= moderate_min)
        ),
        "first_high_risk_time_sec": first_time(
            df, lambda data: (data["fatigue_score"] >= high_min) | (data["distraction_score"] >= high_min)
        ),
        "first_alert_time_sec": first_time(df, lambda data: data["alert_message"].fillna("").astype(str).ne("")),
        "first_low_vigilance_time_sec": low_start,
        "first_drowsiness_time_sec": drowsy_start,
        "detected_yawns": max_yawns,
        "detected_long_eye_closures": max_long_closures,
    }
    correct = evaluate_expected_event(row, metrics, predicted, validation_config)
    preventive = preventive_detection(row, metrics, predicted, validation_config)
    false_positive = is_false_positive(row, metrics, predicted, validation_config)
    false_negative = not correct and not false_positive

    return {
        "video_id": row.get("video_id", ""),
        "video_path": row.get("video_path", ""),
        "dataset": row.get("dataset", ""),
        "expected_class": row.get("expected_class", ""),
        "expected_event": row.get("expected_event", ""),
        **metrics,
        "predicted_class": predicted,
        "is_correct_basic": bool_text(correct),
        "is_preventive_detection": bool_text(preventive),
        "false_positive_flag": bool_text(false_positive),
        "false_negative_flag": bool_text(false_negative),
        "processing_error": "",
        "notes": row.get("notes", ""),
    }


def error_summary(row: dict[str, Any], error: str, duration_sec: float = 0.0) -> dict[str, Any]:
    """Linha de resumo para videos que falharam, mantendo o lote de pe."""
    summary = {column: "" for column in SUMMARY_COLUMNS}
    summary.update(
        {
            "video_id": row.get("video_id", ""),
            "video_path": row.get("video_path", ""),
            "dataset": row.get("dataset", ""),
            "expected_class": row.get("expected_class", ""),
            "expected_event": row.get("expected_event", ""),
            "duration_sec": duration_sec,
            "predicted_class": "",
            "is_correct_basic": False,
            "is_preventive_detection": False,
            "false_positive_flag": False,
            "false_negative_flag": False,
            "processing_error": error,
            "notes": row.get("notes", ""),
        }
    )
    return summary
