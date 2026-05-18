"""Modelo de risco baseado em regras temporais ajustaveis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .utils import clamp


DEFAULT_SCORE_CONFIG = {
    "levels": {
        "attention_min": 31,
        "moderate_min": 61,
        "high_min": 81,
    },
    "fatigue": {
        "perclos_reference": 0.40,
        "perclos_weight": 55.0,
        "yawn_reference_count": 4.0,
        "yawn_weight": 15.0,
        "long_closure_reference_count": 2.0,
        "long_closure_weight": 18.0,
        "active_long_closure_min_score": 25.0,
        "blink_duration_enabled": True,
        "blink_duration_reference_sec": 0.35,
        "blink_duration_weight": 7.0,
        "pupil_movement_enabled": False,
        "pupil_movement_reference": 0.12,
        "pupil_movement_min_samples": 10,
        "pupil_movement_weight": 8.0,
        "head_tilt_weight": 5.0,
    },
    "final_decision": {
        "fatigue_high_min": 61,
    },
}


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    merged = dict(DEFAULT_SCORE_CONFIG[key])
    merged.update(config.get(key, {}) or {})
    return merged


def _safe_reference(value: float) -> float:
    return max(float(value), 1e-6)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "sim", "on"}
    return bool(value)


@dataclass
class RiskResult:
    fatigue_score: float
    fatigue_level: str
    final_state: str


class RiskModel:
    """Combina indicadores em scores simples de 0 a 100."""

    def __init__(self, thresholds: dict[str, Any], score_config: dict[str, Any] | None = None) -> None:
        self.long_eye_closure_sec = float(thresholds.get("long_eye_closure_sec", 2.0))
        self.score_config = score_config or {}
        self.levels = _section(self.score_config, "levels")
        self.fatigue = _section(self.score_config, "fatigue")
        self.final_decision = _section(self.score_config, "final_decision")

    def evaluate(self, temporal: dict[str, Any], current_head_tilt_score: float) -> RiskResult:
        """Calcula fadiga e estado final.

        PERCLOS recebe maior peso por ser o indicador ocular temporal principal.
        Bocejos, fechamentos prolongados e cabeca inclinada entram como reforcos.
        """
        perclos = float(temporal.get("perclos", 0.0))
        yawn_count = int(temporal.get("yawn_count_window", 0))
        long_closures = int(temporal.get("long_eye_closure_count", 0))
        avg_blink_duration = float(
            temporal.get("avg_blink_duration", temporal.get("avg_eye_closure_duration", 0.0))
        )
        current_closure = float(temporal.get("current_eye_closure_duration", 0.0))
        avg_pupil_movement = float(temporal.get("avg_pupil_movement", 0.0))
        pupil_movement_samples = int(temporal.get("pupil_movement_samples", 0))

        perclos_score = min(perclos / _safe_reference(self.fatigue["perclos_reference"]), 1.0) * float(
            self.fatigue["perclos_weight"]
        )
        yawn_score = min(yawn_count / _safe_reference(self.fatigue["yawn_reference_count"]), 1.0) * float(
            self.fatigue["yawn_weight"]
        )
        long_closure_score = min(
            long_closures / _safe_reference(self.fatigue["long_closure_reference_count"]), 1.0
        ) * float(self.fatigue["long_closure_weight"])
        blink_duration_score = 0.0
        if _as_bool(self.fatigue["blink_duration_enabled"]):
            blink_duration_score = min(
                avg_blink_duration / _safe_reference(self.fatigue["blink_duration_reference_sec"]), 1.0
            ) * float(self.fatigue["blink_duration_weight"])

        pupil_movement_score = 0.0
        if (
            _as_bool(self.fatigue["pupil_movement_enabled"])
            and pupil_movement_samples >= int(self.fatigue["pupil_movement_min_samples"])
        ):
            movement_ratio = min(avg_pupil_movement / _safe_reference(self.fatigue["pupil_movement_reference"]), 1.0)
            pupil_movement_score = (1.0 - movement_ratio) * float(self.fatigue["pupil_movement_weight"])

        head_score = min(current_head_tilt_score / 100.0, 1.0) * float(self.fatigue["head_tilt_weight"])

        if current_closure >= self.long_eye_closure_sec:
            long_closure_score = max(long_closure_score, float(self.fatigue["active_long_closure_min_score"]))

        fatigue_score = clamp(
            perclos_score
            + yawn_score
            + long_closure_score
            + blink_duration_score
            + pupil_movement_score
            + head_score
        )

        fatigue_level = self._level_from_score(fatigue_score)
        final_state = self._final_state(fatigue_score)

        return RiskResult(
            fatigue_score=fatigue_score,
            fatigue_level=fatigue_level,
            final_state=final_state,
        )

    def _level_from_score(self, score: float) -> str:
        if score < float(self.levels["attention_min"]):
            return "NORMAL"
        if score < float(self.levels["moderate_min"]):
            return "ATENCAO"
        if score < float(self.levels["high_min"]):
            return "RISCO_MODERADO"
        return "RISCO_ALTO"

    def _final_state(self, fatigue_score: float) -> str:
        fatigue_high = fatigue_score >= float(self.final_decision["fatigue_high_min"])
        if fatigue_high:
            return "Risco por fadiga"
        if fatigue_score >= float(self.levels["attention_min"]):
            return "Atencao"
        return "Normal"
