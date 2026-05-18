"""Gerenciamento de alertas visuais e sonoros."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass


@dataclass
class Alert:
    message: str = ""
    level: str = "NORMAL"


class AlertManager:
    """Escolhe mensagens e evita repeticao sonora a cada frame."""

    def __init__(self, sound_enabled: bool = True, cooldown_sec: float = 5.0) -> None:
        self.sound_enabled = sound_enabled
        self.cooldown_sec = cooldown_sec
        self._last_sound_time = 0.0

    def evaluate(self, risk_result, temporal: dict, current: dict) -> Alert:
        """Retorna o alerta mais importante para o estado atual."""
        message = ""
        level = "NORMAL"

        if temporal.get("current_eye_closure_duration", 0.0) >= current.get("long_eye_closure_sec", 2.0):
            message = "Risco: olhos fechados por tempo prolongado"
            level = "RISCO_ALTO"
        elif risk_result.final_state == "Risco por fadiga":
            message = "Atencao: sinais de sonolencia"
            level = risk_result.fatigue_level
        elif current.get("yawn_detected_recently"):
            message = "Atencao: bocejo detectado"
            level = "ATENCAO"
        elif current.get("head_tilt_score", 0.0) > 65:
            message = "Atencao: cabeca inclinada"
            level = "ATENCAO"
        elif risk_result.final_state == "Atencao":
            message = "Atencao: sinais iniciais de risco"
            level = "ATENCAO"

        alert = Alert(message=message, level=level)
        if message and level in {"RISCO_MODERADO", "RISCO_ALTO", "ATENCAO"}:
            self._maybe_play_sound()
        return alert

    def _maybe_play_sound(self) -> None:
        now = time.time()
        if not self.sound_enabled or now - self._last_sound_time < self.cooldown_sec:
            return
        self._last_sound_time = now

        try:
            if sys.platform.startswith("win"):
                import winsound

                winsound.Beep(1200, 180)
            else:
                print("\a", end="", flush=True)
        except Exception:
            # O alerta sonoro e opcional; falha de audio nao deve derrubar o sistema.
            pass
