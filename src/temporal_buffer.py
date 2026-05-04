"""Buffer temporal para decisoes baseadas em janelas de tempo."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemporalBuffer:
    """Armazena frames recentes e calcula eventos temporais."""

    window_sec: float = 60.0
    yawn_window_sec: float = 120.0
    long_eye_closure_sec: float = 2.0
    yawn_min_duration_sec: float = 1.5
    face_missing_sec: float = 3.0
    records: deque[dict[str, Any]] = field(default_factory=deque)
    yawn_events: deque[float] = field(default_factory=deque)
    blink_events: deque[float] = field(default_factory=deque)
    long_eye_closure_events: deque[float] = field(default_factory=deque)
    eye_closure_durations: deque[tuple[float, float]] = field(default_factory=deque)
    blink_durations: deque[tuple[float, float]] = field(default_factory=deque)
    pupil_movements: deque[tuple[float, float]] = field(default_factory=deque)
    _eye_closed_since: float | None = None
    _long_closure_counted: bool = False
    _mouth_open_since: float | None = None
    _yawn_active: bool = False
    _face_missing_since: float | None = None
    _looking_side_since: float | None = None
    _looking_down_since: float | None = None
    _last_pupil_position: tuple[float, float, float] | None = None
    total_yawn_count: int = 0

    def add(self, record: dict[str, Any]) -> dict[str, Any]:
        """Adiciona um registro e atualiza eventos de piscada/bocejo/ausencia."""
        timestamp = float(record["timestamp"])
        self.records.append(record)
        self._update_eye_state(timestamp, bool(record.get("eye_closed")))
        self._update_yawn_state(timestamp, bool(record.get("mouth_open")))
        self._update_face_missing_state(timestamp, bool(record.get("face_detected")))
        self._update_attention_state(timestamp, bool(record.get("looking_side")), bool(record.get("looking_down")))
        self._update_pupil_state(
            timestamp,
            record.get("pupil_x"),
            record.get("pupil_y"),
            bool(record.get("face_detected")) and not bool(record.get("eye_closed")),
        )
        self._trim(timestamp)
        return self.summary(timestamp)

    def _update_eye_state(self, timestamp: float, eye_closed: bool) -> None:
        if eye_closed and self._eye_closed_since is None:
            self._eye_closed_since = timestamp
            self._long_closure_counted = False

        if eye_closed and self._eye_closed_since is not None:
            duration = timestamp - self._eye_closed_since
            if duration >= self.long_eye_closure_sec and not self._long_closure_counted:
                self.long_eye_closure_events.append(timestamp)
                self._long_closure_counted = True

        if not eye_closed and self._eye_closed_since is not None:
            duration = timestamp - self._eye_closed_since
            self.eye_closure_durations.append((timestamp, duration))
            if 0.08 <= duration < self.long_eye_closure_sec:
                self.blink_events.append(timestamp)
                self.blink_durations.append((timestamp, duration))
            self._eye_closed_since = None
            self._long_closure_counted = False

    def _update_yawn_state(self, timestamp: float, mouth_open: bool) -> None:
        if mouth_open and self._mouth_open_since is None:
            self._mouth_open_since = timestamp

        if mouth_open and self._mouth_open_since is not None:
            duration = timestamp - self._mouth_open_since
            if duration >= self.yawn_min_duration_sec and not self._yawn_active:
                self.yawn_events.append(timestamp)
                self.total_yawn_count += 1
                self._yawn_active = True

        if not mouth_open:
            self._mouth_open_since = None
            self._yawn_active = False

    def _update_face_missing_state(self, timestamp: float, face_detected: bool) -> None:
        if face_detected:
            self._face_missing_since = None
        elif self._face_missing_since is None:
            self._face_missing_since = timestamp

    def _update_attention_state(self, timestamp: float, looking_side: bool, looking_down: bool) -> None:
        if looking_side and self._looking_side_since is None:
            self._looking_side_since = timestamp
        elif not looking_side:
            self._looking_side_since = None

        if looking_down and self._looking_down_since is None:
            self._looking_down_since = timestamp
        elif not looking_down:
            self._looking_down_since = None

    def _update_pupil_state(self, timestamp: float, pupil_x: Any, pupil_y: Any, valid: bool) -> None:
        if not valid or pupil_x is None or pupil_y is None:
            self._last_pupil_position = None
            return

        current_x = float(pupil_x)
        current_y = float(pupil_y)
        if self._last_pupil_position is not None:
            previous_timestamp, previous_x, previous_y = self._last_pupil_position
            delta_sec = timestamp - previous_timestamp
            if delta_sec > 1e-6:
                movement = ((current_x - previous_x) ** 2 + (current_y - previous_y) ** 2) ** 0.5
                self.pupil_movements.append((timestamp, movement / delta_sec))
        self._last_pupil_position = (timestamp, current_x, current_y)

    def _trim(self, now: float) -> None:
        risk_cutoff = now - self.window_sec
        yawn_cutoff = now - self.yawn_window_sec

        while self.records and float(self.records[0]["timestamp"]) < risk_cutoff:
            self.records.popleft()
        while self.blink_events and self.blink_events[0] < risk_cutoff:
            self.blink_events.popleft()
        while self.long_eye_closure_events and self.long_eye_closure_events[0] < risk_cutoff:
            self.long_eye_closure_events.popleft()
        while self.eye_closure_durations and self.eye_closure_durations[0][0] < risk_cutoff:
            self.eye_closure_durations.popleft()
        while self.blink_durations and self.blink_durations[0][0] < risk_cutoff:
            self.blink_durations.popleft()
        while self.pupil_movements and self.pupil_movements[0][0] < risk_cutoff:
            self.pupil_movements.popleft()
        while self.yawn_events and self.yawn_events[0] < yawn_cutoff:
            self.yawn_events.popleft()

    def perclos(self) -> float:
        """Percentual de frames validos com olhos fechados na janela."""
        valid = [record for record in self.records if record.get("face_detected")]
        if not valid:
            return 0.0
        closed = sum(1 for record in valid if record.get("eye_closed"))
        return closed / len(valid)

    def ratio(self, key: str, face_only: bool = False) -> float:
        """Proporcao de frames verdadeiros para uma chave booleana."""
        records = list(self.records)
        if face_only:
            records = [record for record in records if record.get("face_detected")]
        if not records:
            return 0.0
        positive = sum(1 for record in records if record.get(key))
        return positive / len(records)

    def current_eye_closure_duration(self, now: float) -> float:
        if self._eye_closed_since is None:
            return 0.0
        return max(0.0, now - self._eye_closed_since)

    def face_missing_duration(self, now: float) -> float:
        if self._face_missing_since is None:
            return 0.0
        return max(0.0, now - self._face_missing_since)

    def looking_side_duration(self, now: float) -> float:
        if self._looking_side_since is None:
            return 0.0
        return max(0.0, now - self._looking_side_since)

    def looking_down_duration(self, now: float) -> float:
        if self._looking_down_since is None:
            return 0.0
        return max(0.0, now - self._looking_down_since)

    def avg_eye_closure_duration(self) -> float:
        if not self.eye_closure_durations:
            return 0.0
        return sum(duration for _, duration in self.eye_closure_durations) / len(self.eye_closure_durations)

    def avg_blink_duration(self) -> float:
        if not self.blink_durations:
            return 0.0
        return sum(duration for _, duration in self.blink_durations) / len(self.blink_durations)

    def avg_pupil_movement(self) -> float:
        if not self.pupil_movements:
            return 0.0
        return sum(movement for _, movement in self.pupil_movements) / len(self.pupil_movements)

    def summary(self, now: float) -> dict[str, Any]:
        """Retorna estatisticas consolidadas para score, alerta e log."""
        return {
            "perclos": self.perclos(),
            "yawn_count_window": len(self.yawn_events),
            "blink_count": len(self.blink_events),
            "long_eye_closure_count": len(self.long_eye_closure_events),
            "avg_eye_closure_duration": self.avg_eye_closure_duration(),
            "avg_blink_duration": self.avg_blink_duration(),
            "current_eye_closure_duration": self.current_eye_closure_duration(now),
            "avg_pupil_movement": self.avg_pupil_movement(),
            "pupil_movement_samples": len(self.pupil_movements),
            "looking_down_ratio": self.ratio("looking_down", face_only=True),
            "looking_side_ratio": self.ratio("looking_side", face_only=True),
            "non_frontal_ratio": 1.0 - self.ratio("face_frontal", face_only=True),
            "face_missing_ratio": 1.0 - self.ratio("face_detected") if self.records else 0.0,
            "face_missing_duration": self.face_missing_duration(now),
            "looking_side_duration": self.looking_side_duration(now),
            "looking_down_duration": self.looking_down_duration(now),
            "total_yawn_count": self.total_yawn_count,
        }
