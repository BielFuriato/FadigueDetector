"""Logger CSV leve para registros por frame."""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from .utils import ensure_dir


LOG_COLUMNS = [
    "timestamp",
    "frame_id",
    "face_detected",
    "ear_left",
    "ear_right",
    "ear_avg",
    "mar",
    "eye_closed",
    "mouth_open",
    "perclos",
    "yawn_count_window",
    "long_eye_closure_count",
    "avg_blink_duration",
    "avg_pupil_movement",
    "pupil_movement_samples",
    "head_tilt_score",
    "looking_down",
    "looking_side",
    "fatigue_score",
    "distraction_score",
    "final_state",
    "alert_message",
    "fps",
]


class CsvLogger:
    """Escreve CSV em thread separada para reduzir impacto no loop."""

    def __init__(self, output_dir: str | Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.output_dir = ensure_dir(output_dir)
        self.path = self.output_dir / f"session_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._queue: Queue[dict[str, Any] | None] = Queue()
        self._thread: threading.Thread | None = None

        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def log(self, row: dict[str, Any]) -> None:
        if self.enabled:
            self._queue.put(row)

    def close(self) -> None:
        if not self.enabled:
            return
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _worker(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
            writer.writeheader()

            while True:
                try:
                    item = self._queue.get(timeout=0.5)
                except Empty:
                    continue

                if item is None:
                    break

                writer.writerow({column: item.get(column, "") for column in LOG_COLUMNS})
                self._queue.task_done()
