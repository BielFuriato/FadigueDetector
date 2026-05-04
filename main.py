"""Entrada principal do prototipo de deteccao de fadiga e distracao."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.alerts import AlertManager
from src.camera import CameraStream
from src.face_landmarks import FaceLandmarkDetector
from src.logger import CsvLogger
from src.metrics import (
    calculate_eye_aspect_ratio,
    calculate_mouth_aspect_ratio,
    calculate_pupil_position,
    estimate_head_pose_2d,
)
from src.risk_model import RiskModel
from src.temporal_buffer import TemporalBuffer
from src.utils import load_config, parse_camera_source
from src.visualization import draw_landmarks, draw_status_panel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detector academico de fadiga e distracao em motoristas.")
    parser.add_argument("--source", help="Indice da webcam ou caminho de video.")
    parser.add_argument("--config", default="config.yaml", help="Caminho do arquivo YAML de configuracao.")
    parser.add_argument("--no-sound", action="store_true", help="Desativa alerta sonoro.")
    parser.add_argument("--no-log", action="store_true", help="Desativa gravacao de logs CSV.")
    return parser


def empty_metrics() -> dict:
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


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = load_config(config_path)
    camera_config = config.get("camera", {})
    thresholds = config.get("thresholds", {})
    temporal_config = config.get("temporal", {})
    alert_config = config.get("alerts", {})
    logging_config = config.get("logging", {})

    source = parse_camera_source(args.source if args.source is not None else camera_config.get("source", 0))
    camera = CameraStream(
        source=source,
        width=int(camera_config.get("width", 1280)),
        height=int(camera_config.get("height", 720)),
        fps=int(camera_config.get("fps", 30)),
    )
    detector = FaceLandmarkDetector()
    buffer = TemporalBuffer(
        window_sec=float(temporal_config.get("risk_window_sec", temporal_config.get("perclos_window_sec", 60))),
        yawn_window_sec=float(temporal_config.get("yawn_window_sec", 120)),
        long_eye_closure_sec=float(thresholds.get("long_eye_closure_sec", 2.0)),
        yawn_min_duration_sec=float(thresholds.get("yawn_min_duration_sec", 1.5)),
        face_missing_sec=float(thresholds.get("face_missing_sec", 3.0)),
    )
    risk_model = RiskModel(thresholds, config.get("scores", {}))
    alerts = AlertManager(
        sound_enabled=bool(alert_config.get("sound_enabled", True)) and not args.no_sound,
        cooldown_sec=float(alert_config.get("cooldown_sec", 5)),
    )

    output_dir = Path(logging_config.get("output_dir", "data/logs"))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    logger = CsvLogger(output_dir=output_dir, enabled=bool(logging_config.get("enabled", True)) and not args.no_log)

    frame_id = 0
    fps = 0.0
    previous_time = time.perf_counter()
    previous_total_yawn_count = 0

    try:
        camera.open()
        print("Detector iniciado. Pressione Q para sair.")

        while True:
            ok, frame = camera.read()
            if not ok:
                print("Fim do video ou falha na captura.")
                break

            frame_id += 1
            now = time.time()
            current_time = time.perf_counter()
            instant_fps = 1.0 / max(current_time - previous_time, 1e-6)
            fps = instant_fps if fps == 0.0 else (0.90 * fps + 0.10 * instant_fps)
            previous_time = current_time

            face = detector.detect(frame)
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
                        "eye_closed": ear.average < float(thresholds.get("ear_closed", 0.21)),
                        "mouth_open": mar > float(thresholds.get("mar_open", 0.65)),
                        "pupil_x": pupil.x if pupil is not None else None,
                        "pupil_y": pupil.y if pupil is not None else None,
                        "head_tilt_score": head.head_tilt_score,
                        "looking_down": head.looking_down,
                        "looking_side": head.looking_side,
                        "face_frontal": head.face_frontal,
                    }
                )

            record = {
                "timestamp": now,
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
            temporal = buffer.add(record)
            yawn_detected_recently = temporal["total_yawn_count"] > previous_total_yawn_count
            previous_total_yawn_count = temporal["total_yawn_count"]

            risk = risk_model.evaluate(temporal, metrics["head_tilt_score"])
            alert = alerts.evaluate(
                risk,
                temporal,
                {
                    **metrics,
                    "long_eye_closure_sec": float(thresholds.get("long_eye_closure_sec", 2.0)),
                    "looking_side_sec": float(thresholds.get("looking_side_sec", 3.0)),
                    "looking_down_sec": float(thresholds.get("looking_down_sec", 3.0)),
                    "yawn_detected_recently": yawn_detected_recently,
                },
            )

            draw_landmarks(frame, landmarks_px)
            draw_status_panel(frame, metrics, temporal, risk, alert, fps)
            if face is None:
                cv2.putText(
                    frame,
                    "Face nao detectada",
                    (28, 388),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            logger.log(
                {
                    "timestamp": now,
                    "frame_id": frame_id,
                    "face_detected": face is not None,
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
                    "distraction_score": risk.distraction_score,
                    "final_state": risk.final_state,
                    "alert_message": alert.message,
                    "fps": fps,
                }
            )

            cv2.imshow("Driver Fatigue and Distraction Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nExecucao interrompida pelo usuario.")
    except Exception as exc:
        print(f"Erro: {exc}")
        return 1
    finally:
        logger.close()
        detector.close()
        camera.release()
        cv2.destroyAllWindows()

    print("Recursos liberados. Encerrado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
