"""Gera um clipe MP4 anotado para demonstracao do detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts import AlertManager
from src.face_landmarks import FaceLandmarkDetector
from src.metrics import (
    calculate_eye_aspect_ratio,
    calculate_mouth_aspect_ratio,
    calculate_pupil_position,
    estimate_head_pose_2d,
)
from src.risk_model import RiskModel
from src.temporal_buffer import TemporalBuffer
from src.utils import load_config
from src.visualization import draw_landmarks, draw_status_panel


DEFAULT_SOURCE = Path("data/test_videos/uta_rldd/drowsy/10_5.MOV")
DEFAULT_OUTPUT = Path("demo_apresentacao/saida/demo_sonolencia_10_5.mp4")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera video anotado para demonstracao em sala.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Video de entrada.")
    parser.add_argument("--config", default="config.yaml", help="Arquivo YAML de configuracao.")
    parser.add_argument("--start-sec", type=float, default=390.0, help="Inicio visivel do clipe.")
    parser.add_argument("--duration-sec", type=float, default=90.0, help="Duracao visivel do clipe.")
    parser.add_argument("--preload-sec", type=float, default=120.0, help="Aquecimento temporal antes do clipe.")
    parser.add_argument("--render-fps", type=float, default=10.0, help="FPS do MP4 gerado.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Arquivo MP4 de saida.")
    parser.add_argument("--demo-title", default="DEMO SONOLENCIA", help="Titulo exibido no overlay.")
    parser.add_argument("--expected-label", default="UTA-RLDD / drowsy", help="Classe esperada exibida no overlay.")
    parser.add_argument("--no-landmarks", action="store_true", help="Nao desenha pontos faciais.")
    return parser


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def empty_metrics() -> dict[str, Any]:
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


def build_temporal_buffer(config: dict[str, Any]) -> TemporalBuffer:
    thresholds = config.get("thresholds", {})
    temporal_config = config.get("temporal", {})
    return TemporalBuffer(
        window_sec=float(temporal_config.get("risk_window_sec", temporal_config.get("perclos_window_sec", 60))),
        yawn_window_sec=float(temporal_config.get("yawn_window_sec", 120)),
        long_eye_closure_sec=float(thresholds.get("long_eye_closure_sec", 2.0)),
        yawn_min_duration_sec=float(thresholds.get("yawn_min_duration_sec", 1.5)),
        face_missing_sec=float(thresholds.get("face_missing_sec", 3.0)),
    )


def analyze_frame(
    frame,
    timestamp_sec: float,
    detector: FaceLandmarkDetector,
    buffer: TemporalBuffer,
    risk_model: RiskModel,
    alerts: AlertManager,
    config: dict[str, Any],
    previous_total_yawn_count: int,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any, int]:
    thresholds = config.get("thresholds", {})
    metrics = empty_metrics()
    landmarks_px = None

    face = detector.detect(frame)
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

    temporal = buffer.add(
        {
            "timestamp": timestamp_sec,
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
    )
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
    return metrics, temporal, risk, alert, landmarks_px, previous_total_yawn_count


def format_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, remaining = divmod(seconds, 60)
    return f"{minutes:02d}:{remaining:02d}"


def draw_demo_overlay(
    frame,
    timestamp_sec: float,
    start_sec: float,
    end_sec: float,
    source_name: str,
    demo_title: str,
    expected_label: str,
) -> None:
    height, width = frame.shape[:2]
    if width >= 900:
        x1, y1 = width - 426, 12
        x2, y2 = width - 12, 132
    else:
        x1, y1 = 12, 356
        x2, y2 = min(width - 12, 430), 476

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    lines = [
        (demo_title, 0.58, (255, 255, 255), 2),
        (f"Video: {source_name}", 0.48, (220, 220, 220), 1),
        (f"Tempo: {format_time(timestamp_sec)}", 0.48, (220, 220, 220), 1),
        (f"Trecho: {format_time(start_sec)}-{format_time(end_sec)} | {expected_label}", 0.43, (220, 220, 220), 1),
    ]
    y = y1 + 26
    for text, scale, color, thickness in lines:
        cv2.putText(frame, text, (x1 + 14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += 27


def draw_face_missing(frame) -> None:
    y = 388 if frame.shape[1] >= 900 else 520
    cv2.putText(
        frame,
        "Face nao detectada",
        (28, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def video_metadata(capture: cv2.VideoCapture, fallback_fps: float) -> tuple[float, int, int, int]:
    nominal_fps = float(capture.get(cv2.CAP_PROP_FPS) or fallback_fps or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return nominal_fps, frame_count, width, height


def create_writer(output_path: Path, render_fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, render_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Nao foi possivel criar o video de saida: {output_path}")
    return writer


def generate_demo(args: argparse.Namespace) -> Path:
    source_path = resolve_project_path(args.source)
    config_path = resolve_project_path(args.config)
    output_path = resolve_project_path(args.output)

    if not source_path.exists():
        raise FileNotFoundError(f"Video de entrada nao encontrado: {source_path}")
    if args.duration_sec <= 0:
        raise ValueError("--duration-sec deve ser maior que zero.")
    if args.render_fps <= 0:
        raise ValueError("--render-fps deve ser maior que zero.")

    config = load_config(config_path)
    camera_config = config.get("camera", {})
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {source_path}")

    detector = FaceLandmarkDetector()
    writer = None
    try:
        nominal_fps, frame_count, width, height = video_metadata(
            capture,
            float(camera_config.get("fps", 30) or 30),
        )
        if width <= 0 or height <= 0:
            raise RuntimeError("Nao foi possivel ler a resolucao do video.")

        frame_stride = max(1, int(round(nominal_fps / args.render_fps)))
        sampled_fps = nominal_fps / frame_stride
        start_sec = max(0.0, float(args.start_sec))
        end_sec = start_sec + float(args.duration_sec)
        preload_start_sec = max(0.0, start_sec - max(0.0, float(args.preload_sec)))
        start_frame = max(0, int(round(preload_start_sec * nominal_fps)))
        end_frame = min(frame_count - 1, int(round(end_sec * nominal_fps))) if frame_count > 0 else int(round(end_sec * nominal_fps))

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        buffer = build_temporal_buffer(config)
        risk_model = RiskModel(config.get("thresholds", {}), config.get("scores", {}))
        alerts = AlertManager(sound_enabled=False, cooldown_sec=9999)
        writer = create_writer(output_path, args.render_fps, width, height)
        previous_total_yawn_count = 0
        written_frames = 0
        processed_frames = 0
        visible_start_frame = int(round(start_sec * nominal_fps))
        source_name = source_path.name

        print(f"Entrada: {source_path}")
        print(f"Saida: {output_path}")
        print(f"Trecho visivel: {format_time(start_sec)} ate {format_time(end_sec)}")
        print(f"Aquecimento oculto: desde {format_time(preload_start_sec)}")
        print(
            f"FPS original: {nominal_fps:.2f} | "
            f"amostragem: {sampled_fps:.2f} | MP4: {args.render_fps:.2f}"
        )

        current_frame_index = start_frame
        while current_frame_index <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break

            if (current_frame_index - start_frame) % frame_stride != 0:
                current_frame_index += 1
                continue

            timestamp_sec = current_frame_index / max(nominal_fps, 1.0)
            metrics, temporal, risk, alert, landmarks_px, previous_total_yawn_count = analyze_frame(
                frame,
                timestamp_sec,
                detector,
                buffer,
                risk_model,
                alerts,
                config,
                previous_total_yawn_count,
            )
            processed_frames += 1

            if current_frame_index >= visible_start_frame:
                if not args.no_landmarks:
                    draw_landmarks(frame, landmarks_px)
                draw_status_panel(frame, metrics, temporal, risk, alert, args.render_fps)
                draw_demo_overlay(
                    frame,
                    timestamp_sec,
                    start_sec,
                    end_sec,
                    source_name,
                    args.demo_title,
                    args.expected_label,
                )
                if landmarks_px is None:
                    draw_face_missing(frame)
                writer.write(frame)
                written_frames += 1

                if written_frames % max(int(args.render_fps * 10), 1) == 0:
                    elapsed = timestamp_sec - start_sec
                    print(
                        f"  renderizados {elapsed:5.1f}s | "
                        f"estado={risk.final_state} | fadiga={risk.fatigue_score:5.1f}"
                    )

            current_frame_index += 1

        if written_frames == 0:
            raise RuntimeError("Nenhum frame foi gravado. Verifique inicio, duracao e video de entrada.")

        print(f"Frames processados: {processed_frames}")
        print(f"Frames gravados: {written_frames}")
        return output_path
    finally:
        if writer is not None:
            writer.release()
        detector.close()
        capture.release()
        cv2.destroyAllWindows()


def main() -> int:
    args = build_parser().parse_args()
    try:
        output_path = generate_demo(args)
    except Exception as exc:
        print(f"Erro: {exc}")
        return 1

    print(f"Demo gerada com sucesso: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
