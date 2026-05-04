"""Executa validacao em lote com videos locais classificados."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts import AlertManager
from src.camera import CameraStream
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
from validation.evaluator import FRAME_COLUMNS, SUMMARY_COLUMNS, error_summary, summarize_video
from validation.report_generator import generate_report


def empty_metrics() -> dict[str, Any]:
    return {
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


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def filtered_manifest(
    manifest: pd.DataFrame,
    dataset_filter: str | None,
    class_filter: str | None,
    max_videos: int | None,
) -> pd.DataFrame:
    df = manifest.copy()
    if dataset_filter:
        accepted = {item.strip().lower() for item in dataset_filter.split(",") if item.strip()}
        df = df[df["dataset"].astype(str).str.lower().isin(accepted)]
    if class_filter:
        accepted = {item.strip().lower() for item in class_filter.split(",") if item.strip()}
        df = df[df["expected_class"].astype(str).str.lower().isin(accepted)]
    if max_videos is not None:
        df = df.head(max_videos)
    return df


def video_duration_sec(camera: CameraStream, fallback_fps: float) -> float:
    if camera.capture is None:
        return 0.0
    frame_count = float(camera.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    fps = float(camera.capture.get(cv2.CAP_PROP_FPS) or fallback_fps or 0.0)
    if frame_count > 0 and fps > 0:
        return frame_count / fps
    return 0.0


def write_summary_csv(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    """Salva o resumo acumulado para nao perder progresso em lotes longos."""
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def process_video(
    manifest_row: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    detector: FaceLandmarkDetector,
    no_display: bool = True,
    verbose: bool = False,
    sample_fps: float | None = None,
) -> dict[str, Any]:
    """Processa um video e retorna a linha consolidada do resumo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = str(manifest_row["video_id"])
    video_path = resolve_project_path(str(manifest_row["video_path"]))
    frame_csv = output_dir / f"{video_id}_frames.csv"

    if not video_path.exists():
        return error_summary(manifest_row, f"Arquivo nao encontrado: {video_path}")

    thresholds = config.get("thresholds", {})
    temporal_config = config.get("temporal", {})
    camera_config = config.get("camera", {})

    camera = CameraStream(source=str(video_path))
    buffer = TemporalBuffer(
        window_sec=float(temporal_config.get("risk_window_sec", temporal_config.get("perclos_window_sec", 60))),
        yawn_window_sec=float(temporal_config.get("yawn_window_sec", 120)),
        long_eye_closure_sec=float(thresholds.get("long_eye_closure_sec", 2.0)),
        yawn_min_duration_sec=float(thresholds.get("yawn_min_duration_sec", 1.5)),
        face_missing_sec=float(thresholds.get("face_missing_sec", 3.0)),
    )
    risk_model = RiskModel(thresholds, config.get("scores", {}))
    alerts = AlertManager(sound_enabled=False, cooldown_sec=9999)
    previous_total_yawn_count = 0
    duration = 0.0

    try:
        camera.open()
        nominal_fps = camera.get_fps() or float(camera_config.get("fps", 30) or 30)
        frame_stride = 1
        if sample_fps is not None and sample_fps > 0:
            frame_stride = max(1, int(round(nominal_fps / sample_fps)))
        duration = video_duration_sec(camera, nominal_fps)

        with frame_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FRAME_COLUMNS)
            writer.writeheader()

            frame_id = 0
            target_frame_index = 0
            total_frames = int(camera.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if camera.capture is not None else 0
            while True:
                if frame_stride > 1 and camera.capture is not None:
                    if total_frames > 0 and target_frame_index >= total_frames:
                        break
                    camera.capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
                    ok, frame = camera.read()
                    if not ok:
                        break
                    frame_id = target_frame_index + 1
                    target_frame_index += frame_stride
                else:
                    ok, frame = camera.read()
                    if not ok:
                        break
                    frame_id += 1

                timestamp_sec = (frame_id - 1) / max(nominal_fps, 1.0)
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

                writer.writerow(
                    {
                        "video_id": video_id,
                        "timestamp_sec": timestamp_sec,
                        "frame_id": frame_id,
                        "face_detected": face is not None,
                        "ear_avg": metrics["ear_avg"],
                        "mar": metrics["mar"],
                        "eye_closed": metrics["eye_closed"],
                        "mouth_open": metrics["mouth_open"],
                        "perclos": temporal["perclos"],
                        "blink_count": temporal["blink_count"],
                        "long_eye_closure_count": temporal["long_eye_closure_count"],
                        "avg_blink_duration": temporal["avg_blink_duration"],
                        "avg_pupil_movement": temporal["avg_pupil_movement"],
                        "pupil_movement_samples": temporal["pupil_movement_samples"],
                        "yawn_count": temporal["yawn_count_window"],
                        "head_tilt_score": metrics["head_tilt_score"],
                        "looking_down": metrics["looking_down"],
                        "looking_side": metrics["looking_side"],
                        "fatigue_score": risk.fatigue_score,
                        "distraction_score": risk.distraction_score,
                        "final_state": risk.final_state,
                        "alert_message": alert.message,
                    }
                )

                if not no_display:
                    draw_landmarks(frame, landmarks_px)
                    draw_status_panel(frame, metrics, temporal, risk, alert, nominal_fps)
                    cv2.imshow("Validation", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if verbose and frame_id % max(int(nominal_fps * 10), 1) == 1:
                    print(f"  {video_id}: {timestamp_sec:.1f}s processados...")

        return summarize_video(
            manifest_row,
            frame_csv,
            duration,
            validation_config=config.get("validation", {}),
            score_levels=config.get("scores", {}).get("levels", {}),
        )

    except Exception as exc:
        return error_summary(manifest_row, str(exc), duration_sec=duration)
    finally:
        camera.release()
        if not no_display:
            cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida o pipeline atual em lote com videos locais.")
    parser.add_argument("--manifest", required=True, help="CSV manifest gerado por build_manifest.py.")
    parser.add_argument("--config", default="config.yaml", help="Arquivo YAML de configuracao do detector.")
    parser.add_argument("--no-display", action="store_true", help="Nao abre janela OpenCV.")
    parser.add_argument("--output-dir", default="data/validation/outputs", help="Pasta dos CSVs de saida.")
    parser.add_argument("--max-videos", type=int, help="Processa no maximo N videos.")
    parser.add_argument("--dataset-filter", help="Filtra datasets por nome, separados por virgula.")
    parser.add_argument("--class-filter", help="Filtra classes esperadas por nome, separadas por virgula.")
    parser.add_argument("--sample-fps", type=float, help="Processa ate N frames por segundo por video para acelerar testes.")
    parser.add_argument("--verbose", action="store_true", help="Mostra progresso por video.")
    args = parser.parse_args()

    manifest_path = resolve_project_path(args.manifest)
    config_path = resolve_project_path(args.config)
    output_dir = resolve_project_path(args.output_dir)
    reports_dir = PROJECT_ROOT / "data" / "validation" / "reports"
    plots_dir = PROJECT_ROOT / "data" / "validation" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"Manifest nao encontrado: {manifest_path}")
        return 1

    manifest = pd.read_csv(manifest_path).fillna("")
    selected = filtered_manifest(manifest, args.dataset_filter, args.class_filter, args.max_videos)
    config = load_config(config_path)
    summary_path = output_dir / "validation_summary.csv"

    print(f"Videos no manifest: {len(manifest)}")
    print(f"Videos selecionados: {len(selected)}")

    summary_rows: list[dict[str, Any]] = []
    if not selected.empty:
        detector = FaceLandmarkDetector()
        try:
            for index, (_, row) in enumerate(selected.iterrows(), start=1):
                manifest_row = row.to_dict()
                print(f"[{index}/{len(selected)}] Processando {manifest_row.get('video_id')}...")
                result = process_video(
                    manifest_row=manifest_row,
                    config=config,
                    output_dir=output_dir,
                    detector=detector,
                    no_display=args.no_display,
                    verbose=args.verbose,
                    sample_fps=args.sample_fps,
                )
                summary_rows.append(result)
                write_summary_csv(summary_path, summary_rows)
                if result.get("processing_error"):
                    print(f"  erro: {result['processing_error']}")
        finally:
            detector.close()
            cv2.destroyAllWindows()

    write_summary_csv(summary_path, summary_rows)

    report_path = generate_report(
        summary_csv=summary_path,
        outputs_dir=output_dir,
        reports_dir=reports_dir,
        plots_dir=plots_dir,
    )

    print(f"Resumo salvo em: {summary_path}")
    print(f"Relatorio salvo em: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
