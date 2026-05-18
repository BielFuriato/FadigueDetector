"""Interface Gradio/FastRTC para o detector de fadiga."""

from __future__ import annotations

import os
import socket
import tempfile
import time
import uuid
from html import escape
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import cv2

from src.utils import load_config

import gradio as gr

try:
    from fastrtc import AdditionalOutputs, WebRTC, get_cloudflare_turn_credentials_async
except ImportError:  # pragma: no cover - permite mensagem clara se a dependencia faltar.
    AdditionalOutputs = None
    WebRTC = None
    get_cloudflare_turn_credentials_async = None
    get_current_context = None
else:
    try:
        from fastrtc import get_current_context
    except ImportError:  # pragma: no cover - compatibilidade com versoes sem contexto publico.
        get_current_context = None


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
SAMPLE_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CONFIG = load_config(CONFIG_PATH)
LIVE_PROCESSORS: dict[str, Any] = {}
STATE_SEVERITY = {
    "Normal": 0,
    "Atencao": 1,
    "Risco por fadiga": 2,
}
APP_CSS = """
.gradio-container {
    max-width: 1180px !important;
}
/* Header & navigation — must stay above video at all times */
.app-header {
    margin-bottom: 0.75rem;
    position: relative;
    z-index: 20;
}
.app-header h1 {
    margin-bottom: 0.15rem;
}
.app-header p {
    margin: 0.15rem 0;
}
.tabs {
    position: relative;
    z-index: 15 !important;
}
nav {
    position: relative;
    z-index: 15 !important;
}
/* Video panels */
.video-panel video,
#upload-video video,
#processed-video video {
    width: 100% !important;
    max-height: min(52vh, 400px) !important;
    object-fit: contain !important;
    background: #111827 !important;
}
/* Live webcam */
.live-webcam-layout {
    align-items: stretch;
    gap: 1rem;
}
.live-camera-column,
.live-metrics-column {
    min-width: 0 !important;
}
.live-camera-panel {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    min-height: 420px;
    overflow: hidden;
}
.live-camera-panel > div {
    border: 0 !important;
    box-shadow: none !important;
}
#live-webcam {
    width: 100% !important;
    min-height: 420px !important;
    background: #111827;
    overflow: hidden;
}
#live-webcam video {
    width: 100% !important;
    height: 100% !important;
    max-height: 420px !important;
    object-fit: contain !important;
    background: #111827 !important;
}
.live-metrics-column {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    overflow: hidden;
}
.live-metrics-column > * {
    overflow: hidden;
}
.live-actions-row {
    align-items: center;
    justify-content: flex-start;
}
/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(142px, 1fr));
    gap: 0.65rem;
    margin: 0.25rem 0 0.8rem;
}
.metric-card {
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 0.75rem;
    background: #ffffff;
    min-height: 82px;
}
.metric-label {
    color: #4b5563;
    font-size: 0.78rem;
    line-height: 1.1;
    margin-bottom: 0.35rem;
}
.metric-value {
    color: #111827;
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1.1;
    overflow-wrap: anywhere;
}
.metric-note {
    color: #4b5563;
    font-size: 0.82rem;
    line-height: 1.25;
    margin-top: 0.25rem;
}
.state-normal {
    border-left: 4px solid #16a34a;
    background: #f0fdf4;
}
.state-normal .metric-value {
    color: #15803d;
}
.state-attention {
    border-left: 4px solid #ca8a04;
    background: #fffbeb;
}
.state-attention .metric-value {
    color: #a16207;
}
.state-risk {
    border-left: 4px solid #dc2626;
    background: #fef2f2;
}
.state-risk .metric-value {
    color: #b91c1c;
}
.summary-line {
    color: #374151;
    font-size: 0.92rem;
    margin: 0.25rem 0 0.8rem;
}
.warning-line {
    border-left: 4px solid #ca8a04;
    background: #fffbeb;
    padding: 0.65rem 0.75rem;
    border-radius: 6px;
    color: #713f12;
    margin-top: 0.75rem;
}
/* In-tab warning banner — visible even if header scrolls away */
.live-warning-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #ca8a04;
    border-radius: 6px;
    margin-bottom: 0.75rem;
    font-size: 0.84rem;
    color: #713f12;
}
.live-warning-bar strong {
    color: #92400e;
}
"""


def create_processor(*, sound_enabled: bool = False):
    """Importa MediaPipe apenas quando a analise comeca."""
    from src.session_processor import FatigueSessionProcessor

    return FatigueSessionProcessor(CONFIG, sound_enabled=sound_enabled)


def discover_sample_videos() -> list[list[str]]:
    """Retorna exemplos no formato esperado por gr.Examples."""
    if not SAMPLES_DIR.exists():
        return []
    files = sorted(path for path in SAMPLES_DIR.iterdir() if path.suffix.lower() in SAMPLE_EXTENSIONS)
    return [[str(path)] for path in files]


def resolve_video_path(video_file: Any) -> Path | None:
    if not video_file:
        return None
    if isinstance(video_file, dict):
        candidate = video_file.get("video") or video_file.get("name") or video_file.get("path")
        return Path(candidate) if candidate else None
    return Path(str(video_file))


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def round_float(value: Any, digits: int = 3) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def state_severity(state: Any) -> int:
    return STATE_SEVERITY.get(str(state), 0)


def state_class(state: Any) -> str:
    severity = state_severity(state)
    if severity >= 2:
        return "state-risk"
    if severity == 1:
        return "state-attention"
    return "state-normal"


def int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def metric_card(label: str, value: Any, note: str = "", css_class: str = "") -> str:
    note_html = f'<div class="metric-note">{escape(str(note))}</div>' if note else ""
    class_attr = f"metric-card {css_class}".strip()
    return (
        f'<div class="{class_attr}">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(str(value))}</div>'
        f"{note_html}"
        "</div>"
    )


def format_upload_summary_html(payload: dict[str, Any], output_warning: str = "") -> str:
    if payload.get("processing_error"):
        message = payload.get("message", "Nenhum frame valido foi analisado.")
        return f"<p>{escape(str(message))}</p>"

    cards = [
        metric_card(
            "Avaliacao final",
            payload["final_state"],
            f"pior estado observado no frame {payload['final_frame_id']}",
            state_class(payload["final_state"]),
        ),
        metric_card(
            "Pico de fadiga",
            f"{payload['max_fatigue_score']:.1f}",
            f"frame {payload['peak_fatigue_frame']}",
        ),
        metric_card(
            "PERCLOS pico",
            f"{payload['max_perclos_percent']:.1f}%",
            f"frame {payload['peak_perclos_frame']}",
        ),
        metric_card(
            "Eventos",
            f"{payload['alert_seconds']:.1f}s em alerta",
            (
                f"{payload['detected_yawns']} bocejo(s), "
                f"{payload['detected_long_eye_closures']} fechamento(s)"
            ),
        ),
    ]
    warning_html = f'<div class="warning-line">{escape(output_warning)}</div>' if output_warning else ""
    alert_html = escape(payload.get("final_alert") or "-")
    return (
        '<div class="metric-grid">'
        + "".join(cards)
        + "</div>"
        f'<div class="summary-line">'
        f"Duracao: {payload['duration_sec']:.2f}s | "
        f"Frames: {payload['frames_analyzed']} | "
        f"Face valida: {payload['valid_face_ratio'] * 100:.1f}% | "
        f"Alerta associado: {alert_html}"
        "</div>"
        f"{warning_html}"
    )


def format_live_html(payload: dict[str, Any]) -> str:
    state = payload.get("state", "Aguardando dados")
    cards = [
        metric_card("Estado", state, payload.get("alert") or "-", state_class(state)),
        metric_card("PERCLOS", f"{payload.get('perclos', 0.0):.1f}%"),
        metric_card(
            "Fadiga",
            f"{payload.get('fatigue_score', 0.0):.1f}",
            payload.get("fatigue_level", "NORMAL"),
        ),
        metric_card("EAR / MAR", f"{payload.get('ear', 0.0):.3f} / {payload.get('mar', 0.0):.3f}"),
        metric_card(
            "Captura",
            f"{payload.get('fps', 0.0):.1f} FPS",
            "face detectada" if payload.get("face_detected") else "sem face",
        ),
    ]
    return '<div id="live-metrics-root"><div class="metric-grid">' + "".join(cards) + "</div></div>"


def current_metrics_payload(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {
            "state": "Aguardando dados",
            "ear": 0.0,
            "mar": 0.0,
            "perclos": 0.0,
            "fatigue_score": 0.0,
            "face_detected": False,
            "alert": "",
            "fps": 0.0,
        }

    metrics = result.get("metrics", {})
    temporal = result.get("temporal", {})
    risk = result.get("risk", {})
    alert = result.get("alert", {})
    return {
        "state": risk.get("final_state", "Normal"),
        "ear": round_float(metrics.get("ear_avg")),
        "mar": round_float(metrics.get("mar")),
        "perclos": round_float(float(temporal.get("perclos", 0.0)) * 100, 1),
        "fatigue_score": round_float(risk.get("fatigue_score"), 1),
        "fatigue_level": risk.get("fatigue_level", "NORMAL"),
        "face_detected": bool(result.get("face_detected")),
        "alert": alert.get("message", ""),
        "fps": round_float(result.get("fps"), 1),
        "frame_id": int(result.get("frame_id", 0)),
    }


def estimate_alert_seconds(alert_count: int, frames_analyzed: int, duration_sec: float) -> float:
    if alert_count <= 0 or frames_analyzed <= 0 or duration_sec <= 0:
        return 0.0
    return (alert_count / frames_analyzed) * duration_sec


def summarize_upload(rows: list[dict[str, Any]], duration_sec: float, output_warning: str = "") -> tuple[str, dict[str, Any]]:
    if not rows:
        message = "Nenhum frame valido foi analisado."
        if output_warning:
            message += f" {output_warning}"
        return f"<p>{escape(message)}</p>", {"processing_error": "empty_result", "message": message}

    fatigue_scores = [float(row["fatigue_score"]) for row in rows]
    perclos_values = [float(row["perclos"]) for row in rows]
    valid_face_count = sum(1 for row in rows if row["face_detected"])
    alert_count = sum(1 for row in rows if row.get("alert_message"))
    worst_row = max(
        rows,
        key=lambda row: (
            state_severity(row.get("final_state")),
            round_float(row.get("fatigue_score"), 3),
            round_float(row.get("perclos"), 3),
            int_value(row.get("frame_id")),
        ),
    )
    peak_fatigue_row = max(rows, key=lambda row: round_float(row.get("fatigue_score"), 3))
    peak_perclos_row = max(rows, key=lambda row: round_float(row.get("perclos"), 3))
    payload = {
        "duration_sec": round_float(duration_sec, 2),
        "frames_analyzed": len(rows),
        "valid_face_ratio": round_float(valid_face_count / len(rows), 3),
        "mean_perclos_percent": round_float(safe_mean(perclos_values) * 100, 1),
        "max_perclos_percent": round_float(max(perclos_values) * 100, 1),
        "mean_fatigue_score": round_float(safe_mean(fatigue_scores), 1),
        "max_fatigue_score": round_float(max(fatigue_scores), 1),
        "detected_yawns": int(max(row["yawn_count_window"] for row in rows)),
        "detected_long_eye_closures": int(max(row["long_eye_closure_count"] for row in rows)),
        "alert_frames": alert_count,
        "alert_seconds": round_float(estimate_alert_seconds(alert_count, len(rows), duration_sec), 1),
        "final_state": worst_row["final_state"],
        "final_alert": worst_row.get("alert_message", ""),
        "final_frame_id": int_value(worst_row.get("frame_id")),
        "peak_fatigue_frame": int_value(peak_fatigue_row.get("frame_id")),
        "peak_perclos_frame": int_value(peak_perclos_row.get("frame_id")),
        "decision_basis": "worst_observed_state",
    }

    summary = format_upload_summary_html(payload, output_warning)
    return summary, payload


def analyze_uploaded_video(video_file: Any, progress: gr.Progress = gr.Progress(track_tqdm=False)):
    video_path = resolve_video_path(video_file)
    if video_path is None:
        return None, "Envie um video para iniciar a analise.", {}
    if not video_path.exists():
        return None, f"Arquivo nao encontrado: `{video_path}`.", {"processing_error": "file_not_found"}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None, "Nao foi possivel abrir o video enviado.", {"processing_error": "open_failed"}

    processor = create_processor(sound_enabled=False)
    writer = None
    output_path = None
    rows: list[dict[str, Any]] = []
    output_warning = ""

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or CONFIG.get("camera", {}).get("fps", 30) or 30)
        fps = fps if fps > 0 else 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if width > 0 and height > 0:
            output_handle = tempfile.NamedTemporaryFile(prefix="fadigue_detector_", suffix=".mp4", delete=False)
            output_path = Path(output_handle.name)
            output_handle.close()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                writer.release()
                writer = None
                output_path = None
                output_warning = "O video anotado nao pode ser gerado; o resumo foi calculado normalmente."

        frame_id = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_sec = frame_id / max(fps, 1.0)
            annotated_frame, result = processor.process_frame(
                frame,
                timestamp=timestamp_sec,
                input_color="bgr",
                fps=fps,
            )
            rows.append(result["log_row"])
            if writer is not None:
                writer.write(annotated_frame)

            frame_id += 1
            if frame_count > 0 and frame_id % max(int(fps), 1) == 0:
                progress(min(frame_id / frame_count, 1.0), desc="Processando video")

        duration_sec = frame_id / max(fps, 1.0)
        summary, payload = summarize_upload(rows, duration_sec, output_warning)
        return str(output_path) if output_path and output_path.exists() else None, summary, payload
    except Exception as exc:
        return None, f"Erro durante a analise: {exc}", {"processing_error": str(exc)}
    finally:
        if writer is not None:
            writer.release()
        processor.close()
        capture.release()


def live_session_key(session_id: str | None) -> str:
    if session_id:
        return session_id
    if get_current_context is not None:
        try:
            context = get_current_context()
            webrtc_id = getattr(context, "webrtc_id", None)
            if webrtc_id:
                return str(webrtc_id)
        except Exception:
            pass
    return uuid.uuid4().hex


def get_live_processor(session_id: str):
    processor = LIVE_PROCESSORS.get(session_id)
    if processor is None:
        processor = create_processor(sound_enabled=False)
        LIVE_PROCESSORS[session_id] = processor
    return processor


def reset_live_session(session_id: str | None):
    if session_id:
        processor = LIVE_PROCESSORS.pop(session_id, None)
        if processor is not None:
            processor.close()
    payload = current_metrics_payload(None)
    return None, format_live_html(payload), payload


def process_live_frame(frame, session_id: str | None):
    if frame is None:
        return None

    session_key = live_session_key(session_id)
    processor = get_live_processor(session_key)
    annotated_frame, result = processor.process_frame(frame, timestamp=time.time(), input_color="bgr")

    payload = current_metrics_payload(result)
    last_update = float(getattr(processor, "_last_ui_update", 0.0))
    now = time.monotonic()
    if AdditionalOutputs is not None and now - last_update >= 0.5:
        setattr(processor, "_last_ui_update", now)
        return annotated_frame, AdditionalOutputs(session_key, format_live_html(payload), payload)
    return annotated_frame


def rtc_kwargs() -> dict[str, Any]:
    if get_cloudflare_turn_credentials_async is None:
        return {}
    has_turn_env = bool(
        os.getenv("HF_TOKEN")
        or os.getenv("CLOUDFLARE_TURN_KEY_ID")
        or os.getenv("CLOUDFLARE_TURN_KEY_API_TOKEN")
    )
    return {"rtc_configuration": get_cloudflare_turn_credentials_async} if has_turn_env else {}


def find_available_port(start: int = 7860, end: int = 8059) -> int:
    for port in range(start, end + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except PermissionError:
            return start
        with sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except PermissionError:
                return start
            except OSError:
                continue
            return port
    raise OSError(f"Nenhuma porta livre encontrada entre {start} e {end}.")


def build_app() -> gr.Blocks:
    sample_examples = discover_sample_videos()
    initial_live_payload = current_metrics_payload(None)

    with gr.Blocks(title="Driver Fatigue Detector", css=APP_CSS) as demo:
        gr.HTML(
            '<div class="app-header">'
            "<h1>Driver Fatigue Detector</h1>"
            "<p>Analise visual de fadiga com OpenCV, MediaPipe e regras interpretaveis.</p>"
            "</div>"
        )

        with gr.Tab("Analyze Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_video = gr.Video(
                        label="Video",
                        sources=["upload"],
                        elem_id="upload-video",
                        elem_classes=["video-panel"],
                    )
                    analyze_button = gr.Button("Analyze Video", variant="primary")
                    if sample_examples:
                        gr.Examples(examples=sample_examples, inputs=[upload_video], label="Samples")
                    else:
                        gr.Markdown("Coloque videos em `data/samples/` para habilitar exemplos pre-carregados.")
                with gr.Column(scale=1):
                    processed_video = gr.Video(
                        label="Annotated video",
                        format="mp4",
                        elem_id="processed-video",
                        elem_classes=["video-panel"],
                    )
                    upload_summary = gr.HTML("<p>Aguardando video.</p>")
                    with gr.Accordion("Detailed JSON", open=False):
                        upload_json = gr.JSON(label="Structured summary")

            analyze_button.click(
                fn=analyze_uploaded_video,
                inputs=[upload_video],
                outputs=[processed_video, upload_summary, upload_json],
            )

        with gr.Tab("Live Webcam"):
            if WebRTC is None:
                gr.Markdown("FastRTC nao esta instalado. Instale as dependencias com `pip install -r requirements.txt`.")
            else:
                live_session = gr.State(value=None)
                with gr.Row(elem_classes=["live-webcam-layout"], equal_height=False):
                    with gr.Column(scale=3, min_width=420, elem_classes=["live-camera-column"]):
                        with gr.Group(elem_classes=["live-camera-panel"]):
                            webcam = WebRTC(
                                label="Live webcam",
                                mode="send-receive",
                                modality="video",
                                height=420,
                                width="100%",
                                min_width=360,
                                full_screen=False,
                                elem_id="live-webcam",
                                **rtc_kwargs(),
                            )
                    with gr.Column(scale=2, min_width=320, elem_classes=["live-metrics-column"]):
                        live_metrics = gr.HTML(format_live_html(initial_live_payload))
                        with gr.Row(elem_classes=["live-actions-row"]):
                            reset_button = gr.Button("Reset Session", variant="secondary")
                        with gr.Accordion("Live JSON", open=False):
                            live_json = gr.JSON(label="Live metrics", value=initial_live_payload)

                webcam.stream(
                    fn=process_live_frame,
                    inputs=[webcam, live_session],
                    outputs=[webcam],
                )
                webcam.on_additional_outputs(
                    lambda session_id, markdown, payload: (session_id, markdown, payload),
                    outputs=[live_session, live_metrics, live_json],
                    queue=False,
                    show_progress="hidden",
                )
                reset_button.click(
                    fn=reset_live_session,
                    inputs=[live_session],
                    outputs=[live_session, live_metrics, live_json],
                    queue=False,
                )

    return demo


if __name__ == "__main__":
    launch_kwargs: dict[str, Any] = {}
    if not os.getenv("GRADIO_SERVER_PORT"):
        launch_kwargs["server_port"] = find_available_port()
    build_app().queue().launch(**launch_kwargs)
