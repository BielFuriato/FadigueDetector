"""Entrada principal do prototipo de deteccao de fadiga."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.camera import CameraStream
from src.logger import CsvLogger
from src.session_processor import FatigueSessionProcessor
from src.utils import load_config, parse_camera_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detector academico de fadiga em motoristas.")
    parser.add_argument("--source", help="Indice da webcam ou caminho de video.")
    parser.add_argument("--config", default="config.yaml", help="Caminho do arquivo YAML de configuracao.")
    parser.add_argument("--no-sound", action="store_true", help="Desativa alerta sonoro.")
    parser.add_argument("--no-log", action="store_true", help="Desativa gravacao de logs CSV.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = load_config(config_path)
    camera_config = config.get("camera", {})
    alert_config = config.get("alerts", {})
    logging_config = config.get("logging", {})

    source = parse_camera_source(args.source if args.source is not None else camera_config.get("source", 0))
    camera = CameraStream(
        source=source,
        width=int(camera_config.get("width", 1280)),
        height=int(camera_config.get("height", 720)),
        fps=int(camera_config.get("fps", 30)),
    )
    processor = FatigueSessionProcessor(
        config,
        sound_enabled=bool(alert_config.get("sound_enabled", True)) and not args.no_sound,
    )

    output_dir = Path(logging_config.get("output_dir", "data/logs"))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    logger = CsvLogger(output_dir=output_dir, enabled=bool(logging_config.get("enabled", True)) and not args.no_log)

    frame_id = 0

    try:
        camera.open()
        nominal_fps = camera.get_fps() or float(camera_config.get("fps", 30) or 30)
        print("Detector iniciado. Pressione Q para sair.")

        while True:
            ok, frame = camera.read()
            if not ok:
                print("Fim do video ou falha na captura.")
                break

            frame_id += 1
            timestamp = time.time() if isinstance(source, int) else (frame_id - 1) / max(nominal_fps, 1.0)
            annotated_frame, result = processor.process_frame(frame, timestamp=timestamp, input_color="bgr")
            logger.log(result["log_row"])

            cv2.imshow("Driver Fatigue Detector", annotated_frame)
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
        processor.close()
        camera.release()
        cv2.destroyAllWindows()

    print("Recursos liberados. Encerrado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
