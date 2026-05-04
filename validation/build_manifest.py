"""Gera um manifest CSV a partir de videos locais classificados."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

MANIFEST_COLUMNS = [
    "video_id",
    "video_path",
    "dataset",
    "subject_id",
    "expected_class",
    "expected_event",
    "event_start_sec",
    "event_end_sec",
    "notes",
]

FOLDER_RULES = {
    ("uta_rldd", "alert"): ("UTA-RLDD", "alert", "none"),
    ("uta_rldd", "low_vigilance"): ("UTA-RLDD", "low_vigilance", "early_drowsiness"),
    ("uta_rldd", "drowsy"): ("UTA-RLDD", "drowsy", "drowsiness"),
    ("nitymed", "yawning"): ("NITYMED", "yawning", "yawn"),
    ("nitymed", "microsleep"): ("NITYMED", "microsleep", "microsleep"),
    ("yawdd", "normal"): ("YawDD", "alert", "none"),
    ("yawdd", "talking"): ("YawDD", "alert", "talking"),
    ("yawdd", "yawning"): ("YawDD", "yawning", "yawn"),
}

OWN_EVENT_RULES = {
    "normal": ("normal", "none"),
    "eyes_closed": ("eyes_closed", "microsleep"),
    "yawn": ("yawn", "yawn"),
    "looking_down": ("looking_down", "looking_down"),
    "looking_side": ("looking_side", "looking_side"),
    "no_face": ("no_face", "no_face"),
}


def infer_subject_id(video_path: Path) -> str:
    """Tenta inferir um identificador simples pelo nome do arquivo."""
    stem = video_path.stem
    patterns = [
        r"(?:subject|participant|driver|person|user|subj|s|p)[_-]?(\d+)",
        r"^(\d+)[_-]",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def safe_video_id(relative_path: Path, used_ids: set[str]) -> str:
    """Cria um id estavel e unico a partir do caminho relativo."""
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", "_".join(relative_path.with_suffix("").parts)).strip("_")
    candidate = base or "video"
    counter = 2
    while candidate in used_ids:
        candidate = f"{base}_{counter}"
        counter += 1
    used_ids.add(candidate)
    return candidate


def classify_from_path(relative_path: Path) -> tuple[str, str, str] | None:
    """Retorna dataset, classe esperada e evento esperado pelo caminho."""
    parts = [part.lower() for part in relative_path.parts]
    if len(parts) < 3:
        return None

    dataset_folder, class_folder = parts[0], parts[1]
    if dataset_folder == "own_recordings":
        expected_class, expected_event = OWN_EVENT_RULES.get(class_folder, (class_folder, class_folder))
        return "Own", expected_class, expected_event

    return FOLDER_RULES.get((dataset_folder, class_folder))


def build_manifest(videos_dir: Path) -> list[dict[str, str]]:
    """Varre videos_dir e monta linhas do manifest."""
    rows: list[dict[str, str]] = []
    used_ids: set[str] = set()

    if not videos_dir.exists():
        return rows

    for video_path in sorted(videos_dir.rglob("*")):
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        relative = video_path.relative_to(videos_dir)
        classification = classify_from_path(relative)
        if classification is None:
            continue

        dataset, expected_class, expected_event = classification
        rows.append(
            {
                "video_id": safe_video_id(relative, used_ids),
                "video_path": str(video_path),
                "dataset": dataset,
                "subject_id": infer_subject_id(video_path),
                "expected_class": expected_class,
                "expected_event": expected_event,
                "event_start_sec": "",
                "event_end_sec": "",
                "notes": "",
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera manifest de validacao a partir de data/test_videos.")
    parser.add_argument("--videos-dir", default="data/test_videos", help="Pasta raiz dos videos classificados.")
    parser.add_argument("--output", default="data/validation/manifests/validation_manifest.csv", help="CSV de saida.")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = build_manifest(videos_dir)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest salvo em: {output}")
    print(f"Videos encontrados: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

