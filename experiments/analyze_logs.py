"""Analise simples dos logs gerados pelo detector."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def latest_session_log(log_dir: Path) -> Path | None:
    files = sorted(log_dir.glob("session_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Analisa logs CSV do detector.")
    parser.add_argument("--file", help="Caminho de um CSV especifico.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    log_dir = root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.file) if args.file else latest_session_log(log_dir)

    if csv_path is None or not csv_path.exists():
        print("Nenhum log session_*.csv encontrado em data/logs.")
        return 1

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Log vazio: {csv_path}")
        return 1

    risk_mask = df["final_state"].astype(str).str.contains("Risco", case=False, na=False)
    alert_count = df["alert_message"].fillna("").astype(str).ne("").sum()
    yawn_count = int(df["yawn_count_window"].max()) if "yawn_count_window" in df else 0
    fps_mean = max(float(df["fps"].replace(0, pd.NA).dropna().mean() or 30.0), 1.0)
    total_risk_sec = float(risk_mask.sum() / fps_mean)

    summary = {
        "arquivo": str(csv_path),
        "perclos_medio": float(df["perclos"].mean()),
        "perclos_maximo": float(df["perclos"].max()),
        "fadiga_media": float(df["fatigue_score"].mean()),
        "fadiga_maxima": float(df["fatigue_score"].max()),
        "quantidade_alertas": int(alert_count),
        "quantidade_bocejos": yawn_count,
        "tempo_total_em_risco_sec": total_risk_sec,
    }

    print("\nResumo do log")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.3f}")
        else:
            print(f"- {key}: {value}")

    x = df["frame_id"] if "frame_id" in df else df.index
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots = [
        ("perclos", "PERCLOS ao longo do tempo", "perclos"),
        ("fatigue_score", "Score de fadiga ao longo do tempo", "fatigue_score"),
    ]

    for column, title, filename_part in plots:
        plt.figure(figsize=(10, 4))
        plt.plot(x, df[column], linewidth=1.5)
        plt.title(title)
        plt.xlabel("Frame")
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        output = log_dir / f"{filename_part}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(output, dpi=140)
        plt.close()
        print(f"Grafico salvo: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
