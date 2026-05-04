"""Registro simples da escala KSS para comparacao experimental."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "logs" / "kss_records.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    participant = input("Codigo ou nome do participante: ").strip()
    while True:
        try:
            kss = int(input("KSS (1 a 9): ").strip())
            if 1 <= kss <= 9:
                break
        except ValueError:
            pass
        print("Informe um valor inteiro entre 1 e 9.")

    notes = input("Observacoes: ").strip()
    exists = output.exists()

    with output.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "participant", "kss", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "participant": participant,
                "kss": kss,
                "notes": notes,
            }
        )

    print(f"Registro KSS salvo em: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
