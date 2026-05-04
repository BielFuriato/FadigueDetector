"""Teste PVT simples de tempo de reacao no terminal."""

from __future__ import annotations

import csv
import random
import statistics
import time
from datetime import datetime
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "logs" / "pvt_records.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    participant = input("Codigo ou nome do participante: ").strip()
    attempts_text = input("Quantidade de tentativas [10]: ").strip()
    attempts = int(attempts_text) if attempts_text.isdigit() else 10

    print("\nPressione ENTER o mais rapido possivel quando aparecer o estimulo.")
    print("Aguarde cada rodada sem pressionar antes do sinal.\n")
    input("Pressione ENTER para iniciar.")

    reaction_times: list[float] = []
    for attempt in range(1, attempts + 1):
        wait_time = random.uniform(2.0, 5.0)
        print(f"Rodada {attempt}/{attempts}: aguarde...")
        time.sleep(wait_time)
        print("AGORA!")
        start = time.perf_counter()
        input()
        reaction_ms = (time.perf_counter() - start) * 1000.0
        reaction_times.append(reaction_ms)
        print(f"Tempo de reacao: {reaction_ms:.0f} ms\n")

    mean_ms = statistics.mean(reaction_times)
    median_ms = statistics.median(reaction_times)
    lapses = sum(1 for value in reaction_times if value > 500)
    exists = output.exists()

    with output.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timestamp", "participant", "attempts", "mean_ms", "median_ms", "lapses_over_500ms"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "participant": participant,
                "attempts": attempts,
                "mean_ms": f"{mean_ms:.2f}",
                "median_ms": f"{median_ms:.2f}",
                "lapses_over_500ms": lapses,
            }
        )

    print(f"Media: {mean_ms:.0f} ms | Mediana: {median_ms:.0f} ms | Lapsos > 500 ms: {lapses}")
    print(f"Resultado salvo em: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
