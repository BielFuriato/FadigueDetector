"""Geracao de relatorio Markdown e graficos da validacao."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes", "sim"})


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 10) -> str:
    if df.empty:
        return "Nenhum item.\n"
    table = df[columns].head(max_rows).fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in table.to_numpy()]
    return "\n".join([header, separator, *rows])


def _counts_table(series: pd.Series, name: str) -> str:
    counts = series.value_counts(dropna=False).rename_axis(name).reset_index(name="count")
    return _markdown_table(counts, [name, "count"], max_rows=len(counts))


def _save_group_bar(df: pd.DataFrame, value_column: str, title: str, output: Path) -> None:
    grouped = df.groupby("expected_class")[value_column].agg(["mean", "max"]).sort_index()
    if grouped.empty:
        return
    grouped.plot(kind="bar", figsize=(10, 4))
    plt.title(title)
    plt.xlabel("Classe esperada")
    plt.ylabel(value_column)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=140)
    plt.close()


def _save_timeline(frame_csv: Path, output: Path) -> None:
    df = pd.read_csv(frame_csv)
    if df.empty:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp_sec"], df["fatigue_score"], label="fatigue_score", linewidth=1.3)
    plt.plot(df["timestamp_sec"], df["perclos"] * 100.0, label="PERCLOS (%)", linewidth=1.3)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Valor")
    plt.title(frame_csv.stem.replace("_frames", ""))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=140)
    plt.close()


def _save_confusion_matrix(df: pd.DataFrame, output: Path) -> str:
    try:
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    except Exception:
        return "Scikit-learn nao disponivel; matriz de confusao nao foi gerada."

    valid = df[df["processing_error"].fillna("").eq("")]
    if valid.empty:
        return "Sem videos processados para matriz de confusao."

    expected = valid["expected_class"].replace({"normal": "alert", "yawning": "drowsy"}).astype(str)
    predicted = valid["predicted_class"].astype(str)
    labels = sorted(set(expected) | set(predicted))
    matrix = confusion_matrix(expected, predicted, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(cmap="Blues", xticks_rotation=35)
    plt.title("Matriz de confusao aproximada")
    plt.tight_layout()
    plt.savefig(output, dpi=140)
    plt.close()
    return f"Matriz de confusao salva em `{output}`."


def generate_report(
    summary_csv: Path,
    outputs_dir: Path,
    reports_dir: Path,
    plots_dir: Path,
) -> Path:
    """Gera relatorio Markdown e graficos a partir do resumo consolidado."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv)

    if df.empty:
        report_path = reports_dir / "validation_report.md"
        report_path.write_text("# Relatorio de Validacao\n\nNenhum video foi processado.\n", encoding="utf-8")
        return report_path

    for column in ["is_correct_basic", "is_preventive_detection", "false_positive_flag", "false_negative_flag"]:
        if column in df:
            df[column] = _as_bool(df[column])

    processed = df[df["processing_error"].fillna("").eq("")]
    errors = df[df["processing_error"].fillna("").ne("")]
    total = len(df)
    processed_total = len(processed)
    accuracy = float(processed["is_correct_basic"].mean()) if processed_total else 0.0
    false_positives = int(processed["false_positive_flag"].sum()) if processed_total else 0
    false_negatives = int(processed["false_negative_flag"].sum()) if processed_total else 0

    low_vigilance = processed[processed["expected_class"].astype(str).str.lower().eq("low_vigilance")]
    preventive_rate = float(low_vigilance["is_preventive_detection"].mean()) if not low_vigilance.empty else 0.0

    _save_group_bar(processed, "max_fatigue_score", "Fatigue score por classe esperada", plots_dir / "fatigue_score_by_class.png")
    _save_group_bar(processed, "max_perclos", "PERCLOS por classe esperada", plots_dir / "perclos_by_class.png")
    confusion_note = _save_confusion_matrix(processed, plots_dir / "confusion_matrix.png")

    for frame_csv in sorted(outputs_dir.glob("*_frames.csv")):
        video_id = frame_csv.name.replace("_frames.csv", "")
        _save_timeline(frame_csv, plots_dir / f"timeline_{video_id}.png")

    best = processed.sort_values(["is_correct_basic", "max_fatigue_score"], ascending=[False, False])
    worst = processed[(~processed["is_correct_basic"]) | (processed["false_positive_flag"]) | (processed["false_negative_flag"])]
    worst = worst.sort_values(["false_negative_flag", "false_positive_flag", "max_fatigue_score"], ascending=[False, False, False])

    perclos_by_class = processed.groupby("expected_class")["max_perclos"].agg(["mean", "max"]).reset_index()
    fatigue_by_class = processed.groupby("expected_class")["max_fatigue_score"].agg(["mean", "max"]).reset_index()

    observations: list[str] = []
    if false_positives:
        observations.append("- Ha falsos positivos; revise limiares de EAR/MAR e postura para videos alertas.")
    if false_negatives:
        observations.append("- Ha falsos negativos; alguns eventos esperados nao atingiram os limiares atuais.")
    if not low_vigilance.empty and preventive_rate >= 0.7:
        observations.append("- A deteccao preventiva em baixa vigilancia foi consistente nesta amostra.")
    elif not low_vigilance.empty:
        observations.append("- A deteccao preventiva em baixa vigilancia pode exigir calibracao.")
    if processed_total == 0:
        observations.append("- Nenhum video foi processado com sucesso.")
    if not observations:
        observations.append("- Nenhuma observacao automatica critica foi encontrada.")

    report = f"""# Relatorio de Validacao

## Resumo

- Total de videos no manifest: {total}
- Total de videos processados: {processed_total}
- Videos com erro: {len(errors)}
- Acuracia basica: {accuracy:.1%}
- Falsos positivos: {false_positives}
- Falsos negativos: {false_negatives}
- Taxa de deteccao preventiva para low_vigilance: {preventive_rate:.1%}

## Quantidade por Dataset

{_counts_table(df["dataset"], "dataset")}

## Quantidade por Classe Esperada

{_counts_table(df["expected_class"], "expected_class")}

## Quantidade por Classe Prevista

{_counts_table(processed["predicted_class"], "predicted_class") if not processed.empty else "Nenhum video processado."}

## PERCLOS por Classe

{_markdown_table(perclos_by_class, ["expected_class", "mean", "max"], max_rows=len(perclos_by_class)) if not perclos_by_class.empty else "Sem dados."}

## Fatigue Score por Classe

{_markdown_table(fatigue_by_class, ["expected_class", "mean", "max"], max_rows=len(fatigue_by_class)) if not fatigue_by_class.empty else "Sem dados."}

## Melhores Videos

{_markdown_table(best, ["video_id", "dataset", "expected_class", "predicted_class", "max_fatigue_score", "first_drowsiness_time_sec", "is_correct_basic"], max_rows=8)}

## Piores Videos

{_markdown_table(worst, ["video_id", "dataset", "expected_class", "expected_event", "predicted_class", "first_low_vigilance_time_sec", "first_drowsiness_time_sec", "false_positive_flag", "false_negative_flag"], max_rows=8)}

## Videos com Erro

{_markdown_table(errors, ["video_id", "dataset", "expected_class", "processing_error"], max_rows=20)}

## Graficos

- `fatigue_score_by_class.png`
- `perclos_by_class.png`
- `confusion_matrix.png`
- `timeline_{{video_id}}.png` para cada video processado

{confusion_note}

## Observacoes Automaticas

{chr(10).join(observations)}
"""

    report_path = reports_dir / "validation_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Gera relatorio Markdown da validacao.")
    parser.add_argument("--summary", default="data/validation/outputs/validation_summary.csv")
    parser.add_argument("--outputs-dir", default="data/validation/outputs")
    parser.add_argument("--reports-dir", default="data/validation/reports")
    parser.add_argument("--plots-dir", default="data/validation/plots")
    args = parser.parse_args()

    report_path = generate_report(
        summary_csv=Path(args.summary),
        outputs_dir=Path(args.outputs_dir),
        reports_dir=Path(args.reports_dir),
        plots_dir=Path(args.plots_dir),
    )
    print(f"Relatorio salvo em: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
