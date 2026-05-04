"""Funcoes utilitarias do projeto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Carrega o YAML de configuracao."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    return data


def ensure_dir(path: str | Path) -> Path:
    """Cria um diretorio se ele ainda nao existir."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def parse_camera_source(source: Any) -> int | str:
    """Converte fonte numerica de camera para int, mantendo caminhos como string."""
    if isinstance(source, int):
        return source

    if source is None:
        return 0

    source_text = str(source).strip()
    if source_text.isdigit():
        return int(source_text)

    return source_text


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    """Limita um valor numerico a um intervalo."""
    return max(minimum, min(maximum, value))


def level_from_score(score: float) -> str:
    """Classifica um score de risco de 0 a 100."""
    if score <= 30:
        return "NORMAL"
    if score <= 60:
        return "ATENCAO"
    if score <= 80:
        return "RISCO_MODERADO"
    return "RISCO_ALTO"
