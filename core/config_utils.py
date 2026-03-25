"""default_config.yaml を読み、ResonantCore / AutopoieticInference 用 kwargs を返す。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def default_config_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "configs" / "default_config.yaml"


def load_yaml_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else default_config_path()
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resonant_core_kwargs(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_yaml_config()
    model = cfg.get("model", {})
    ap = cfg.get("autopoiesis", {})
    opt = cfg.get("optimization", {})
    return {
        "alpha": ap.get("alpha", 0.7),
        "beta": ap.get("beta", 0.2),
        "gamma": ap.get("gamma", 0.1),
        "tau": model.get("temperature", 1.0),
        "stability_bound": ap.get("stability_bound", 5.0),
        "drift_scale": 0.01,
        "initial_obscurity": opt.get("initial_obscurity", 0.15),
    }


def autopoietic_kwargs(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_yaml_config()
    ap = cfg.get("autopoiesis", {})
    return {
        "alpha": ap.get("alpha", 0.7),
        "beta": ap.get("beta", 0.2),
        "gamma": ap.get("gamma", 0.1),
        "stability_bound": ap.get("stability_bound", 5.0),
        "drift_scale": 0.01,
    }
