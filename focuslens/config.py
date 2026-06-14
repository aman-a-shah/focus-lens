"""Configuration model and loader.

Config is a small typed tree (pydantic) loaded from YAML. ``load_config`` reads the
bundled ``configs/default.yaml`` and deep-merges an optional user override on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class CaptureConfig(BaseModel):
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    buffer_seconds: float = 5.0


class FaceMeshConfig(BaseModel):
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class VizConfig(BaseModel):
    draw_tesselation: bool = False
    draw_iris: bool = True
    show_fps: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    face_mesh: FaceMeshConfig = Field(default_factory=FaceMeshConfig)
    viz: VizConfig = Field(default_factory=VizConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins for scalars)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(override_path: str | Path | None = None) -> Config:
    """Load defaults, then deep-merge an optional user override file on top."""
    data = _read_yaml(_DEFAULT_CONFIG_PATH)
    if override_path is not None:
        data = _deep_merge(data, _read_yaml(Path(override_path)))
    return Config.model_validate(data)
