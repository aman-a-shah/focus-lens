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


class PoseConfig(BaseModel):
    """MediaPipe PoseLandmarker (body-language sensing)."""

    enabled: bool = True
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Landmark visibility below this is treated as "not reliably seen" by body features.
    min_visibility: float = 0.5
    # Run pose every Nth frame (1 = every frame). Pose is the heaviest model; thinning it keeps
    # FPS up since posture/hands change far slower than the camera rate.
    every_n_frames: int = 1


class ActivityConfig(BaseModel):
    """Active-application context → activity category (privacy-sensitive; off-switchable)."""

    enabled: bool = True
    # Seconds between frontmost-app polls. App/window changes are coarse; 1s is plenty and keeps
    # the per-frame cost at zero (the reader is cached between polls).
    poll_interval_s: float = 1.0
    # App-name / window-title keyword → category. Lowercased substring match. User-editable.
    work_keywords: list[str] = Field(
        default_factory=lambda: [
            "code", "vscode", "visual studio", "pycharm", "intellij", "xcode", "sublime",
            "vim", "neovim", "emacs", "terminal", "iterm", "warp", "ghostty", "kitty",
            "word", "excel", "powerpoint", "keynote", "pages", "numbers", "notion", "obsidian",
            "docs.google", "sheets.google", "overleaf", "jupyter", "rstudio", "android studio",
            "figma", "blender", "photoshop", "illustrator", "premiere", "davinci",
        ]
    )
    communication_keywords: list[str] = Field(
        default_factory=lambda: [
            "slack", "discord", "messages", "imessage", "zoom", "teams", "meet.google",
            "telegram", "whatsapp", "signal", "mail", "outlook", "gmail",
        ]
    )
    social_keywords: list[str] = Field(
        default_factory=lambda: [
            "twitter", "x.com", "reddit", "instagram", "tiktok", "facebook", "threads",
            "snapchat", "linkedin", "pinterest", "tumblr", "9gag",
        ]
    )
    entertainment_keywords: list[str] = Field(
        default_factory=lambda: [
            "youtube", "netflix", "twitch", "hulu", "disney", "spotify", "primevideo",
            "hbo", "crunchyroll", "vlc",
        ]
    )
    gaming_keywords: list[str] = Field(
        default_factory=lambda: [
            "steam", "epic games", "league of legends", "valorant", "minecraft", "roblox",
            "fortnite", "dota", "csgo", "counter-strike", "battle.net", "riot", "game",
        ]
    )
    browser_keywords: list[str] = Field(
        default_factory=lambda: ["chrome", "safari", "firefox", "edge", "arc", "brave", "opera"]
    )


class VizConfig(BaseModel):
    draw_tesselation: bool = False
    draw_iris: bool = True
    show_fps: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    face_mesh: FaceMeshConfig = Field(default_factory=FaceMeshConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
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
