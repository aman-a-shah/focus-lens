"""Shared filesystem locations.

One source of truth for the gitignored ``checkpoints/`` directory, used by every trainer and
model loader so they agree on where artifacts live.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
