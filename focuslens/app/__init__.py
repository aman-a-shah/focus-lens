"""App shell & UX (roadmap Phase 10).

Wraps the runtime in a daily-driver control surface: a settings model (sensitivity slider,
pause/resume), an ``AppController`` that drives the pipeline with those settings, a post-session
distraction-heatmap summary, and a thin Tkinter shell. The GUI is lazily imported so the headless
core (settings/controller/summary) is fully testable without a display.
"""

from .controller import AppController
from .settings import AppSettings, sensitivity_to_thresholds
from .summary import SessionSummary, build_summary

__all__ = [
    "AppController",
    "AppSettings",
    "sensitivity_to_thresholds",
    "SessionSummary",
    "build_summary",
]
