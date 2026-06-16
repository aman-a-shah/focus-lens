"""Frontmost-application reader (macOS).

Returns the name (and best-effort window title) of the app currently in the foreground. On
macOS this shells out to ``osascript`` — no extra dependency. The window title needs
Accessibility permission and is treated as optional: when it's unavailable we still get the
app name, which already separates an editor from a browser from a game.

Polling is throttled (``poll_interval_s``): foreground apps change on a human timescale, so we
cache the last reading and only re-poll occasionally, keeping the per-frame cost at zero. The
reader is a no-op (returns an unavailable ``AppContext``) when disabled or off macOS.
"""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from ..logging import get_logger

log = get_logger(__name__)

# One round-trip: print "<app name>|||<front window title>". Title errors (no window / no
# Accessibility permission) are swallowed so we still get the app name.
_OSASCRIPT = (
    'tell application "System Events"\n'
    "  set frontApp to first application process whose frontmost is true\n"
    "  set appName to name of frontApp\n"
    '  set winTitle to ""\n'
    "  try\n"
    "    set winTitle to name of first window of frontApp\n"
    "  end try\n"
    'end tell\n'
    'return appName & "|||" & winTitle'
)


@dataclass(frozen=True)
class AppContext:
    """The frontmost app and its window title. ``available`` is False when unknown/disabled."""

    app_name: str = ""
    window_title: str = ""
    available: bool = False

    @property
    def haystack(self) -> str:
        """Lowercased app name + window title for keyword matching."""
        return f"{self.app_name} {self.window_title}".lower().strip()


_UNAVAILABLE = AppContext()


def _osascript_runner() -> tuple[str, str] | None:
    """Query the macOS foreground app via osascript. Returns (app, title) or None on failure."""
    try:
        out = subprocess.run(
            ["osascript", "-e", _OSASCRIPT],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        log.debug("osascript failed: %s", exc)
        return None
    if out.returncode != 0:
        log.debug("osascript returned %d: %s", out.returncode, out.stderr.strip())
        return None
    app, _, title = out.stdout.strip().partition("|||")
    return app.strip(), title.strip()


class ActiveAppReader:
    """Throttled reader of the foreground application.

    ``runner`` overrides the platform query (used in tests); when omitted, the macOS osascript
    backend is used and any other platform yields an unavailable context.
    """

    def __init__(
        self,
        poll_interval_s: float = 1.0,
        enabled: bool = True,
        runner: Callable[[], tuple[str, str] | None] | None = None,
    ) -> None:
        self.poll_interval_s = poll_interval_s
        self.enabled = enabled
        if runner is not None:
            self._runner: Callable[[], tuple[str, str] | None] | None = runner
        elif platform.system() == "Darwin":
            self._runner = _osascript_runner
        else:
            self._runner = None  # unsupported platform — stays unavailable
        self._cached = _UNAVAILABLE
        self._last_poll: float | None = None

    def read(self, now: float) -> AppContext:
        """Return the foreground app context as of time ``now`` (frame timestamp, seconds)."""
        if not self.enabled or self._runner is None:
            return _UNAVAILABLE
        if self._last_poll is not None and (now - self._last_poll) < self.poll_interval_s:
            return self._cached
        self._last_poll = now
        result = self._runner()
        if result is not None:
            app, title = result
            self._cached = AppContext(app_name=app, window_title=title, available=bool(app))
        return self._cached
