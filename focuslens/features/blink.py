"""Blink detection via an EAR threshold state machine.

Two states (OPEN / CLOSED) with hysteresis: the eye must drop below ``ear_close`` to start a
blink and rise back above ``ear_open`` to end it. Each completed blink records its duration
and timestamp; a rolling window yields a blinks-per-minute rate. State is per-user/per-frame
and lives across calls, so one detector instance per session.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class BlinkState:
    """Snapshot returned per frame."""

    eye_closed: bool
    blinks_per_min: float
    last_blink_duration_s: float
    blink_completed: bool  # True on the exact frame a blink finished


class BlinkDetector:
    def __init__(
        self,
        ear_close: float = 0.21,
        ear_open: float = 0.24,
        rate_window_s: float = 60.0,
    ) -> None:
        if ear_close > ear_open:
            raise ValueError("ear_close must be <= ear_open (hysteresis)")
        self.ear_close = ear_close
        self.ear_open = ear_open
        self.rate_window_s = rate_window_s
        self._closed = False
        self._close_start: float | None = None
        self._blink_times: deque[float] = deque()
        self.last_blink_duration_s = 0.0

    @property
    def is_closed(self) -> bool:
        return self._closed

    def blinks_per_min(self, now: float) -> float:
        self._trim(now)
        if not self._blink_times:
            return 0.0
        return len(self._blink_times) * 60.0 / self.rate_window_s

    def _trim(self, now: float) -> None:
        while self._blink_times and now - self._blink_times[0] > self.rate_window_s:
            self._blink_times.popleft()

    def update(self, ear: float, timestamp: float) -> BlinkState:
        """Advance the state machine with this frame's mean EAR."""
        completed = False
        if not self._closed:
            if ear < self.ear_close:
                self._closed = True
                self._close_start = timestamp
        else:
            if ear > self.ear_open:
                self._closed = False
                if self._close_start is not None:
                    self.last_blink_duration_s = timestamp - self._close_start
                self._blink_times.append(timestamp)
                completed = True

        return BlinkState(
            eye_closed=self._closed,
            blinks_per_min=self.blinks_per_min(timestamp),
            last_blink_duration_s=self.last_blink_duration_s,
            blink_completed=completed,
        )
