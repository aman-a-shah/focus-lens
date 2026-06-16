"""Tkinter control-panel shell (roadmap Phase 10).

A daily-driver surface over ``AppController``: a status light for the current attention state, a
pause/resume toggle, a sensitivity slider, a "Calibrate" launcher, and a "Show summary" button
that renders the post-session distraction heatmap. Capture runs on a background thread feeding the
controller; the Tk main loop owns the UI. ``tkinter``/``cv2`` are imported lazily inside ``run`` so
importing this module never needs a display — the logic lives in the (tested) controller.
"""

from __future__ import annotations

import threading
import time

from ..config import Config
from ..logging import get_logger
from .controller import AppController

log = get_logger(__name__)


class _CaptureThread(threading.Thread):
    """Runs capture → Face Mesh → features → controller on a worker thread."""

    def __init__(self, controller: AppController, source: int | str) -> None:
        super().__init__(daemon=True)
        self.controller = controller
        self.source = source
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        from ..capture import WebcamCapture
        from ..face_mesh import FaceMeshTracker
        from ..features import FeatureExtractor

        cfg = self.controller.config
        extractor = FeatureExtractor()
        try:
            with (
                WebcamCapture(
                    source=self.source,
                    width=cfg.capture.width,
                    height=cfg.capture.height,
                    target_fps=cfg.capture.target_fps,
                ) as cap,
                FaceMeshTracker(cfg.face_mesh) as tracker,
            ):
                for frame in cap.frames():
                    if self._stop.is_set():
                        break
                    result = tracker.process(frame.image, timestamp_ms=int(frame.timestamp * 1000))
                    features = extractor.extract(
                        result, frame.image.shape, frame.timestamp, image=frame.image
                    )
                    self.controller.process_frame(features)
        except Exception as exc:  # never let the capture thread crash the UI
            log.error("capture thread stopped: %s", exc)


def run_app(
    config: Config | None = None,
    *,
    source: int | str = 0,
    db_path: str = "focuslens.sqlite",
    notify: bool = True,
) -> None:
    """Launch the Tkinter control panel (needs a display + camera)."""
    import tkinter as tk
    from tkinter import messagebox, scrolledtext

    config = config or Config()
    controller = AppController(config=config, db_path=db_path, notify=notify)
    controller.start_session(time.time())
    capture = _CaptureThread(controller, source)
    capture.start()

    root = tk.Tk()
    root.title("FocusLens")
    root.geometry("360x230")

    status = tk.Label(root, text="Starting…", font=("Helvetica", 16))
    status.pack(pady=10)

    def on_pause() -> None:
        paused = controller.toggle_pause()
        pause_btn.config(text="Resume" if paused else "Pause")

    pause_btn = tk.Button(root, text="Pause", width=12, command=on_pause)
    pause_btn.pack()

    tk.Label(root, text="Sensitivity").pack(pady=(10, 0))
    slider = tk.Scale(
        root,
        from_=0,
        to=100,
        orient=tk.HORIZONTAL,
        length=240,
        command=lambda v: controller.set_sensitivity(int(v) / 100.0),
    )
    slider.set(int(controller.settings.sensitivity * 100))
    slider.pack()

    def on_calibrate() -> None:
        from ..gaze.calibrate_session import run_calibration

        def worker() -> None:
            try:
                run_calibration(config, source=source)
            except Exception as exc:  # noqa: BLE001 - surface to the user, don't crash
                log.error("calibration failed: %s", exc)

        threading.Thread(target=worker, daemon=True).start()

    def on_summary() -> None:
        summary = controller.summary()
        win = tk.Toplevel(root)
        win.title("Session summary")
        box = scrolledtext.ScrolledText(win, width=60, height=8, font=("Courier", 11))
        box.insert("1.0", summary.report() if summary else "No session data yet.")
        box.config(state="disabled")
        box.pack()

    btns = tk.Frame(root)
    btns.pack(pady=12)
    tk.Button(btns, text="Calibrate", width=12, command=on_calibrate).grid(row=0, column=0, padx=4)
    tk.Button(btns, text="Summary", width=12, command=on_summary).grid(row=0, column=1, padx=4)

    def refresh() -> None:
        state = controller.current_state
        status.config(text=str(state) if state else "Watching…")
        root.after(300, refresh)

    def on_close() -> None:
        capture.stop()
        controller.end_session(time.time())
        summary = controller.summary()
        if summary is not None:
            messagebox.showinfo("Session summary", summary.report())
        controller.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh()
    root.mainloop()
