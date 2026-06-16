"""FocusLens desktop dashboard (Tkinter).

A polished, dark-themed control surface over ``AppController``:

- a live camera preview with the face/pose overlay baked in,
- a big color-coded status card (state + activity + the reason it fired),
- live signal meters (attention, gaze drift, looking-down, phone-in-hand),
- a rolling state timeline and a session breakdown,
- pause, a sensitivity slider, calibrate, and a summary view.

Capture runs on a worker thread feeding the controller and publishing the latest annotated
frame; the Tk main loop owns the UI and polls the controller's thread-safe snapshot. ``tkinter``
and ``cv2`` are imported lazily so importing this module never needs a display — the logic lives
in the (tested) controller.
"""

from __future__ import annotations

import base64
import threading
import time

from ..config import Config
from ..logging import get_logger
from ..states import DistractionState
from .controller import AppController

log = get_logger(__name__)

# Dark palette (hex; Tk wants hex strings).
_BG = "#0e1014"
_PANEL = "#171b22"
_PANEL_HI = "#1f242d"
_TEXT = "#e9eaee"
_MUTED = "#8b919c"
_TRACK = "#2a2f39"

_STATE_HEX = {
    "FOCUSED": "#34d399",
    "DRIFTING": "#f59e0b",
    "DISTRACTED": "#ef4444",
    "FATIGUED": "#a78bfa",
}
_STATE_LABEL = {
    "FOCUSED": "Focused",
    "DRIFTING": "Losing focus",
    "DISTRACTED": "Distracted",
    "FATIGUED": "Fatigued",
}


class _CaptureThread(threading.Thread):
    """Runs capture → Face Mesh + Pose → features → controller, publishing annotated frames."""

    def __init__(self, controller: AppController, source: int | str) -> None:
        super().__init__(daemon=True)
        self.controller = controller
        self.source = source
        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest = None  # most recent annotated BGR frame

    def stop(self) -> None:
        self._stop.set()

    def latest_frame(self):
        with self._frame_lock:
            return None if self._latest is None else self._latest.copy()

    def run(self) -> None:
        from contextlib import ExitStack

        from ..capture import WebcamCapture
        from ..context import ActiveAppReader, ActivityClassifier
        from ..face_mesh import FaceMeshTracker
        from ..features import FeatureExtractor
        from ..pose import PoseTracker
        from ..viz import draw_overlay

        cfg = self.controller.config
        extractor = FeatureExtractor(body_min_visibility=cfg.pose.min_visibility)
        pose_every_n = max(1, cfg.pose.every_n_frames)
        app_reader = ActiveAppReader(
            poll_interval_s=cfg.activity.poll_interval_s, enabled=cfg.activity.enabled
        )
        activity_classifier = ActivityClassifier(cfg.activity)
        fps = 0.0
        last_t = None
        try:
            with ExitStack() as stack:
                cap = stack.enter_context(
                    WebcamCapture(
                        source=self.source,
                        width=cfg.capture.width,
                        height=cfg.capture.height,
                        target_fps=cfg.capture.target_fps,
                    )
                )
                tracker = stack.enter_context(FaceMeshTracker(cfg.face_mesh))
                pose_tracker = (
                    stack.enter_context(PoseTracker(cfg.pose)) if cfg.pose.enabled else None
                )
                last_pose = None
                for n, frame in enumerate(cap.frames()):
                    if self._stop.is_set():
                        break
                    ts_ms = int(frame.timestamp * 1000)
                    result = tracker.process(frame.image, timestamp_ms=ts_ms)
                    if pose_tracker is not None and n % pose_every_n == 0:
                        last_pose = pose_tracker.process(frame.image, timestamp_ms=ts_ms)
                    features = extractor.extract(
                        result, frame.image.shape, frame.timestamp, image=frame.image,
                        pose=last_pose,
                    )
                    activity = activity_classifier.classify(app_reader.read(frame.timestamp))
                    self.controller.process_frame(features, activity)

                    if last_t is not None and frame.timestamp > last_t:
                        inst = 1.0 / (frame.timestamp - last_t)
                        fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
                    last_t = frame.timestamp

                    snap = self.controller.snapshot()
                    annotated = draw_overlay(
                        frame.image, result, fps, cfg.viz,
                        state=str(snap.state) if snap.state else None,
                        pose=last_pose, reason=snap.reason,
                        pose_min_visibility=cfg.pose.min_visibility,
                    )
                    with self._frame_lock:
                        self._latest = annotated
        except Exception as exc:  # never let the capture thread crash the UI
            log.error("capture thread stopped: %s", exc)


def run_app(
    config: Config | None = None,
    *,
    source: int | str = 0,
    db_path: str = "focuslens.sqlite",
    notify: bool = True,
) -> None:
    """Launch the FocusLens dashboard (needs a display + camera)."""
    import tkinter as tk
    from tkinter import messagebox, scrolledtext

    config = config or Config()
    controller = AppController(config=config, db_path=db_path, notify=notify)
    controller.start_session(time.time())
    capture = _CaptureThread(controller, source)
    capture.start()

    root = tk.Tk()
    root.title("FocusLens")
    root.configure(bg=_BG)
    root.geometry("900x560")
    root.minsize(820, 520)

    def _font(size, weight="normal"):
        return ("SF Pro Display", size, weight)  # falls back to the system default if absent

    # ---- header ------------------------------------------------------------------------------
    header = tk.Frame(root, bg=_BG)
    header.pack(fill="x", padx=18, pady=(14, 6))
    tk.Label(header, text="FocusLens", bg=_BG, fg=_TEXT, font=_font(20, "bold")).pack(side="left")
    meta = tk.Label(header, text="", bg=_BG, fg=_MUTED, font=_font(11))
    meta.pack(side="right")

    body = tk.Frame(root, bg=_BG)
    body.pack(fill="both", expand=True, padx=18, pady=6)

    # ---- left: camera preview ----------------------------------------------------------------
    left = tk.Frame(body, bg=_PANEL, highlightthickness=0)
    left.pack(side="left", fill="both", expand=True, padx=(0, 12))
    preview = tk.Label(left, bg="#000000", text="Starting camera…", fg=_MUTED, font=_font(12))
    preview.pack(fill="both", expand=True, padx=10, pady=10)

    # ---- right: status, meters, timeline, stats ----------------------------------------------
    right = tk.Frame(body, bg=_BG, width=320)
    right.pack(side="right", fill="y")
    right.pack_propagate(False)

    # Status card.
    card = tk.Frame(right, bg=_PANEL)
    card.pack(fill="x", pady=(0, 12))
    dot = tk.Canvas(card, width=18, height=18, bg=_PANEL, highlightthickness=0)
    dot_id = dot.create_oval(3, 3, 16, 16, fill=_MUTED, outline="")
    dot.grid(row=0, column=0, padx=(14, 8), pady=(14, 0), sticky="w")
    state_lbl = tk.Label(card, text="Watching…", bg=_PANEL, fg=_TEXT, font=_font(20, "bold"))
    state_lbl.grid(row=0, column=1, pady=(12, 0), sticky="w")
    activity_lbl = tk.Label(card, text="", bg=_PANEL, fg=_MUTED, font=_font(11))
    activity_lbl.grid(row=1, column=1, sticky="w")
    reason_lbl = tk.Label(
        card, text="", bg=_PANEL, fg=_TEXT, font=_font(12), wraplength=260, justify="left"
    )
    reason_lbl.grid(row=2, column=0, columnspan=2, padx=14, pady=(8, 14), sticky="w")
    card.grid_columnconfigure(1, weight=1)

    # Signal meters (drawn on a canvas).
    meters = tk.Canvas(right, bg=_PANEL, highlightthickness=0, height=140)
    meters.pack(fill="x", pady=(0, 12))
    _METERS = [
        ("Attention", "attention", "#34d399"),
        ("Gaze drift", "gaze_drift", "#f59e0b"),
        ("Looking down", "looking_down", "#60a5fa"),
        ("Phone in hand", "hands_near_face", "#ef4444"),
    ]

    # Live state timeline.
    timeline = tk.Canvas(right, bg=_PANEL, highlightthickness=0, height=54)
    timeline.pack(fill="x", pady=(0, 12))

    # Session breakdown.
    stats = tk.Canvas(right, bg=_PANEL, highlightthickness=0, height=70)
    stats.pack(fill="x")

    # ---- footer controls ---------------------------------------------------------------------
    footer = tk.Frame(root, bg=_BG)
    footer.pack(fill="x", padx=18, pady=12)

    def on_pause() -> None:
        paused = controller.toggle_pause()
        pause_btn.config(text="▶  Resume" if paused else "❚❚  Pause")

    def _btn(parent, text, cmd):
        return tk.Button(
            parent, text=text, command=cmd, bg=_PANEL_HI, fg=_TEXT, activebackground=_TRACK,
            activeforeground=_TEXT, relief="flat", font=_font(11), bd=0, padx=14, pady=6,
            highlightthickness=0, cursor="hand2",
        )

    pause_btn = _btn(footer, "❚❚  Pause", on_pause)
    pause_btn.pack(side="left")

    tk.Label(footer, text="Sensitivity", bg=_BG, fg=_MUTED, font=_font(10)).pack(
        side="left", padx=(16, 6)
    )
    slider = tk.Scale(
        footer, from_=0, to=100, orient="horizontal", length=160, showvalue=False,
        bg=_BG, fg=_TEXT, troughcolor=_TRACK, highlightthickness=0, bd=0,
        activebackground=_STATE_HEX["FOCUSED"],
        command=lambda v: controller.set_sensitivity(int(v) / 100.0),
    )
    slider.set(int(controller.settings.sensitivity * 100))
    slider.pack(side="left")

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
        win = tk.Toplevel(root, bg=_BG)
        win.title("Session summary")
        box = scrolledtext.ScrolledText(
            win, width=64, height=9, font=("Menlo", 11), bg=_PANEL, fg=_TEXT, relief="flat",
            insertbackground=_TEXT,
        )
        box.insert("1.0", summary.report() if summary else "No session data yet.")
        box.config(state="disabled")
        box.pack(padx=12, pady=12)

    _btn(footer, "Calibrate", on_calibrate).pack(side="right")
    _btn(footer, "Summary", on_summary).pack(side="right", padx=(0, 8))

    # ---- rendering helpers -------------------------------------------------------------------

    def _draw_meters(snap) -> None:
        meters.delete("all")
        width = meters.winfo_width() or 300
        x0, bar_w = 110, max(80, width - 124)
        for i, (label, attr, color) in enumerate(_METERS):
            y = 18 + i * 30
            value = max(0.0, min(1.0, float(getattr(snap, attr))))
            meters.create_text(14, y, text=label, anchor="w", fill=_MUTED, font=_font(10))
            meters.create_rectangle(x0, y - 6, x0 + bar_w, y + 6, fill=_TRACK, outline="")
            if value > 0:
                meters.create_rectangle(
                    x0, y - 6, x0 + int(bar_w * value), y + 6, fill=color, outline=""
                )

    def _draw_timeline() -> None:
        timeline.delete("all")
        recent = controller.recent_states()
        width = timeline.winfo_width() or 300
        timeline.create_text(8, 10, text="Last 30s", anchor="w", fill=_MUTED, font=_font(9))
        if not recent:
            return
        n = len(recent)
        cell = max(1.0, width / max(n, 60))
        for i, st in enumerate(recent):
            x = i * cell
            timeline.create_rectangle(
                x, 22, x + cell + 1, 50, fill=_STATE_HEX.get(str(st), _MUTED), outline=""
            )

    def _draw_stats() -> None:
        stats.delete("all")
        pct = controller.state_percentages()
        width = stats.winfo_width() or 300
        # Stacked proportion bar.
        x = 10
        bar_w = width - 20
        order = [DistractionState.FOCUSED, DistractionState.DRIFTING,
                 DistractionState.FATIGUED, DistractionState.DISTRACTED]
        stats.create_text(10, 12, text="This session", anchor="w", fill=_MUTED, font=_font(9))
        for st in order:
            frac = pct.get(str(st), 0.0)
            seg = bar_w * frac
            if seg > 0:
                stats.create_rectangle(
                    x, 26, x + seg, 42, fill=_STATE_HEX.get(str(st), _MUTED), outline=""
                )
                x += seg
        focused = pct.get(str(DistractionState.FOCUSED), 0.0)
        distracted = pct.get(str(DistractionState.DISTRACTED), 0.0)
        stats.create_text(
            10, 58, anchor="w", fill=_TEXT, font=_font(10),
            text=f"Focused {focused * 100:.0f}%    Distracted {distracted * 100:.0f}%",
        )

    def _render_preview() -> None:
        import cv2

        frame = capture.latest_frame()
        if frame is None:
            return
        avail_w = max(320, left.winfo_width() - 20)
        h, w = frame.shape[:2]
        scale = min(avail_w / w, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        try:
            ok, buf = cv2.imencode(".png", frame)
            if not ok:
                return
            data = base64.b64encode(buf.tobytes()).decode("ascii")
            photo = tk.PhotoImage(data=data)
            preview.configure(image=photo, text="")
            preview.image = photo  # keep a reference so Tk doesn't GC it
        except Exception:  # old Tk without PNG support — leave the placeholder text
            preview.configure(text="(camera preview unavailable)")

    # ---- main UI tick ------------------------------------------------------------------------

    def refresh() -> None:
        snap = controller.snapshot()
        state = str(snap.state) if snap.state else None
        color = _STATE_HEX.get(state, _MUTED)
        dot.itemconfig(dot_id, fill=color)
        state_lbl.config(text=_STATE_LABEL.get(state, "Watching…"), fg=color)

        sensors = []
        sensors.append("face ✓" if snap.face_present else "no face")
        if snap.body_present:
            sensors.append("body ✓")
        activity = str(snap.activity)
        act_txt = "" if activity in ("UNKNOWN", "IDLE") else activity.capitalize()
        activity_lbl.config(text="  ·  ".join(filter(None, [act_txt, " ".join(sensors)])))
        reason_lbl.config(text=snap.reason or ("On task." if state == "FOCUSED" else ""))
        meta.config(text=f"{snap.fps:.0f} FPS")

        _draw_meters(snap)
        _draw_timeline()
        _draw_stats()
        _render_preview()
        root.after(60, refresh)

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
