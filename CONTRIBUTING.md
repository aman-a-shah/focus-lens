# Contributing to FocusLens

Thanks for your interest! FocusLens is an on-device webcam attention-theft detector. This guide
covers local setup, conventions, and how the codebase is organized.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"      # core + torch + lint/test tooling
pre-commit install           # run ruff + black on every commit
```

Optional extras: `.[plot]` (matplotlib figures), `.[demo]` (Gradio app).

## Development loop

```bash
pytest                       # full suite (no camera needed — uses simulated frames)
ruff check focuslens tests   # lint
black focuslens tests        # format
```

CI / pre-commit run all three. Keep the suite green and the tree formatted before pushing.

- **Line length** 100, target Python 3.11+.
- **Style** match the surrounding code: module docstrings explaining *why*, dataclasses for
  records, pure functions where possible, type hints throughout.
- **No camera in tests.** Drive the pipeline with `focuslens.simulate.generate_frames` or the
  synthetic dataset generators. Anything needing a real webcam/display is a thin wrapper over a
  tested, headless core (see `app/`, `demo/`, `gaze/calibrate_session.py`).
- **Data never commits.** `data/`, `checkpoints/`, `*.pt`, `*.sqlite` are gitignored.

## Architecture

The system is a **walking skeleton**: webcam → features → state → notification ran end-to-end
before any model existed, then each box was replaced with real ML. See `roadmap.md` for the
phase-by-phase build and `README.md` for the diagram. Rough map:

| Area | Package | Notes |
|------|---------|-------|
| Capture + perception | `capture.py`, `face_mesh.py`, `features/` | OpenCV + MediaPipe FaceLandmarker |
| Walking skeleton | `window.py`, `classifier.py`, `pipeline.py`, `session.py`, `notify.py` | windowing → rule state → SQLite/notify |
| Gaze regression | `gaze/` | MLP/CNN + personal calibration |
| Learned classifier | `focusnet/` | PersonalFocusNet + continual learning (EWC/replay) |
| Intervention timing | `intervention/` | Cox proportional-hazards model |
| App + demo | `app/`, `demo/` | Tkinter control panel, Gradio demo |

## Pull requests

1. Branch from `main`.
2. Add tests for new behavior; keep `pytest` green.
3. Run `ruff` + `black`.
4. Describe the change and link the relevant roadmap phase.
