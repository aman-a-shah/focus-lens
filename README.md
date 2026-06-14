# FocusLens

On-device webcam **attention-theft detector**. Watches you through your laptop camera,
learns your personal distraction signature, and intervenes in real time when you lose focus.
Runs fully offline — no cloud, no data leaving your machine.

See [`plan.md`](plan.md) for the full PRD and [`roadmap.md`](roadmap.md) for the phased build plan.

## Status

Early scaffold. Current phases:

- **Phase 0 — Scaffold** ✅ repo, deps, config, logging, tests
- **Phase 1 — Webcam + Face Mesh spike** ✅ live landmark/iris overlay with FPS

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
focuslens --version          # print version
focuslens live               # live webcam overlay (Face Mesh landmarks + iris + FPS)
focuslens live --camera 1    # pick a different camera index
```

Press `q` or `Esc` in the overlay window to quit. On first run the MediaPipe
`face_landmarker.task` model (~3.6 MB) is auto-downloaded into `checkpoints/` (gitignored).

> Uses MediaPipe's Tasks API (`FaceLandmarker`); the legacy `solutions.face_mesh` module is
> absent in recent MediaPipe builds. The landmarker emits 478 landmarks including iris.

## Develop

```bash
pytest          # run tests
ruff check .    # lint
black .         # format
```
