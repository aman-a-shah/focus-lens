# FocusLens

On-device webcam **attention-theft detector**. Watches you through your laptop camera,
learns your personal distraction signature, and intervenes in real time when you lose focus.
Runs fully offline — no cloud, no data leaving your machine.

See [`plan.md`](plan.md) for the full PRD and [`roadmap.md`](roadmap.md) for the phased build plan.

## Status

Early scaffold. Current phases:

- **Phase 0 — Scaffold** ✅ repo, deps, config, logging, tests
- **Phase 1 — Webcam + Face Mesh spike** ✅ live landmark/iris overlay with FPS
- **Phase 2 — Feature extractors** ✅ EAR, blink detection, head pose, naive gaze → per-frame
  feature stream (console + CSV)
- **Phase 3 — Walking skeleton** ✅ 200ms windowing → rule classifier (FOCUSED/DRIFTING/
  DISTRACTED/FATIGUED) → debounce → desktop notifications → SQLite session log; offline
  `simulate` driver
- **Phase 4 — Gaze regression pipeline** ✅ synthetic + MPIIFaceGaze datasets, MLP & CNN
  architectures, angular-error eval, checkpointing, latency (`train-gaze`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
focuslens --version                    # print version
focuslens live                         # live webcam overlay (landmarks + iris + FPS)
focuslens live --camera 1              # pick a different camera index
focuslens live --source clip.mp4       # replay a video file instead of the camera
focuslens live --no-window --snapshot out.png --max-frames 30   # headless, save one frame
focuslens live --features-csv feats.csv --print-features        # stream per-frame features
focuslens live --db session.sqlite --no-notify                  # log session, no notifications
focuslens bench --frames 150           # measure perception throughput (no camera)
focuslens simulate                     # scripted session through the pipeline (no camera)
focuslens simulate --scenario 'focused:5,distracted:6,fatigued:5' --notify
focuslens train-gaze --arch mlp        # train gaze model on synthetic data
focuslens train-gaze --arch cnn --epochs 15
focuslens train-gaze --arch cnn --data checkpoints/mpiifacegaze.npz   # real dataset
```

`simulate` runs synthetic behaviour through the **same pipeline** as `live` (windowing →
classification → notification → SQLite), so the end-to-end logic is verifiable without a
webcam. `train-gaze` defaults to a self-contained synthetic gaze dataset; for the real data
run `python scripts/download_mpiifacegaze.py` (manual registration required).

Per-frame features (one row per frame): EAR (per eye + mean), eye-closed / blink rate /
last blink duration, head pose (yaw/pitch/roll in degrees), and a naive gaze proxy
(x/y iris offset, placeholder for the trained gaze head). Head-pose angles are uncalibrated
absolute estimates; the gaze proxy's neutral point is per-person and gets removed by
calibration in Phase 5.

Press `q` or `Esc` in the overlay window to quit. On first run the MediaPipe
`face_landmarker.task` model (~3.6 MB) is auto-downloaded into `checkpoints/` (gitignored).

`bench` reports mean FPS and p50/p95 latency for the Face Mesh stage. Reference run
(Apple Silicon, CPU): **~104 FPS mean, 8.9 ms p50** — comfortably past Phase 1's ≥20 FPS bar.

> Uses MediaPipe's Tasks API (`FaceLandmarker`); the legacy `solutions.face_mesh` module is
> absent in recent MediaPipe builds. The landmarker emits 478 landmarks including iris.

## Develop

```bash
pytest          # run tests
ruff check .    # lint
black .         # format
```
