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
- **Phase 5 — Personal calibration** ✅ on-screen 5/9-point routine → fine-tune the population
  gaze head into a per-user checkpoint (`calibrate`), swapped into the live pipeline
  (`live --gaze-model`); reflection-masking + low-light TTA for the iris signal
- **Phase 6 — Self-supervised labelling** ✅ retrospective "I drifted" marks (press `d`) + weak
  idle/app-switch events → label-propagation that back-dates distraction onset, stored in SQLite
  (`label`)
- **Phase 7 — PersonalFocusNet** ✅ 1D-conv + temporal-attention sequence model with an
  uncertainty head, trained on the Phase-6 labels (`train-focus`); swaps into the live pipeline
  behind the rule-classifier interface (`live --focus-model`) with cold-start + uncertainty gating
- **Phase 8 — Continual learning** ✅ EWC (per-task Fisher) + reservoir experience-replay buffer,
  post-session fine-tune with checkpoint rotation (`continual-update`); backward-transfer ablation
  (`continual-eval`) — over 10 sessions, **EWC+replay forgets 0.7% vs naive's 69%**

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
focuslens calibrate --user alice       # on-screen calibration -> per-user gaze checkpoint
focuslens live --gaze-model checkpoints/gaze_user_alice.pt           # use the calibrated head
focuslens label --db session.sqlite    # self-supervised labels from marks + weak signals
focuslens train-focus                  # train PersonalFocusNet (synthetic data by default)
focuslens train-focus --db session.sqlite                           # train on labelled sessions
focuslens live --focus-model checkpoints/focusnet.pt                # learned classifier in the loop
focuslens continual-eval --sessions 10 # EWC+replay vs naive vs frozen forgetting ablation
focuslens continual-update --db session.sqlite                      # post-session fine-tune (EWC+replay)
```

`simulate` runs synthetic behaviour through the **same pipeline** as `live` (windowing →
classification → notification → SQLite), so the end-to-end logic is verifiable without a
webcam. `train-gaze` defaults to a self-contained synthetic gaze dataset; for the real data
run `python scripts/download_mpiifacegaze.py` (manual registration required).

The learned-classifier workflow (Phases 6–7): run `live` (press `d` whenever you notice you
drifted), then `label` turns those marks plus weak idle/app-switch signals into per-window
labels — back-dating distraction *onset* before you consciously noticed it. `train-focus --db`
trains PersonalFocusNet on those labels; `live --focus-model` swaps it in behind the same
interface as the rule classifier. A cold-start window and an uncertainty gate keep a fresh model
("day 1 knows nothing") from firing until it's confident. `train-focus` with no `--db` trains on
synthetic data so the pipeline is runnable immediately.

Continual learning (Phase 8): `continual-update` fine-tunes the model on each new session while
**EWC** (per-task Fisher anchoring) and a reservoir **experience-replay** buffer guard against
forgetting earlier sessions; Fisher + buffer ride along in rotated per-session checkpoints.
`continual-eval` runs the EWC+replay vs naive vs frozen ablation over a sequence of synthetic
sessions and reports backward transfer — over 10 sessions EWC+replay forgets ~0.7% where naive
fine-tuning forgets ~69%.

Per-frame features (one row per frame): EAR (per eye + mean), eye-closed / blink rate /
last blink duration, head pose (yaw/pitch/roll in degrees), and a naive gaze proxy
(x/y iris offset, placeholder for the trained gaze head). Head-pose angles are uncalibrated
absolute estimates; the gaze proxy's neutral point is per-person and is recentred by
`calibrate` (Phase 5) — a ~2-min on-screen routine that fine-tunes the gaze head into a
per-user checkpoint; `live --gaze-model` then emits calibrated on-screen gaze in the same units.

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
