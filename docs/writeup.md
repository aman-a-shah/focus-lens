# FocusLens: building an on-device attention-theft detector, one box at a time

FocusLens watches you through your webcam and learns *your* personal signature of losing focus —
then nudges you ~20 seconds before you'd consciously notice you've drifted. Everything runs
locally; no frame ever leaves the device.

This is a build log of how it was assembled, with the numbers that mattered.

## The core idea: a walking skeleton

The temptation in an ML project is to start with the model. FocusLens started with the
*integration*: webcam → features → a dumb rule-based state → a desktop notification, running
end-to-end before any model existed. That kills integration risk early and means there is always
a demoable artifact. Each subsequent phase replaces exactly one box in that skeleton with real ML,
against a pipeline that already runs.

```
capture → FaceMesh → features → window → classifier → notify
            (P1)       (P2)       (P3)    (P3→P7)      (P3→P9)
                                   gaze proxy → trained + calibrated (P4–P5)
```

## Perception (Phases 1–2)

MediaPipe FaceLandmarker gives 478 landmarks + iris at **~104 FPS / 8.9 ms p50** on Apple Silicon
CPU — comfortably real-time. From the landmarks we derive deterministic per-frame signals: eye
aspect ratio, a blink state machine, head pose via `solvePnP`, and a naive iris-in-socket gaze
proxy. No ML yet — but the whole downstream pipeline can be developed against this stream.

## Gaze regression + calibration (Phases 4–5)

The naive proxy is replaced by a small regressor (an MLP on geometric features; a CNN on the eye
crop as a bake-off). On synthetic data it hits **1.2–1.8° mean angular error at <0.5 ms/frame**.

The interesting result is **calibration**. Everyone's eye geometry differs, so the population head
reads a frontal face as off-centre. A ~2-minute on-screen routine (5/9 points) collects labelled
samples and fine-tunes the head per user. On a synthetic user whose iris→gaze mapping is shifted
from the population, the *pretrained* head's error is **24.6°**; after fine-tuning on the
calibration samples it drops to **0.9°** — far past the 20–40% target. The calibrated head is a
drop-in for the proxy in the same coordinate convention, so nothing downstream changes.

## Self-supervised labels + the learned classifier (Phases 6–7)

You can't hand-label "I lost focus." FocusLens manufactures labels: press a key the moment you
notice you've drifted, and label-propagation back-dates the distraction *onset* ~20 s before the
keypress (you notice late); weak idle/app-switch signals add more; the rule classifier's output is
a weak prior for everything else.

On those labels, **PersonalFocusNet** — a 1D-conv encoder → temporal self-attention → classifier
head **+ an uncertainty head** — replaces the heuristic. It carries a cold-start + uncertainty
gate so a fresh model ("day 1 knows nothing") doesn't fire until it's confident. On the synthetic
labelled set it reaches **DISTRACTED precision/recall = 1.00/1.00**, and swaps into the live
pipeline behind the same `classify(window)` interface as the rule classifier.

## Continual learning: the headline result (Phase 8)

A model that fine-tunes on each new session **catastrophically forgets** earlier ones. The
ablation makes this stark. Over **10 sequential sessions** (each a rotation of the feature space,
so the decision boundary genuinely conflicts):

| Strategy | Backward degradation | Final avg accuracy |
|----------|---------------------:|-------------------:|
| Naive fine-tune | **69.0%** | 37.9% |
| Frozen after session 0 | 0.0% | 42.0% |
| **EWC + experience replay** | **0.7%** | **99.4%** |

EWC (per-session diagonal Fisher anchoring) plus a reservoir experience-replay buffer (30%
mix-in) cuts forgetting from catastrophic to negligible — well under the <5% target — while
frozen weights show the no-learning floor. Fisher and the replay buffer are serialized into
rotated per-session checkpoints, so the defence survives across the real post-session job.

## Intervention timing (Phase 9)

Knowing you're *distracted* isn't enough — you want a nudge *before* you drift. A Cox
proportional-hazards model scores per-moment distraction risk from trend features (60 s distraction
fraction, gaze velocity, blink rate, head motion, time-of-day, session length), fit by Breslow
partial likelihood — no `lifelines` dependency. It reaches a **concordance index of 0.83**, and a
user-tunable hazard threshold (calibrated to a target lead) fires the nudge a **median of ~22 s
before drift**, inside the 10–30 s target window. Each intervention logs a "was this helpful?"
slot for later tuning.

## What I'd tell the next person

- **Skeleton first.** Having a runnable, demoable pipeline from week one changed how every later
  decision was made — each model was validated against reality, not a notebook.
- **Smooth your losses.** The gaze loss used `1 − cos` rather than `arccos`; the latter has
  unbounded gradient as predictions approach targets and reliably NaN'd. There's a regression test
  guarding it.
- **Replay is the cheap win.** EWC helps, but the experience-replay buffer did most of the work
  against forgetting and is trivial to implement.
- **Everything is testable without a camera.** Simulated frame streams and synthetic survival /
  rotated-task generators drive the full suite (135 tests), so the GUI and webcam are thin wrappers
  over a tested core.

All numbers above are reproducible from the CLI: `train-gaze`, `calibrate`, `train-focus`,
`continual-eval`, `train-timing`.
