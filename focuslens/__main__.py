"""FocusLens command-line entrypoint."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import load_config
from .logging import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="focuslens", description=__doc__)
    parser.add_argument("--version", action="version", version=f"focuslens {__version__}")
    parser.add_argument("--config", default=None, help="Path to a YAML config override")

    sub = parser.add_subparsers(dest="command")

    live = sub.add_parser("live", help="Run the live session (capture -> state -> notify -> log)")
    live.add_argument("--camera", type=int, default=None, help="Camera index override")
    live.add_argument("--source", default=None, help="Video file to replay instead of a camera")
    live.add_argument("--no-window", action="store_true", help="Run without the preview window")
    live.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    live.add_argument("--snapshot", default=None, help="Save the first annotated face frame here")
    live.add_argument("--features-csv", default=None, help="Stream per-frame features to this CSV")
    live.add_argument(
        "--print-features", action="store_true", help="Echo throttled feature lines to console"
    )
    live.add_argument("--db", default="focuslens.sqlite", help="SQLite session log path")
    live.add_argument("--no-db", action="store_true", help="Disable SQLite session logging")
    live.add_argument("--no-notify", action="store_true", help="Disable desktop notifications")

    bench = sub.add_parser("bench", help="Benchmark Face Mesh throughput offline (no camera)")
    bench.add_argument("--frames", type=int, default=120, help="Number of frames to time")
    bench.add_argument("--image", default=None, help="Face image to benchmark on (default: sample)")

    sim = sub.add_parser("simulate", help="Run a scripted session through the pipeline (no camera)")
    sim.add_argument(
        "--scenario",
        default=None,
        help="e.g. 'focused:4,drifting:4,distracted:6,fatigued:5' (default: built-in)",
    )
    sim.add_argument("--fps", type=int, default=30, help="Simulated frame rate")
    sim.add_argument("--seed", type=int, default=0, help="RNG seed")
    sim.add_argument("--db", default=":memory:", help="SQLite path (default: in-memory)")
    sim.add_argument("--notify", action="store_true", help="Actually fire desktop notifications")

    tg = sub.add_parser("train-gaze", help="Train the gaze regression model (synthetic by default)")
    tg.add_argument("--arch", choices=["mlp", "cnn"], default="mlp", help="Architecture")
    tg.add_argument("--epochs", type=int, default=15)
    tg.add_argument("--n-samples", type=int, default=4000, help="Synthetic dataset size")
    tg.add_argument("--batch-size", type=int, default=128)
    tg.add_argument("--lr", type=float, default=1e-3)
    tg.add_argument("--seed", type=int, default=0)
    tg.add_argument("--data", default=None, help="Path to a normalized MPIIFaceGaze .npz")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    setup_logging(config.logging.level)

    if args.command == "live":
        if args.camera is not None:
            config.capture.camera_index = args.camera
        from .runtime import run_live

        try:
            run_live(
                config,
                source=args.source,
                show_window=not args.no_window,
                max_frames=args.max_frames,
                snapshot_path=args.snapshot,
                features_csv=args.features_csv,
                print_features=args.print_features,
                db_path=None if args.no_db else args.db,
                notify=not args.no_notify,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "bench":
        from .runtime import run_benchmark

        try:
            result = run_benchmark(config, frames=args.frames, image_path=args.image)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(result.summary())
        return 0

    if args.command == "simulate":
        from .simulate import parse_scenario, run_simulation

        scenario = parse_scenario(args.scenario) if args.scenario else None
        summary = run_simulation(
            scenario=scenario, fps=args.fps, seed=args.seed, db_path=args.db, notify=args.notify
        )
        print(summary.report())
        return 0

    if args.command == "train-gaze":
        from .gaze.train import train_gaze

        dataset = None
        if args.data is not None:
            from .gaze.dataset import MPIIFaceGazeDataset

            try:
                dataset = MPIIFaceGazeDataset(args.data)
            except FileNotFoundError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
        result = train_gaze(
            arch=args.arch,
            dataset=dataset,
            n_samples=args.n_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
        print(result.summary())
        return 0

    # No subcommand: print a short usage hint.
    _build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
