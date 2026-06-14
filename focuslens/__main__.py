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

    live = sub.add_parser("live", help="Run the live webcam Face Mesh overlay (Phase 1)")
    live.add_argument("--camera", type=int, default=None, help="Camera index override")
    live.add_argument("--source", default=None, help="Video file to replay instead of a camera")
    live.add_argument("--no-window", action="store_true", help="Run without the preview window")
    live.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    live.add_argument("--snapshot", default=None, help="Save the first annotated face frame here")
    live.add_argument("--features-csv", default=None, help="Stream per-frame features to this CSV")
    live.add_argument(
        "--print-features", action="store_true", help="Echo throttled feature lines to console"
    )

    bench = sub.add_parser("bench", help="Benchmark Face Mesh throughput offline (no camera)")
    bench.add_argument("--frames", type=int, default=120, help="Number of frames to time")
    bench.add_argument("--image", default=None, help="Face image to benchmark on (default: sample)")

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

    # No subcommand: print a short usage hint.
    _build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
