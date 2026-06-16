"""Gradio demo app for Hugging Face Spaces (roadmap Phase 11).

A thin UI over ``Analyzer``: upload or webcam-snap a face, get back the annotated overlay
(landmarks + iris + gaze + a snapshot attention state) and a caption. ``gradio`` is imported
lazily so the package installs/imports without it; only ``launch`` needs it.

Images are converted to BGR, analyzed in-memory, and discarded — nothing is stored.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .inference import Analyzer

_DESCRIPTION = (
    "# FocusLens\n"
    "On-device webcam attention analysis. Upload or snap a face — FocusLens overlays the "
    "MediaPipe landmarks, iris, head pose and gaze, and reads a snapshot attention state. "
    "Images are processed in-memory and never stored."
)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """Gradio hands us RGB (H, W, 3); the perception stack expects BGR."""
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, ::-1].copy()
    return image


def build_demo(config: Config | None = None):
    """Construct the Gradio Blocks interface (requires ``gradio``)."""
    import gradio as gr

    analyzer = Analyzer(config or Config())

    def run(image: np.ndarray | None):
        if image is None:
            return None, "Provide an image."
        result = analyzer.analyze(_to_bgr(image))
        rgb = result.annotated[:, :, ::-1]  # back to RGB for display
        return rgb, result.caption()

    with gr.Blocks(title="FocusLens") as demo:
        gr.Markdown(_DESCRIPTION)
        with gr.Row():
            inp = gr.Image(sources=["upload", "webcam"], type="numpy", label="Input")
            out = gr.Image(label="Analysis")
        caption = gr.Textbox(label="Readout", interactive=False)
        gr.Button("Analyze").click(run, inputs=inp, outputs=[out, caption])
        inp.change(run, inputs=inp, outputs=[out, caption])
    return demo


def launch(config: Config | None = None, **launch_kwargs: object) -> None:
    """Build and launch the Gradio demo."""
    build_demo(config).launch(**launch_kwargs)
