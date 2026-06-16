"""Public demo (roadmap Phase 11).

A self-contained inference entry point — ``Analyzer`` runs the full perception stack on a single
image (Face Mesh → features → state) and returns an annotated overlay, for a Hugging Face Spaces
Gradio demo. Frames are processed in-memory and never stored (anonymized inference).
"""

from .inference import Analyzer, DemoResult

__all__ = ["Analyzer", "DemoResult"]
