"""Hugging Face Spaces entry point (roadmap Phase 11).

Spaces runs ``app.py`` at the repo root. Install with the ``demo`` extra
(``pip install -e ".[demo]"``) and run ``python app.py`` locally, or let Spaces launch it.
"""

from focuslens.demo.app import launch

if __name__ == "__main__":
    launch()
