"""PersonalFocusNet: the learned attention classifier (roadmap Phase 7).

Replaces the Phase-3 rule classifier with a 1D-conv + temporal-self-attention sequence model
that carries an uncertainty head, trained on the self-supervised labels from Phase 6.
"""

from .classifier import LearnedClassifier
from .model import PersonalFocusNet, predict

__all__ = ["PersonalFocusNet", "predict", "LearnedClassifier"]
