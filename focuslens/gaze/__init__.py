"""Gaze regression + personal calibration: data, models, metrics, training (roadmap Phase 4–5)."""

from .calibration import (
    CALIB_POINTS_5,
    CALIB_POINTS_9,
    CalibrationData,
    CalibrationResult,
    calibrate_user,
    fine_tune,
)
from .metrics import angular_error_deg, mean_angular_error_deg
from .model import MLPGazeNet, SmallGazeCNN, build_model
from .predictor import CalibratedGazePredictor, NaiveGazePredictor, features_from

__all__ = [
    "angular_error_deg",
    "mean_angular_error_deg",
    "MLPGazeNet",
    "SmallGazeCNN",
    "build_model",
    "CALIB_POINTS_5",
    "CALIB_POINTS_9",
    "CalibrationData",
    "CalibrationResult",
    "calibrate_user",
    "fine_tune",
    "CalibratedGazePredictor",
    "NaiveGazePredictor",
    "features_from",
]
