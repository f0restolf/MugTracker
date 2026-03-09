# ROCm Face Tracker
from .detector import FaceDetector, BoundingBox
from .tracker import SmoothTracker, VelocityTracker, TrackedPosition
from .cropper import FrameCropper, AdaptiveZoomCropper
from .output import VirtualCameraOutput

__version__ = "0.1.0"
__all__ = [
    "FaceDetector",
    "BoundingBox",
    "SmoothTracker",
    "VelocityTracker",
    "TrackedPosition",
    "FrameCropper",
    "AdaptiveZoomCropper",
    "VirtualCameraOutput",
]
