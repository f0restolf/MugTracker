"""
Smooth Tracking Module - EMA filtering with deadzone
"""
from dataclasses import dataclass
from typing import Optional
from detector import BoundingBox


@dataclass
class TrackedPosition:
    """Smoothed tracking position"""
    cx: float      # center x
    cy: float      # center y
    width: float   # face width
    height: float  # face height
    
    @classmethod
    def from_bbox(cls, bbox: BoundingBox) -> "TrackedPosition":
        return cls(
            cx=(bbox.x1 + bbox.x2) / 2,
            cy=(bbox.y1 + bbox.y2) / 2,
            width=bbox.width,
            height=bbox.height
        )


class SmoothTracker:
    """
    Smooth face tracking with exponential moving average.
    
    Features:
    - Deadzone to prevent micro-jitter
    - Smooth transitions using EMA (lerp)
    - Holds last position when face temporarily lost
    - Signals zoom-out after extended loss
    """
    
    def __init__(
        self,
        smoothing_factor: float = 0.15,
        deadzone: float = 20.0,
        size_smoothing: float = 0.1,
        max_frames_without_detection: int = 30
    ):
        """
        Args:
            smoothing_factor: 0.1 (smooth) to 0.3 (responsive) for position
            deadzone: Minimum pixel movement to trigger update
            size_smoothing: Smoothing for face size changes (slower = less zoom jitter)
            max_frames_without_detection: Frames before signaling "no face"
        """
        self.smoothing_factor = smoothing_factor
        self.deadzone = deadzone
        self.size_smoothing = size_smoothing
        self.max_frames_without_detection = max_frames_without_detection
        
        self.current: Optional[TrackedPosition] = None
        self.frames_without_detection = 0
    
    def update(self, bbox: Optional[BoundingBox]) -> Optional[TrackedPosition]:
        """
        Update tracking state with new detection.
        
        Args:
            bbox: Detected face bounding box, or None if no face found
            
        Returns:
            Current tracked position, or None if should zoom out to full frame
        """
        if bbox is None:
            self.frames_without_detection += 1
            if self.frames_without_detection > self.max_frames_without_detection:
                return None  # Signal: zoom out to full frame
            return self.current  # Hold last known position
        
        # Got a detection
        self.frames_without_detection = 0
        new_pos = TrackedPosition.from_bbox(bbox)
        
        # First detection - snap to position
        if self.current is None:
            self.current = new_pos
            return self.current
        
        # Check deadzone (only for position, not size)
        dx = abs(new_pos.cx - self.current.cx)
        dy = abs(new_pos.cy - self.current.cy)
        
        in_deadzone = dx < self.deadzone and dy < self.deadzone
        
        # Apply smoothing
        alpha = self.smoothing_factor
        size_alpha = self.size_smoothing
        
        if in_deadzone:
            # Only smooth size, keep position locked
            self.current = TrackedPosition(
                cx=self.current.cx,
                cy=self.current.cy,
                width=self._lerp(self.current.width, new_pos.width, size_alpha),
                height=self._lerp(self.current.height, new_pos.height, size_alpha)
            )
        else:
            # Smooth both position and size
            self.current = TrackedPosition(
                cx=self._lerp(self.current.cx, new_pos.cx, alpha),
                cy=self._lerp(self.current.cy, new_pos.cy, alpha),
                width=self._lerp(self.current.width, new_pos.width, size_alpha),
                height=self._lerp(self.current.height, new_pos.height, size_alpha)
            )
        
        return self.current
    
    def reset(self):
        """Reset tracker state"""
        self.current = None
        self.frames_without_detection = 0
    
    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation"""
        return a + t * (b - a)


class VelocityTracker(SmoothTracker):
    """
    Extended tracker with velocity prediction for reduced lag.
    Use if default SmoothTracker feels sluggish during fast movement.
    """
    
    def __init__(self, prediction_frames: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.prediction_frames = prediction_frames
        self.velocity_cx = 0.0
        self.velocity_cy = 0.0
        self.prev_pos: Optional[TrackedPosition] = None
    
    def update(self, bbox: Optional[BoundingBox]) -> Optional[TrackedPosition]:
        result = super().update(bbox)
        
        if result is None or self.prev_pos is None:
            self.prev_pos = result
            return result
        
        # Calculate velocity
        self.velocity_cx = 0.8 * self.velocity_cx + 0.2 * (result.cx - self.prev_pos.cx)
        self.velocity_cy = 0.8 * self.velocity_cy + 0.2 * (result.cy - self.prev_pos.cy)
        
        # Apply prediction
        predicted = TrackedPosition(
            cx=result.cx + self.velocity_cx * self.prediction_frames,
            cy=result.cy + self.velocity_cy * self.prediction_frames,
            width=result.width,
            height=result.height
        )
        
        self.prev_pos = result
        return predicted
