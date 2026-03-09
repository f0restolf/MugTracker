"""
Virtual Camera Output Module - pyvirtualcam wrapper
"""
import pyvirtualcam
import numpy as np
from typing import Tuple, Optional


class VirtualCameraOutput:
    """
    Virtual camera output using pyvirtualcam and v4l2loopback.
    
    Before using, ensure v4l2loopback is loaded:
        sudo modprobe v4l2loopback devices=1 video_nr=10 \\
            card_label="FaceTrack" exclusive_caps=1 max_buffers=2
    """
    
    def __init__(
        self,
        device: str = "/dev/video10",
        size: Tuple[int, int] = (1920, 1080),
        fps: int = 30
    ):
        """
        Args:
            device: v4l2loopback device path
            size: Output resolution (width, height)
            fps: Target framerate
        """
        self.device = device
        self.size = size
        self.fps = fps
        self.cam: Optional[pyvirtualcam.Camera] = None
        self._frame_count = 0
    
    def __enter__(self):
        """Context manager entry - open virtual camera"""
        try:
            self.cam = pyvirtualcam.Camera(
                width=self.size[0],
                height=self.size[1],
                fps=self.fps,
                device=self.device
            )
            print(f"✓ Virtual camera started: {self.cam.device}")
            print(f"  Resolution: {self.size[0]}x{self.size[1]} @ {self.fps}fps")
        except Exception as e:
            print(f"✗ Failed to open virtual camera: {e}")
            print("\nTroubleshooting:")
            print("  1. Ensure v4l2loopback is loaded:")
            print(f"     sudo modprobe v4l2loopback devices=1 video_nr=10 card_label='FaceTrack' exclusive_caps=1 max_buffers=2")
            print(f"  2. Check device exists: ls -la {self.device}")
            print("  3. Ensure no other process is using the device")
            raise
        return self
    
    def __exit__(self, *args):
        """Context manager exit - close virtual camera"""
        if self.cam:
            self.cam.close()
            print(f"\n✓ Virtual camera closed after {self._frame_count} frames")
    
    def write_frame(self, frame: np.ndarray):
        """
        Write RGB frame to virtual camera.
        
        Args:
            frame: RGB frame (not BGR!) matching output size
        """
        if self.cam is None:
            raise RuntimeError("Virtual camera not initialized. Use with context manager.")
        
        # Validate frame
        expected_shape = (self.size[1], self.size[0], 3)
        if frame.shape != expected_shape:
            raise ValueError(f"Frame shape {frame.shape} doesn't match expected {expected_shape}")
        
        self.cam.send(frame)
        self.cam.sleep_until_next_frame()
        self._frame_count += 1
    
    def write_frame_async(self, frame: np.ndarray):
        """
        Write frame without sleeping (for manual timing control).
        
        Use this when you have your own frame pacing logic.
        """
        if self.cam is None:
            raise RuntimeError("Virtual camera not initialized")
        
        self.cam.send(frame)
        self._frame_count += 1
    
    @property
    def frame_count(self) -> int:
        return self._frame_count


def test_virtual_camera(device: str = "/dev/video10", duration: int = 5):
    """
    Test virtual camera with color bars.
    
    Args:
        device: v4l2loopback device
        duration: Test duration in seconds
    """
    import time
    
    width, height, fps = 1920, 1080, 30
    
    # Generate color bars
    colors = [
        [255, 255, 255],  # White
        [255, 255, 0],    # Yellow
        [0, 255, 255],    # Cyan
        [0, 255, 0],      # Green
        [255, 0, 255],    # Magenta
        [255, 0, 0],      # Red
        [0, 0, 255],      # Blue
        [0, 0, 0],        # Black
    ]
    
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    bar_width = width // len(colors)
    for i, color in enumerate(colors):
        frame[:, i*bar_width:(i+1)*bar_width] = color
    
    print(f"Testing virtual camera at {device}")
    print(f"Open OBS and add 'Video Capture Device (V4L2)' -> 'FaceTrack'")
    print(f"Running for {duration} seconds...")
    
    with VirtualCameraOutput(device=device, size=(width, height), fps=fps) as cam:
        start = time.time()
        while time.time() - start < duration:
            # Add timestamp overlay position indicator
            t = int((time.time() - start) * 100) % width
            display_frame = frame.copy()
            display_frame[:50, t:t+20] = [255, 0, 0]
            
            cam.write_frame(display_frame)
    
    print("✓ Test complete")


if __name__ == "__main__":
    test_virtual_camera()
