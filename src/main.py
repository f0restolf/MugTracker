#!/usr/bin/env python3
"""
ROCm Face Tracker - Main Pipeline

GPU-accelerated face tracking for OBS auto-framing.
Replaces CPU-heavy dlib-based solutions with YOLO-Face on AMD ROCm.

Usage:
    python main.py                     # Use defaults
    python main.py --config config.yaml
    python main.py --input /dev/video2 --zoom 2.5

Environment (REQUIRED for AMD RDNA2):
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    export PYTORCH_ROCM_ARCH=gfx1030
    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
"""
import sys
import os
import time
import signal
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import yaml
from typing import Optional

from detector import FaceDetector, warmup_detector
from tracker import SmoothTracker, VelocityTracker
from cropper import FrameCropper, AdaptiveZoomCropper
from output import VirtualCameraOutput


class FaceTrackingPipeline:
    """
    Complete face tracking pipeline.
    
    Captures from camera, detects faces on GPU, applies smooth tracking,
    crops frame, and outputs to virtual camera.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.running = False
        self.frame_count = 0
        self.fps_history = []
        
        # Initialize components
        print("\n=== Initializing Face Tracking Pipeline ===\n")
        
        # Detector
        model_config = config.get('model', {})
        self.detector = FaceDetector(
            model_path=model_config.get('weights', 'yolov11n-face.pt'),
            device=config.get('rocm', {}).get('device', 'cuda'),
            conf_threshold=model_config.get('confidence', 0.5)
        )
        
        # Tracker
        track_config = config.get('tracking', {})
        tracker_class = VelocityTracker if track_config.get('velocity_prediction', False) else SmoothTracker
        self.tracker = tracker_class(
            smoothing_factor=track_config.get('smoothing', 0.15),
            deadzone=track_config.get('deadzone', 20),
            max_frames_without_detection=track_config.get('no_face_timeout', 30)
        )
        
        # Cropper
        output_config = config.get('output', {})
        cropper_class = AdaptiveZoomCropper if track_config.get('adaptive_zoom', False) else FrameCropper
        self.cropper = cropper_class(
            output_size=tuple(output_config.get('resolution', [1920, 1080])),
            zoom_level=track_config.get('zoom_level', 2.0),
            device=config.get('rocm', {}).get('device', 'cuda')
        )
        
        # Detection interval (skip frames for performance)
        self.detection_interval = track_config.get('detection_interval', 1)
        
        # Stats
        self.show_stats = config.get('debug', {}).get('show_stats', True)
        
        print("\n✓ Pipeline initialized")
    
    def run(self):
        """Main processing loop"""
        cam_config = self.config.get('camera', {})
        output_config = self.config.get('output', {})
        
        input_device = cam_config.get('input_device', '/dev/video0')
        input_res = cam_config.get('input_resolution', [1920, 1080])
        fps = cam_config.get('fps', 30)
        
        output_device = output_config.get('device', '/dev/video10')
        output_res = output_config.get('resolution', [1920, 1080])
        
        # Open input camera
        print(f"\nOpening camera: {input_device}")
        cap = cv2.VideoCapture(input_device)
        
        if not cap.isOpened():
            print(f"✗ Failed to open camera: {input_device}")
            print("\nTroubleshooting:")
            print("  1. Check device exists: ls -la /dev/video*")
            print("  2. Test with: ffplay -f v4l2 -i", input_device)
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_res[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"✓ Camera opened: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        # Warm up detector
        warmup_detector(self.detector, (actual_width, actual_height))
        
        # Start virtual camera output
        self.running = True
        
        try:
            with VirtualCameraOutput(
                device=output_device,
                size=tuple(output_res),
                fps=fps
            ) as vcam:
                print(f"\n=== Pipeline Running ===")
                print("Press Ctrl+C to stop\n")
                
                last_bbox = None
                frame_idx = 0
                
                while self.running:
                    loop_start = time.perf_counter()
                    
                    # Capture frame with buffer flush to minimize latency
                    # (grab discards stale frames from OpenCV's internal buffer)
                    for _ in range(2):
                        cap.grab()
                    ret, frame = cap.retrieve()
                    if not ret:
                        print("⚠ Failed to read frame, retrying...")
                        time.sleep(0.1)
                        continue
                    
                    # Detect faces (optionally skip frames)
                    if frame_idx % self.detection_interval == 0:
                        detect_start = time.perf_counter()
                        last_bbox = self.detector.detect_primary(frame)
                        detect_time = (time.perf_counter() - detect_start) * 1000
                    
                    # Update tracking
                    tracked_pos = self.tracker.update(last_bbox)
                    
                    # Crop and resize
                    output_frame = self.cropper.crop(frame, tracked_pos)
                    
                    # Output to virtual camera
                    vcam.write_frame(output_frame)
                    
                    # Stats
                    frame_idx += 1
                    self.frame_count += 1
                    loop_time = (time.perf_counter() - loop_start) * 1000
                    self.fps_history.append(loop_time)
                    
                    if self.show_stats and frame_idx % 30 == 0:
                        avg_ms = sum(self.fps_history[-30:]) / min(30, len(self.fps_history))
                        current_fps = 1000 / avg_ms if avg_ms > 0 else 0
                        face_status = "✓" if last_bbox else "✗"
                        print(f"\r[{self.frame_count:6d}] {current_fps:5.1f} fps | "
                              f"Face: {face_status} | Loop: {loop_time:5.1f}ms", end="")
        
        except KeyboardInterrupt:
            print("\n\nStopping...")
        
        finally:
            self.running = False
            cap.release()
            
            # Final stats
            if self.fps_history:
                avg_ms = sum(self.fps_history) / len(self.fps_history)
                print(f"\n\n=== Session Stats ===")
                print(f"Total frames: {self.frame_count}")
                print(f"Average loop time: {avg_ms:.1f}ms ({1000/avg_ms:.1f} fps)")
        
        return True
    
    def stop(self):
        """Signal pipeline to stop"""
        self.running = False


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file or return defaults"""
    defaults = {
        'camera': {
            'input_device': '/dev/video0',
            'input_resolution': [1920, 1080],
            'fps': 30
        },
        'output': {
            'device': '/dev/video10',
            'resolution': [1920, 1080]
        },
        'tracking': {
            'smoothing': 0.15,
            'deadzone': 20,
            'zoom_level': 2.0,
            'detection_interval': 1,
            'no_face_timeout': 30,
            'adaptive_zoom': False,
            'velocity_prediction': False
        },
        'model': {
            'weights': 'yolov11n-face.pt',
            'confidence': 0.5
        },
        'rocm': {
            'device': 'cuda'
        },
        'debug': {
            'show_stats': True
        }
    }
    
    if config_path and Path(config_path).exists():
        print(f"Loading config: {config_path}")
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        
        # Deep merge
        def merge(base, override):
            for k, v in override.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    merge(base[k], v)
                else:
                    base[k] = v
        
        merge(defaults, user_config)
    
    return defaults


def main():
    parser = argparse.ArgumentParser(
        description="ROCm Face Tracker - GPU-accelerated face tracking for OBS"
    )
    parser.add_argument('--config', '-c', type=str, help='Path to config.yaml')
    parser.add_argument('--input', '-i', type=str, help='Input camera device')
    parser.add_argument('--output', '-o', type=str, help='Output virtual camera device')
    parser.add_argument('--zoom', '-z', type=float, help='Zoom level (1.0-4.0)')
    parser.add_argument('--model', '-m', type=str, help='Model weights file')
    parser.add_argument('--no-stats', action='store_true', help='Disable FPS display')
    
    args = parser.parse_args()
    
    # Check environment
    if not os.environ.get('HSA_OVERRIDE_GFX_VERSION'):
        print("⚠ Warning: HSA_OVERRIDE_GFX_VERSION not set")
        print("  For AMD RDNA2, set: export HSA_OVERRIDE_GFX_VERSION=10.3.0")
    
    # Load config
    config = load_config(args.config)
    
    # Override with CLI args
    if args.input:
        config['camera']['input_device'] = args.input
    if args.output:
        config['output']['device'] = args.output
    if args.zoom:
        config['tracking']['zoom_level'] = args.zoom
    if args.model:
        config['model']['weights'] = args.model
    if args.no_stats:
        config['debug']['show_stats'] = False
    
    # Run pipeline
    pipeline = FaceTrackingPipeline(config)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        pipeline.stop()
    signal.signal(signal.SIGINT, signal_handler)
    
    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
