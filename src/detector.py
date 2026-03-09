"""
Face Detection Module - YOLO-Face with ROCm/HIP acceleration
"""
from ultralytics import YOLO
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class BoundingBox:
    """Face bounding box with confidence score"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    
    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height


class FaceDetector:
    """
    GPU-accelerated face detection using YOLO-Face.
    
    For AMD ROCm, ensure these environment variables are set:
        HSA_OVERRIDE_GFX_VERSION=10.3.0
        PYTORCH_ROCM_ARCH=gfx1030
    """
    
    def __init__(
        self, 
        model_path: str = "yolov11n-face.pt",
        device: str = "cuda",
        conf_threshold: float = 0.5,
        verbose: bool = False
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.verbose = verbose
        
        # Resolve model path
        model_file = Path(model_path)
        if not model_file.exists():
            # Try relative to script location
            script_dir = Path(__file__).parent.parent
            alt_paths = [
                script_dir / model_path,
                script_dir / "models" / Path(model_path).name,
                Path.cwd() / model_path,
                Path.cwd() / "models" / Path(model_path).name,
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_file = alt_path
                    break
        
        print(f"Loading YOLO-Face model: {model_file}")
        self.model = YOLO(str(model_file))
        
        # Verify GPU
        if torch.cuda.is_available():
            print(f"✓ ROCm/CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  WGPs: {torch.cuda.get_device_properties(0).multi_processor_count}")
        else:
            print("⚠ GPU not available, falling back to CPU")
            self.device = "cpu"
    
    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Detect faces in frame.
        
        Args:
            frame: BGR image (OpenCV format)
            
        Returns:
            List of BoundingBox objects, sorted by area (largest first)
        """
        results = self.model(
            frame, 
            device=self.device, 
            verbose=self.verbose,
            conf=self.conf_threshold
        )
        
        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf.cpu().numpy())
                if conf >= self.conf_threshold:
                    coords = box.xyxy[0].cpu().numpy()
                    boxes.append(BoundingBox(
                        x1=float(coords[0]),
                        y1=float(coords[1]),
                        x2=float(coords[2]),
                        y2=float(coords[3]),
                        confidence=conf
                    ))
        
        # Sort by area (track largest face)
        boxes.sort(key=lambda b: b.area, reverse=True)
        return boxes
    
    def detect_primary(self, frame: np.ndarray) -> Optional[BoundingBox]:
        """Detect and return only the primary (largest) face"""
        boxes = self.detect(frame)
        return boxes[0] if boxes else None


def warmup_detector(detector: FaceDetector, size: tuple = (1920, 1080)):
    """Run a few inference passes to warm up GPU"""
    print("Warming up detector...")
    dummy = np.zeros((*size[::-1], 3), dtype=np.uint8)
    for _ in range(3):
        detector.detect(dummy)
    print("✓ Detector warmed up")
