# ROCm Face Tracker - Master Implementation Plan

**Project Goal:** Replace CPU-based OBS Face Tracker with GPU-accelerated solution  
**Expected Outcome:** CPU usage drops from **106%** → <10%, tracking runs on RX 6900 XT  
**Estimated Effort:** 4-8 hours across 2-3 sessions  
**Last Updated:** January 2025

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Cam Link 4K   │────▶│  ROCm Pipeline   │────▶│  Virtual Cam    │
│   /dev/video0   │     │   (GPU-based)    │     │  /dev/video10   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                         │
                               ▼                         ▼
                        ┌──────────────┐          ┌───────────┐
                        │  YOLO-Face   │          │    OBS    │
                        │  Detection   │          │  Studio   │
                        │  (ROCm/HIP)  │          └───────────┘
                        └──────────────┘
```

**Data Flow:**
1. Capture frame from Cam Link 4K (1080p or 4K)
2. Run face detection on GPU (**YOLO-Face** - not RetinaFace)
3. Calculate crop region centered on face with smoothing
4. Crop/zoom frame to output resolution
5. Write to v4l2loopback virtual camera via **pyvirtualcam**
6. OBS captures virtual camera as standard source

---

## Phase 1: Environment Setup
**Goal:** Verify ROCm + PyTorch + camera access works  
**Time:** 30-60 minutes

### Tasks

- [ ] **1.1** Set up environment variables (CRITICAL for RDNA2)
  ```bash
  # Add to ~/.bashrc
  export HSA_OVERRIDE_GFX_VERSION=10.3.0
  export PYTORCH_ROCM_ARCH=gfx1030
  export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
  export ROCM_PATH=/opt/rocm
  
  # Reload
  source ~/.bashrc
  ```

- [ ] **1.2** Verify ROCm installation
  ```bash
  rocminfo | grep gfx
  # Expected: gfx1030
  
  # Check version (should be 6.2.x or 6.4.x)
  cat /opt/rocm/.info/version
  ```

- [ ] **1.3** Create virtual environment and install PyTorch with ROCm
  ```bash
  # Create venv
  python -m venv ~/venvs/facetrack
  source ~/venvs/facetrack/bin/activate
  
  # Install PyTorch (ROCm 6.2.4 - most stable for RDNA2)
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
      --index-url https://download.pytorch.org/whl/rocm6.2.4
  
  # Verify
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  # Expected: True AMD Radeon RX 6900 XT
  ```

- [ ] **1.4** Install dependencies
  ```bash
  pip install opencv-python numpy pyyaml
  pip install pyvirtualcam  # NOT pyfakewebcam (abandoned since 2018)
  pip install ultralytics   # For YOLO-Face
  ```

- [ ] **1.5** Setup v4l2loopback (with Chrome/WebRTC compatibility)
  ```bash
  # Load module with all required options
  sudo modprobe v4l2loopback \
    devices=1 \
    video_nr=10 \
    card_label="FaceTrack" \
    exclusive_caps=1 \
    max_buffers=2
  
  # Test with color bars
  ffmpeg -f lavfi -i testsrc=size=1920x1080:rate=30 -pix_fmt yuv420p -f v4l2 /dev/video10
  ```

- [ ] **1.6** Verify OBS can see virtual camera
  - Add "Video Capture Device (V4L2)" source
  - Select "FaceTrack" device
  - Should show test pattern
  - Note: Use OBS 32.x+ (31.x had v4l2loopback bugs)

### Phase 1 Deliverable
- [ ] Python script that captures from camera and writes to virtual cam (passthrough, no processing)

```python
# test_passthrough.py
import cv2
import pyvirtualcam
import numpy as np

cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

with pyvirtualcam.Camera(width=1920, height=1080, fps=30, device='/dev/video10') as cam:
    print(f'Virtual camera: {cam.device}')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # pyvirtualcam expects RGB, OpenCV gives BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam.send(frame_rgb)
        cam.sleep_until_next_frame()
```

---

## Phase 2: Face Detection Module
**Goal:** GPU-accelerated face detection with bounding boxes  
**Time:** 1-2 hours

### Tasks

- [ ] **2.1** Clone and setup YOLO-Face

  ```bash
  cd ~/Projects/rocm-face-tracker
  git clone https://github.com/akanametov/yolo-face
  
  # Download pre-trained weights (choose based on speed/accuracy tradeoff)
  # yolov11n-face.pt - fastest, good for real-time
  # yolov11s-face.pt - balanced
  # yolov11m-face.pt - most accurate
  
  # Weights are auto-downloaded on first use, or download manually from releases
  ```

- [ ] **2.2** Create detector.py module
  ```python
  # src/detector.py
  from ultralytics import YOLO
  import torch
  from dataclasses import dataclass
  from typing import List, Optional
  import numpy as np

  @dataclass
  class BoundingBox:
      x1: float
      y1: float
      x2: float
      y2: float
      confidence: float
      
      @property
      def center(self) -> tuple:
          return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
      
      @property
      def area(self) -> float:
          return (self.x2 - self.x1) * (self.y2 - self.y1)

  class FaceDetector:
      def __init__(self, model_path: str = "yolov11n-face.pt", 
                   device: str = "cuda", 
                   conf_threshold: float = 0.5):
          self.model = YOLO(model_path)
          self.device = device
          self.conf_threshold = conf_threshold
          
          # Verify GPU is being used
          print(f"YOLO using device: {self.device}")
          print(f"ROCm available: {torch.cuda.is_available()}")
      
      def detect(self, frame: np.ndarray) -> List[BoundingBox]:
          """Returns list of face bounding boxes, largest first"""
          results = self.model(frame, device=self.device, verbose=False)
          
          boxes = []
          for result in results:
              for box in result.boxes:
                  if box.conf >= self.conf_threshold:
                      x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                      boxes.append(BoundingBox(
                          x1=float(x1), y1=float(y1),
                          x2=float(x2), y2=float(y2),
                          confidence=float(box.conf)
                      ))
          
          # Sort by area (largest first)
          boxes.sort(key=lambda b: b.area, reverse=True)
          return boxes
  ```

- [ ] **2.3** Benchmark detection speed
  ```python
  # benchmark.py
  import cv2
  import time
  import torch
  from detector import FaceDetector

  detector = FaceDetector()
  cap = cv2.VideoCapture('/dev/video0')
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

  times = []
  for _ in range(100):
      ret, frame = cap.read()
      start = time.perf_counter()
      boxes = detector.detect(frame)
      elapsed = time.perf_counter() - start
      times.append(elapsed)

  avg_ms = sum(times) / len(times) * 1000
  fps = 1000 / avg_ms
  print(f"Average detection time: {avg_ms:.1f}ms ({fps:.1f} FPS)")
  # Target: >30 FPS on 1080p input
  ```

- [ ] **2.4** Handle edge cases
  - No face detected → hold last known position
  - Multiple faces → track largest/primary
  - Face at frame edge → constrain crop region

### Phase 2 Deliverable
- [ ] Script that shows camera feed with face bounding boxes overlaid, running on GPU

---

## Phase 3: Tracking & Smoothing
**Goal:** Smooth, non-jittery crop region that follows face naturally  
**Time:** 1-2 hours

### Tasks

- [ ] **3.1** Implement smoothing filter
  ```python
  # src/tracker.py
  from dataclasses import dataclass
  from typing import Optional
  import numpy as np

  @dataclass
  class TrackedPosition:
      cx: float  # center x
      cy: float  # center y
      width: float
      height: float

  class SmoothTracker:
      def __init__(self, smoothing_factor: float = 0.15, deadzone: float = 20):
          self.smoothing_factor = smoothing_factor
          self.deadzone = deadzone
          self.current: Optional[TrackedPosition] = None
          self.frames_without_detection = 0
          self.max_frames_without_detection = 30  # ~1 second at 30fps
      
      def update(self, bbox) -> Optional[TrackedPosition]:
          """Update tracking with new detection (or None if no face)"""
          if bbox is None:
              self.frames_without_detection += 1
              if self.frames_without_detection > self.max_frames_without_detection:
                  return None  # Signal to zoom out to full frame
              return self.current  # Hold last position
          
          self.frames_without_detection = 0
          
          # Convert bbox to tracked position
          new_pos = TrackedPosition(
              cx=(bbox.x1 + bbox.x2) / 2,
              cy=(bbox.y1 + bbox.y2) / 2,
              width=bbox.x2 - bbox.x1,
              height=bbox.y2 - bbox.y1
          )
          
          if self.current is None:
              self.current = new_pos
              return self.current
          
          # Check deadzone
          dx = abs(new_pos.cx - self.current.cx)
          dy = abs(new_pos.cy - self.current.cy)
          
          if dx < self.deadzone and dy < self.deadzone:
              return self.current  # No update needed
          
          # Exponential moving average (lerp)
          alpha = self.smoothing_factor
          self.current = TrackedPosition(
              cx=self.current.cx + alpha * (new_pos.cx - self.current.cx),
              cy=self.current.cy + alpha * (new_pos.cy - self.current.cy),
              width=self.current.width + alpha * (new_pos.width - self.current.width),
              height=self.current.height + alpha * (new_pos.height - self.current.height)
          )
          
          return self.current
  ```

- [ ] **3.2** Add deadzone to prevent micro-movements
  - Already included in SmoothTracker above
  - Only update target if face moved > N pixels

- [ ] **3.3** Implement velocity-based prediction (optional)
  - Reduces lag when moving quickly
  - Predicts next position based on movement history

- [ ] **3.4** Tune parameters
  - `smoothing_factor`: 0.1 (smooth) to 0.3 (responsive)
  - `deadzone`: 10-30 pixels
  - `zoom_level`: 1.5x to 3x depending on wide-angle lens

### Phase 3 Deliverable
- [ ] Smooth tracking visualization with configurable parameters

---

## Phase 4: Crop & Output Pipeline
**Goal:** Crop frames and output to virtual camera  
**Time:** 1-2 hours

### Tasks

- [ ] **4.1** Implement GPU-accelerated cropping
  ```python
  # src/cropper.py
  import torch
  import torchvision.transforms.functional as TF
  import numpy as np
  from typing import Tuple

  class FrameCropper:
      def __init__(self, output_size: Tuple[int, int] = (1920, 1080),
                   zoom_level: float = 2.0,
                   device: str = "cuda"):
          self.output_size = output_size
          self.zoom_level = zoom_level
          self.device = device
      
      def crop(self, frame: np.ndarray, tracked_pos) -> np.ndarray:
          """
          Crop region around tracked position and resize to output_size.
          Uses GPU acceleration via torchvision.
          """
          if tracked_pos is None:
              # No tracking - return resized full frame
              frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).to(self.device)
              resized = TF.resize(frame_tensor, list(self.output_size[::-1]))
              return resized.permute(1, 2, 0).cpu().numpy()
          
          h, w = frame.shape[:2]
          out_w, out_h = self.output_size
          
          # Calculate crop region based on zoom level
          crop_w = w / self.zoom_level
          crop_h = h / self.zoom_level
          
          # Center on tracked position
          x1 = tracked_pos.cx - crop_w / 2
          y1 = tracked_pos.cy - crop_h / 2
          x2 = x1 + crop_w
          y2 = y1 + crop_h
          
          # Clamp to frame bounds
          if x1 < 0:
              x2 -= x1
              x1 = 0
          if y1 < 0:
              y2 -= y1
              y1 = 0
          if x2 > w:
              x1 -= (x2 - w)
              x2 = w
          if y2 > h:
              y1 -= (y2 - h)
              y2 = h
          
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          
          # GPU-accelerated crop and resize
          frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(self.device)
          cropped = frame_tensor[:, y1:y2, x1:x2]
          resized = TF.resize(cropped, [out_h, out_w], antialias=True)
          
          return resized.permute(1, 2, 0).byte().cpu().numpy()
  ```

- [ ] **4.2** Handle aspect ratio
  - Input: wide-angle (16:9 or wider)
  - Output: standard 16:9 crop
  - Maintain face centered in output frame

- [ ] **4.3** Implement pyvirtualcam output
  ```python
  # src/output.py
  import pyvirtualcam
  import numpy as np
  from typing import Tuple

  class VirtualCameraOutput:
      def __init__(self, device: str = "/dev/video10", 
                   size: Tuple[int, int] = (1920, 1080),
                   fps: int = 30):
          self.device = device
          self.size = size
          self.fps = fps
          self.cam = None
      
      def __enter__(self):
          self.cam = pyvirtualcam.Camera(
              width=self.size[0], 
              height=self.size[1], 
              fps=self.fps,
              device=self.device
          )
          print(f"Virtual camera started: {self.cam.device}")
          return self
      
      def __exit__(self, *args):
          if self.cam:
              self.cam.close()
      
      def write_frame(self, frame: np.ndarray):
          """Write RGB frame to virtual camera"""
          self.cam.send(frame)
          self.cam.sleep_until_next_frame()
  ```

- [ ] **4.4** Optimize pipeline for latency
  - Minimize GPU↔CPU transfers
  - Use async processing where possible
  - Target: <50ms end-to-end latency

### Phase 4 Deliverable
- [ ] Complete pipeline writing tracked/cropped output to virtual camera

---

## Phase 5: Configuration & Polish
**Goal:** User-friendly configuration and robust operation  
**Time:** 1 hour

### Tasks

- [ ] **5.1** Create config.yaml
  ```yaml
  camera:
    input_device: "/dev/video0"
    input_resolution: [1920, 1080]
    fps: 30
  
  output:
    device: "/dev/video10"
    resolution: [1920, 1080]
  
  tracking:
    smoothing: 0.15
    deadzone: 20
    zoom_level: 2.0
    detection_interval: 1  # Detect every N frames (1 = every frame)
    no_face_timeout: 30    # Frames before zooming out
  
  model:
    type: "yoloface"
    weights: "yolov11n-face.pt"  # n=fast, s=balanced, m=accurate
    confidence: 0.5
  
  # AMD ROCm settings
  rocm:
    device: "cuda"
    # These should also be set as environment variables
    hsa_override_gfx_version: "10.3.0"
    pytorch_rocm_arch: "gfx1030"
  ```

- [ ] **5.2** Add runtime controls (optional)
  - Keyboard shortcuts for zoom in/out
  - Toggle tracking on/off
  - Reset to center

- [ ] **5.3** Error handling
  - Camera disconnect → graceful retry
  - No face timeout → zoom out to full frame
  - GPU OOM → reduce resolution / skip frames

- [ ] **5.4** Logging and monitoring
  - FPS counter
  - GPU memory usage
  - Detection confidence

### Phase 5 Deliverable
- [ ] Production-ready script with config file

---

## Phase 6: System Integration
**Goal:** Run as system service, integrate with OBS workflow  
**Time:** 30 minutes

### Tasks

- [ ] **6.1** Create systemd service
  ```ini
  # /etc/systemd/system/rocm-facetracker.service
  [Unit]
  Description=ROCm Face Tracker for OBS
  After=graphical.target
  
  [Service]
  Type=simple
  User=YOUR_USERNAME
  
  # CRITICAL: Environment variables for RDNA2
  Environment="HSA_OVERRIDE_GFX_VERSION=10.3.0"
  Environment="PYTORCH_ROCM_ARCH=gfx1030"
  Environment="PYTORCH_HIP_ALLOC_CONF=expandable_segments:True"
  Environment="ROCM_PATH=/opt/rocm"
  
  # Load v4l2loopback with Chrome-compatible settings
  ExecStartPre=/sbin/modprobe v4l2loopback devices=1 video_nr=10 card_label=FaceTrack exclusive_caps=1 max_buffers=2
  
  # Activate venv and run
  ExecStart=/bin/bash -c 'source /home/YOUR_USERNAME/venvs/facetrack/bin/activate && python /home/YOUR_USERNAME/Projects/rocm-face-tracker/src/main.py'
  
  Restart=on-failure
  RestartSec=5
  
  [Install]
  WantedBy=graphical.target
  ```

- [ ] **6.2** Update OBS scene
  - **DISABLE** Face Tracker plugin from camera source
  - Replace camera source with virtual camera (FaceTrack)
  - Adjust any dependent filters/scenes

- [ ] **6.3** Document startup procedure
  ```bash
  # Enable service to start on boot
  sudo systemctl enable rocm-facetracker
  
  # Start service
  sudo systemctl start rocm-facetracker
  
  # Check status
  sudo systemctl status rocm-facetracker
  
  # View logs
  journalctl -u rocm-facetracker -f
  ```

### Phase 6 Deliverable
- [ ] Working systemd service, OBS configured

---

## Success Criteria

| Metric | Target | How to Verify |
|--------|--------|---------------|
| CPU Usage | <10% | `htop` / `ps aux` during stream |
| GPU Usage | 10-20% | `rocm-smi` during stream |
| Latency | <50ms | Visual inspection |
| Stability | No crashes | 2+ hour test session |
| Frame Rate | 30 FPS | OBS stats window |

---

## Fallback Plan

If ROCm face detection proves unstable:

1. **Reduce detection frequency** - Run detection every 3-5 frames, interpolate between
2. **Use smaller model** - yolov11n-face instead of yolov11s/m
3. **Containerize** - Run in Docker to isolate from system GPU issues
4. **CPU fallback with efficiency** - Use YOLO-Face on CPU (still faster than dlib)

---

## Quick Start Commands

```bash
# Create project directory
mkdir -p ~/Projects/rocm-face-tracker/{src,models,systemd}
cd ~/Projects/rocm-face-tracker

# Set up environment variables (add to ~/.bashrc)
echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
echo 'export PYTORCH_ROCM_ARCH=gfx1030' >> ~/.bashrc
echo 'export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
source ~/.bashrc

# Create and activate venv
python -m venv ~/venvs/facetrack
source ~/venvs/facetrack/bin/activate

# Install deps (ROCm 6.2.4 - stable for RDNA2)
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/rocm6.2.4
pip install opencv-python numpy pyyaml pyvirtualcam ultralytics

# Clone YOLO-Face
git clone https://github.com/akanametov/yolo-face

# Setup virtual camera
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="FaceTrack" exclusive_caps=1 max_buffers=2

# Run (once implemented)
python src/main.py --config config.yaml
```

---

## Reference Links

- **YOLO-Face:** https://github.com/akanametov/yolo-face
- **pyvirtualcam:** https://github.com/letmaik/pyvirtualcam
- **v4l2loopback:** https://github.com/umlaeute/v4l2loopback
- **ROCm PyTorch wheels:** https://pytorch.org/get-started/locally/
- **AMD-GPU-BOOST:** https://github.com/Painter3000/AMD-GPU-BOOST
- **Ultralytics YOLO:** https://docs.ultralytics.com/

---

## Changelog

### January 2025 Update
- ❌ Removed RetinaFace as primary recommendation (maintenance stalled)
- ❌ Removed MediaPipe/BlazeFace (no AMD ROCm support - CPU only!)
- ❌ Removed pyfakewebcam (abandoned since 2018)
- ❌ Removed v4l2py (use pyvirtualcam instead)
- ✅ Added YOLO-Face as primary face detection model
- ✅ Added pyvirtualcam for virtual camera output
- ✅ Updated PyTorch install to ROCm 6.2.4 (was 6.0)
- ✅ Added critical RDNA2 environment variables
- ✅ Added max_buffers=2 to v4l2loopback for Chrome compatibility
- ✅ Updated CPU usage measurement (106%, not 45-52%)
- ✅ Added note about CU vs WGP detection (40 shown is correct)
- ✅ Added OBS version requirement (32.x+)
