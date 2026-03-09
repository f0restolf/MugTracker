# ROCm Face Tracker Project - System Knowledge Base

**Created:** December 2024  
**Last Updated:** January 2025  
**System:** Nobara Linux 43 (Fedora-based)  
**Purpose:** GPU-accelerated face tracking for OBS auto-framing

---

## Hardware Specifications

### GPU (Primary Compute Target)
- **Model:** AMD Radeon RX 6900 XT (Navi 21 / SIENNA_CICHLID)
- **Device ID:** 0x73bf
- **VRAM:** 16 GB GDDR6
- **Architecture:** RDNA2
- **ROCm Target:** gfx1030
- **Compute Units:** 80 (reported as 40 WGPs in PyTorch - this is normal, see notes below)
- **PCI Address:** 0000:0c:00.0
- **Official ROCm Support:** Unofficial but works with workarounds

### CPU
- **Model:** AMD Ryzen 7 5800X
- **Cores/Threads:** 8 / 16
- **Current load from face tracking:** ~106% single process (dlib-based, to be replaced)

### RAM
- **Total:** 64 GB DDR4-3600

### Camera Setup
- **Camera:** Wide-angle webcam (unspecified model)
- **Capture Device:** Elgato Cam Link 4K
- **Connection:** USB 3.0, presents as /dev/videoX
- **Use Case:** Auto-framing/following face in wide shot

---

## Software Environment

### Operating System
```
OS: Nobara Linux 43 (KDE Plasma Desktop Edition)
Kernel: 6.17.10-200.nobara.fc43.x86_64
DE: KDE Plasma 6.4.4+ (Wayland)
```

### ROCm Stack
```bash
# Verify with:
rocminfo | grep gfx
# Expected: gfx1030

/opt/rocm/bin/rocm-smi

# Check ROCm version
cat /opt/rocm/.info/version
# Recommended: 6.2.4 or 6.4.x for RDNA2
```

### Required Environment Variables (CRITICAL for RDNA2)
```bash
# Add to ~/.bashrc or systemd service
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export ROCM_PATH=/opt/rocm
```

### Python Environment
```bash
# System Python
python3 --version  # Python 3.12+

# Key packages needed:
# - torch (ROCm version) - see PyTorch Setup section
# - torchvision
# - opencv-python
# - numpy
# - pyvirtualcam (NOT pyfakewebcam - it's abandoned)
# - ultralytics (for YOLO-Face)
```

### Docker (Available for containerized approach)
```bash
# GPU passthrough confirmed working
docker run --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e PYTORCH_ROCM_ARCH=gfx1030 \
  -it rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.3.0
```

### OBS Studio
- **Version:** 32.0.4+ recommended (31.x had v4l2loopback issues)
- **Current Plugin:** Face Tracker (obs-face-tracker 0.9.1 by norihiro)
- **Problem:** Uses dlib → CPU-only → **106% CPU usage** (measured)
- **Goal:** Replace with GPU-accelerated virtual camera input

---

## Virtual Camera Infrastructure

### v4l2loopback (Required)
```bash
# Check if loaded
lsmod | grep v4l2loopback

# Load with proper settings for OBS AND Chrome/WebRTC compatibility
sudo modprobe v4l2loopback \
  devices=1 \
  video_nr=10 \
  card_label="ROCm-FaceTrack" \
  exclusive_caps=1 \
  max_buffers=2

# IMPORTANT: 
# - exclusive_caps=1 is REQUIRED for Chrome to recognize the camera
# - max_buffers=2 prevents video freezing issues
```

### Persistent Configuration
```bash
# /etc/modules-load.d/v4l2loopback.conf
v4l2loopback

# /etc/modprobe.d/v4l2loopback.conf
options v4l2loopback video_nr=10 card_label="ROCm-FaceTrack" exclusive_caps=1 max_buffers=2
```

### Device Paths
```bash
# List video devices
v4l2-ctl --list-devices

# Cam Link 4K typically appears as /dev/video0 or /dev/video2
# Virtual output will be /dev/video10 (as configured above)
```

---

## ROCm PyTorch Setup

### Recommended: Stable Install (ROCm 6.2.4)
```bash
# Create virtual environment (recommended)
python -m venv ~/venvs/facetrack
source ~/venvs/facetrack/bin/activate

# Install PyTorch with ROCm 6.2.4 (most stable for RDNA2)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/rocm6.2.4
```

### Alternative: Cutting Edge (ROCm 6.4)
```bash
# Latest features, potentially less stable
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/rocm6.4
```

### Available ROCm Wheel URLs (January 2025)
| URL | PyTorch Version | Notes |
|-----|-----------------|-------|
| `https://download.pytorch.org/whl/rocm6.4` | 2.8.0, 2.9.0 | Cutting edge |
| `https://download.pytorch.org/whl/rocm6.3` | 2.7.0, 2.7.1 | Recent |
| `https://download.pytorch.org/whl/rocm6.2.4` | 2.6.0 | **Recommended stable** |
| `https://download.pytorch.org/whl/rocm6.2` | 2.5.0, 2.5.1 | Conservative |

### Verification
```python
import torch

print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Multiprocessors (WGPs): {torch.cuda.get_device_properties(0).multi_processor_count}")

# Expected output:
# ROCm available: True
# Device: AMD Radeon RX 6900 XT
# Multiprocessors (WGPs): 40  <-- This is correct! See notes below.
```

---

## Face Detection Models (GPU-Compatible)

### ⚠️ IMPORTANT: MediaPipe Does NOT Support AMD GPUs
MediaPipe requires CUDA and will **only run on CPU** with AMD hardware. Do not use MediaPipe or BlazeFace for this project.

### Recommended: YOLO-Face (Best for AMD ROCm)
- **Repository:** https://github.com/akanametov/yolo-face
- Actively maintained with YOLOv11/YOLOv12 face models (late 2024)
- Works directly with PyTorch ROCm - no modifications needed
- Pre-trained models available for WIDERFace benchmark
- GPL-3.0 license

```bash
# Installation
pip install ultralytics
git clone https://github.com/akanametov/yolo-face
cd yolo-face

# Download pre-trained weights
# Available: yolov11n-face.pt, yolov11s-face.pt, yolov11m-face.pt

# Test inference
python -c "
from ultralytics import YOLO
model = YOLO('yolov11n-face.pt')
results = model('test_image.jpg')
print(results)
"
```

### Alternative: RetinaFace (if higher accuracy needed)
- Original `biubug6/Pytorch_Retinaface` has 141 open issues, maintenance stalled
- **Use instead:** `yakhyo/retinaface-pytorch` (updated November 2024)
- Or: `pip install retinafacex` for ONNX-based inference

### NOT Recommended for AMD
- **MediaPipe/BlazeFace** - No ROCm support, CPU-only
- **InsightFace/SCRFD** - Uses ONNX Runtime, CPU-only on AMD unless custom build

---

## Performance Targets

| Metric | Current (dlib) | Target (ROCm) |
|--------|----------------|---------------|
| CPU Usage | **106%** (measured) | <10% |
| GPU Usage | ~0% | 10-20% |
| Latency | ~50-100ms | <50ms |
| Frame Rate | 30 FPS | 30-60 FPS |
| Memory | ~2GB RAM | ~1-2GB VRAM |

---

## Known Issues & Technical Notes

### PyTorch Reports 40 CUs Instead of 80 (NOT A BUG)
This is **by design**. RDNA2/3 GPUs operate in Workgroup Processor (WGP) mode, grouping CUs in pairs. PyTorch reports WGPs (40), not individual CUs (80). Each WGP executes twice as many Wave32 wavefronts simultaneously, so **performance is not halved**.

The real issue is CUDA-centric libraries (xFormers, FlashAttention) under-scheduling workgroups. For most face detection workloads, this is not significant.

If you experience performance issues, try:
```bash
# AMD-GPU-BOOST environment variables (use with caution)
export BOOST_FORCE_MP_COUNT=80
export BOOST_FORCE_WARP_SIZE=64
```

See: https://github.com/Painter3000/AMD-GPU-BOOST

### ROCm Version Compatibility
- ROCm 6.4.3+ may have SIGSEGV crashes with `HSA_OVERRIDE_GFX_VERSION` on some systems
- If crashes occur, downgrade to ROCm 6.4.1 or 6.2.4
- RX 6900 XT is **not officially supported** by AMD - use workarounds

### GPU Stability Context
- System may experience amdgpu driver page faults (unrelated to this project)
- OBS hardware encoding may conflict - monitor during testing
- Avoid excessive GPU memory allocation that could trigger hangs

### Camera Constraints
- Cam Link 4K: max 4K@30fps or 1080p@60fps
- Wide-angle lens = face is small in frame → need robust detection
- Lighting variations in typical streaming setup

### OBS Integration
- OBS must capture virtual camera as "Video Capture Device (V4L2)"
- No plugin installation needed - pure video source replacement
- Face Tracker plugin should be **DISABLED** when using this solution
- Use OBS 32.x+ (31.x had v4l2loopback bugs)

---

## File Locations

```
Project Directory: ~/Projects/rocm-face-tracker/
├── src/
│   ├── detector.py      # Face detection module (YOLO-Face)
│   ├── tracker.py       # Smoothing/tracking logic
│   ├── cropper.py       # Frame cropping/zooming
│   └── main.py          # Pipeline orchestration
├── models/              # Pre-trained weights (yolov11n-face.pt, etc.)
├── config.yaml          # Runtime configuration
├── requirements.txt
├── Dockerfile           # Optional containerized deployment
└── systemd/
    └── rocm-facetracker.service
```

---

## Reference Commands

```bash
# Test camera access
ffplay -f v4l2 -i /dev/video0

# Test virtual camera output
ffmpeg -re -f lavfi -i testsrc=size=1920x1080:rate=30 \
  -pix_fmt yuv420p -f v4l2 /dev/video10

# Monitor GPU during development
watch -n 1 rocm-smi

# Check for GPU memory leaks
rocm-smi --showmeminfo vram

# Check current CPU usage of OBS
ps aux | grep obs

# Monitor OBS CPU in real-time
top -p $(pgrep -d',' obs)
```

---

## Reference Links

- **YOLO-Face:** https://github.com/akanametov/yolo-face
- **pyvirtualcam:** https://github.com/letmaik/pyvirtualcam
- **v4l2loopback:** https://github.com/umlaeute/v4l2loopback
- **ROCm PyTorch:** https://pytorch.org/get-started/locally/
- **AMD-GPU-BOOST:** https://github.com/Painter3000/AMD-GPU-BOOST
- **obs-face-tracker:** https://github.com/norihiro/obs-face-tracker (CPU-only, for reference)
