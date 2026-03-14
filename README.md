# MugTracker 󰖠

**GPU-accelerated face tracking for OBS — because your CPU has better things to do.**

MugTracker replaces the CPU-hungry [obs-face-tracker](https://github.com/norihiro/obs-face-tracker) plugin with a ROCm-powered pipeline running on your AMD GPU. Face detection runs on the GPU via YOLO-Face, and the cropped/tracked output is piped into OBS through a virtual camera.

> Tested on: AMD RX 6900 XT · Nobara Linux 43 · ROCm 6.2.4 · OBS 32.x

---

## The Problem

`obs-face-tracker` uses dlib under the hood — CPU only. On a Ryzen 7 5800X, that's **~106% CPU** just for face detection while streaming. Not great.

## The Fix

| | Before | After |
|---|---|---|
| **CPU usage** | ~106% | ~33% |
| **GPU usage** | 0% | ~15% |
| **Detection backend** | dlib (CPU) | YOLO-Face (ROCm) |

---

## How It Works

```
Cam Link 4K          ROCm Pipeline          Virtual Camera
/dev/video1    ───▶  YOLO-Face (GPU)  ───▶  /dev/video10
                     EMA smoothing                │
                     Crop & zoom                  ▼
                                              OBS Studio
```

1. Captures from your capture card
2. Runs face detection on the GPU every 3rd frame
3. Applies exponential moving average smoothing to prevent jitter
4. Crops and zooms the frame around your face
5. Writes the output to a v4l2loopback virtual camera
6. OBS reads the virtual camera like any normal video source

---

## Requirements

### Hardware
- AMD GPU with ROCm support (tested on RX 6900 XT / RDNA2)
- Capture card (tested with Elgato Cam Link 4K)

### Software
- Linux with ROCm 6.2.x installed
- Python 3.11 (important — ROCm wheels don't support 3.12+ yet)
- OBS Studio 32.x+
- v4l2loopback kernel module

---

## Installation

### 1. Environment variables (RDNA2 critical)

Add to your `~/.bashrc`:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export ROCM_PATH=/opt/rocm
```

### 2. Python environment

```bash
python3.11 -m venv ~/venvs/facetrack
source ~/venvs/facetrack/bin/activate

pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/rocm6.2.4

pip install opencv-python numpy pyyaml pyvirtualcam ultralytics
```

### 3. Download model weights

```bash
# Weights are pulled automatically on first run from HuggingFace
# Or manually place a yolov11n-face.pt in the models/ directory
```

### 4. Virtual camera setup

```bash
sudo modprobe v4l2loopback \
  devices=2 video_nr=0,10 \
  card_label="OBS,FaceTrack" \
  exclusive_caps=1 max_buffers=2
```

For persistence across reboots:

```bash
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback video_nr=0,10 card_label="OBS,FaceTrack" exclusive_caps=1 max_buffers=2' \
  | sudo tee /etc/modprobe.d/v4l2loopback.conf
```

---

## Usage

```bash
# Set camera format (adjust device as needed)
v4l2-ctl -d /dev/video1 --set-fmt-video=width=1920,height=1080,pixelformat=MJPG

# Activate environment and run
source ~/venvs/facetrack/bin/activate
cd path/to/mugtracker
python src/facetracker.py
```

Then in OBS: add a **Video Capture Device (V4L2)** source and select **FaceTrack** (`/dev/video10`). Disable the obs-face-tracker plugin if you have it.

---

## Configuration

Edit `config.yaml` to tune behaviour:

```yaml
camera:
  input_device: "/dev/video1"
  input_resolution: [1920, 1080]
  fps: 30

output:
  device: "/dev/video10"
  resolution: [1920, 1080]

tracking:
  smoothing: 0.15       # 0.1 = buttery smooth, 0.3 = snappy
  deadzone: 20          # pixels of movement to ignore (prevents micro-jitter)
  zoom_level: 2.0       # how tight the crop is
  detection_interval: 3 # run detection every N frames

model:
  weights: "models/model.pt"
  confidence: 0.5
```

---

## Project Structure

```
mugtracker/
├── src/
│   ├── facetracker.py   # Main pipeline
│   ├── detector.py      # YOLO-Face wrapper
│   ├── tracker.py       # EMA smoothing logic
│   └── cropper.py       # Frame crop/zoom
├── models/              # Model weights (not committed)
├── docs/
│   ├── knowledge-base.md
│   └── implementation-plan.md
├── config.yaml
└── requirements.txt
```

---

## Notes on AMD GPU Support

- The RX 6900 XT is **not officially supported** by AMD ROCm — `HSA_OVERRIDE_GFX_VERSION=10.3.0` is required
- PyTorch reports 40 compute units instead of 80 — this is normal. RDNA2 uses Workgroup Processors (WGPs) which pair CUs; performance is not halved
- MediaPipe/BlazeFace do **not** support AMD GPUs — don't bother
- If you see SIGSEGV crashes, try pinning to ROCm 6.2.4 instead of 6.4.x

---

## License

This project is licensed under the AGPL-3.0 License — see the [LICENSE](LICENSE) file for details.

---

## Now THIS is why RAM is 1200 buckaroonies

I dont know how to code much, but I needed a fix. Now if you need it, you can use it too!

This can be adapted for other GPU acceleration "make the 69000 XT ROCm nice" necessities

## Acknowledgements

- [akanametov/yolo-face](https://github.com/akanametov/yolo-face) — YOLO-Face model
- [letmaik/pyvirtualcam](https://github.com/letmaik/pyvirtualcam) — virtual camera output
- [umlaeute/v4l2loopback](https://github.com/umlaeute/v4l2loopback) — kernel module
- [norihiro/obs-face-tracker](https://github.com/norihiro/obs-face-tracker) — the plugin this replaces
