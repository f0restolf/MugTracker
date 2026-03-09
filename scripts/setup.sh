#!/bin/bash
# ROCm Face Tracker - Setup Script
# Run this on your Nobara Linux system to set up the environment

set -e  # Exit on error

echo "=== ROCm Face Tracker Setup ==="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Nobara/Fedora
if [ ! -f /etc/nobara-release ] && [ ! -f /etc/fedora-release ]; then
    echo -e "${YELLOW}Warning: This script is designed for Nobara/Fedora Linux${NC}"
fi

# 1. Set up environment variables
echo "Step 1: Setting up environment variables..."
ENV_VARS='
# ROCm Face Tracker environment variables (RDNA2)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export ROCM_PATH=/opt/rocm
'

if ! grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.bashrc 2>/dev/null; then
    echo "$ENV_VARS" >> ~/.bashrc
    echo -e "${GREEN}✓ Environment variables added to ~/.bashrc${NC}"
else
    echo -e "${YELLOW}  Environment variables already in ~/.bashrc${NC}"
fi

# Source for current session
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export ROCM_PATH=/opt/rocm

# 2. Check ROCm installation
echo ""
echo "Step 2: Checking ROCm installation..."
if command -v rocminfo &> /dev/null; then
    GFX_VER=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1)
    echo -e "${GREEN}✓ ROCm found: $GFX_VER${NC}"
else
    echo -e "${RED}✗ ROCm not found${NC}"
    echo "  Install ROCm: https://rocm.docs.amd.com/en/latest/"
    exit 1
fi

# 3. Create virtual environment
echo ""
echo "Step 3: Creating Python virtual environment..."
VENV_PATH="$HOME/venvs/facetrack"
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✓ Virtual environment created: $VENV_PATH${NC}"
else
    echo -e "${YELLOW}  Virtual environment already exists${NC}"
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# 4. Install PyTorch with ROCm
echo ""
echo "Step 4: Installing PyTorch with ROCm 6.2.4..."
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/rocm6.2.4

# Verify PyTorch
python3 -c "import torch; print('PyTorch:', torch.__version__); print('ROCm available:', torch.cuda.is_available())"

# 5. Install other dependencies
echo ""
echo "Step 5: Installing dependencies..."
pip install opencv-python numpy pyyaml pyvirtualcam ultralytics

# 6. Clone YOLO-Face and get weights
echo ""
echo "Step 6: Setting up YOLO-Face..."
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d "yolo-face" ]; then
    git clone https://github.com/akanametov/yolo-face
    echo -e "${GREEN}✓ YOLO-Face cloned${NC}"
else
    echo -e "${YELLOW}  YOLO-Face already exists${NC}"
fi

# 7. Set up v4l2loopback
echo ""
echo "Step 7: Setting up v4l2loopback..."

# Check if module is available
if modinfo v4l2loopback &> /dev/null; then
    echo -e "${GREEN}✓ v4l2loopback module available${NC}"
else
    echo -e "${YELLOW}  Installing v4l2loopback...${NC}"
    sudo dnf install -y v4l2loopback
fi

# Create persistent config
sudo tee /etc/modules-load.d/v4l2loopback.conf > /dev/null << 'EOF'
v4l2loopback
EOF

sudo tee /etc/modprobe.d/v4l2loopback.conf > /dev/null << 'EOF'
options v4l2loopback video_nr=10 card_label="FaceTrack" exclusive_caps=1 max_buffers=2
EOF

echo -e "${GREEN}✓ v4l2loopback configured to load on boot${NC}"

# Load module now
if lsmod | grep -q v4l2loopback; then
    echo -e "${YELLOW}  v4l2loopback already loaded${NC}"
else
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="FaceTrack" exclusive_caps=1 max_buffers=2
    echo -e "${GREEN}✓ v4l2loopback loaded${NC}"
fi

# 8. Verify setup
echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the face tracker:"
echo "  source ~/venvs/facetrack/bin/activate"
echo "  cd $PROJECT_DIR"
echo "  python src/main.py"
echo ""
echo "To test virtual camera:"
echo "  python src/output.py"
echo ""
echo "To install systemd service:"
echo "  sudo cp systemd/rocm-facetracker.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable rocm-facetracker"
