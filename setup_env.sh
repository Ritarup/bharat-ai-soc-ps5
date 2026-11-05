#!/usr/bin/env bash
# ==============================================================================
# Bharat AI-SoC PS5: Raspberry Pi 5 (Ubuntu 24.04 LTS) One-Click Environment Setup
# ==============================================================================
set -e

echo "=================================================================="
echo " Bharat AI-SoC PS5: Raspberry Pi 5 (Ubuntu 24.04 LTS) Setup"
echo " Target Architecture: arm64 (aarch64) | AI HAT: 13 TOPS"
echo "=================================================================="

# 1. Install base system dependencies
echo "[1/4] Checking system packages (python3, venv, OpenCV runtime libs)..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-pip python3-venv libgl1 libglib2.0-0 libgomp1 zenity
fi

# 2. Setup Python virtual environment
echo "[2/4] Setting up Python virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# 3. Install Python requirements
echo "[3/4] Installing Python requirements in .venv..."
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements_rpi5.txt

# 4. Verify Model Files
echo "[4/4] Verifying model integrity..."
python3 -c "
from pathlib import Path
models = [
    'models/yolo11n_416_v6.onnx',
    'models/stage0_selector_224.onnx',
    'models/mvtec_efficientad_s.onnx',
    'models/Llama-3.2-1B-UnifiedQC-FT-INT4.pte',
    'models/tokenizer.model',
    'models/params.json'
]
missing = [m for m in models if not Path(m).exists()]
if missing:
    print('WARNING: Missing models:', missing)
    exit(1)
else:
    print('SUCCESS: All 6 core models verified!')
"

echo ""
echo "=================================================================="
echo " Setup Completed Successfully!"
echo " To run the Web QC Workstation (localhost:5000):"
echo "     bash start_web_station.sh"
echo " To run the Desktop SCADA HUD:"
echo "     bash start_desktop_hud.sh"
echo "=================================================================="

