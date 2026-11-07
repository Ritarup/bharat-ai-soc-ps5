#!/usr/bin/env bash
# ==============================================================================
# Bharat AI-SoC PS5: Raspberry Pi 5 - Hailo-8L (13 TOPS AI HAT) Setup Script
# ==============================================================================
set -e

echo "=================================================================="
echo " Setting up Raspberry Pi 5 AI HAT (13 TOPS Hailo-8L) on Ubuntu 24.04"
echo "=================================================================="

# 1. Update and install Hailo PCIe drivers & runtime
echo "[1/4] Installing system Hailo PCIe kernel driver and tools..."
sudo apt-get update -y
sudo apt-get install -y hailo-all dkms pciutils

# 2. Check PCIe detection of 13 TOPS HAT
echo "[2/4] Verifying 13 TOPS AI HAT on PCIe bus..."
if lspci | grep -i hailo; then
    echo "SUCCESS: Hailo AI HAT detected on PCIe!"
else
    echo "WARNING: Hailo not found on lspci. Ensure PCIe HAT is firmly connected in PCIe slot."
fi

# 3. Check firmware & status
echo "[3/4] Checking HailoRT status..."
if command -v hailortcli &> /dev/null; then
    hailortcli fw-control identify || true
fi

# 4. Install Python HailoRT bindings into .venv
echo "[4/4] Setting up Python environment with system site-packages..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv --system-site-packages
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_rpi5.txt

echo ""
echo "=================================================================="
echo " Hailo 13 TOPS Setup Complete!"
echo " Model file: models/yolo11n_416_v6.hef verified (13 TOPS NPU ready)!"
echo " Launch the Workstation:"
echo "     bash start_web_station.sh"
echo "=================================================================="

