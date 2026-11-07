#!/usr/bin/env bash
# ==============================================================================
# Bharat AI-SoC PS5: Raspberry Pi 5 Web QC Workstation Launcher (localhost:5000)
# ==============================================================================
cd "$(dirname "$0")"

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$HOME/executorch/.venv" ]; then
    source "$HOME/executorch/.venv/bin/activate"
fi

# Set shared library path for native ARM64 ExecuTorch and Hailo runtime
export LD_LIBRARY_PATH="$(pwd):$(pwd)/models:${LD_LIBRARY_PATH}"

# Get IP address for remote access
PI_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$PI_IP" ]; then
    PI_IP="localhost"
fi

echo "=================================================================="
echo " BHARAT AI-SoC PS5: INDUSTRIAL EDGE QC WORKSTATION"
echo " Raspberry Pi 5 · Ubuntu Desktop 24.04 LTS · 13 TOPS AI HAT"
echo " Native ARM64 (aarch64) Architecture · Zero WSL Required"
echo "=================================================================="
echo " Starting Web Station on port 5000..."
echo ""
echo " -> Open locally on Pi:      http://localhost:5000"
echo " -> Open from Laptop / LAN:  http://${PI_IP}:5000"
echo "=================================================================="
echo ""

# Run the workstation web server (0.0.0.0 enables local & LAN access)
python3 rpi5_qc_web_station.py

