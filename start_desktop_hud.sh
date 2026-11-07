#!/usr/bin/env bash
# ==============================================================================
# Bharat AI-SoC PS5: Raspberry Pi 5 Desktop SCADA HUD Launcher
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

echo "=================================================================="
echo " BHARAT AI-SoC PS5: INDUSTRIAL EDGE QC DESKTOP HUD"
echo " Raspberry Pi 5 · Ubuntu Desktop 24.04 LTS · 13 TOPS AI HAT"
echo " Native ARM64 (aarch64) Architecture · Zero WSL Required"
echo "=================================================================="

# Run the native desktop OpenCV / Pillow SCADA GUI
python3 rpi5_qc_assistant.py

