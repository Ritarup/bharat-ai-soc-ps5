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

echo "=================================================================="
echo " BHARAT AI-SoC PS5: INDUSTRIAL EDGE QC DESKTOP HUD"
echo " Raspberry Pi 5 · Ubuntu Desktop 24.04 LTS"
echo "=================================================================="

# Run the native desktop OpenCV / Pillow SCADA GUI
python3 rpi5_qc_assistant.py

