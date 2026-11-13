#!/usr/bin/env bash
# ==============================================================================
# Bharat AI-SoC PS5: Native Engine Launcher
# Target: Raspberry Pi 5 (Quad-Core Cortex-A76 @ 2.4 GHz)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "bin/bharat_qc_native.bin" ]; then
    echo "Binary bin/bharat_qc_native.bin not found. Building now..."
    bash scripts/build.sh
fi

# Set shared library path for Hailo PCIe runtime, architecture-matched ONNX Runtime, and ExecuTorch
ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    ORT_LIB="${SCRIPT_DIR}/third_party/onnxruntime-linux-aarch64-1.20.1/lib"
else
    ORT_LIB="${SCRIPT_DIR}/third_party/onnxruntime-linux-x64-1.20.1/lib"
fi
export LD_LIBRARY_PATH="${SCRIPT_DIR}/bin:${ORT_LIB}:/usr/lib:/usr/local/lib:${LD_LIBRARY_PATH}"

echo "=================================================================="
echo " Launching Bharat AI-SoC PS5 Native Engine (.bin)"
echo " Thread Topology: 4 Hardware Cores Pinned"
echo " Web SCADA Workstation URL: http://localhost:5000"
echo "=================================================================="

exec ./bin/bharat_qc_native.bin "$@"
