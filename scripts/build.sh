#!/usr/bin/env bash
# ==============================================================================
# Bharat AI-SoC PS5: Native Engine Ahead-Of-Time (AOT) Build Script
# Target: Raspberry Pi 5 (Ubuntu 24.04 LTS arm64 / Debian 12 Bookworm)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo " BHARAT AI-SOC PS5: COMPILING NATIVE ENGINE (.bin)"
echo " Host Architecture: $(uname -m)"
echo " Compiler: $(g++ --version | head -n 1)"
echo "=================================================================="

# 1. Install build tools and OpenCV C++ dev libraries if missing
if ! command -v g++ &>/dev/null || ! command -v make &>/dev/null || [ ! -d "/usr/include/opencv4" ]; then
    echo "[1/3] Installing build-essential and libopencv-dev..."
    sudo apt-get update -y
    sudo apt-get install -y build-essential libgomp1 libopencv-dev
else
    echo "[1/3] Build tools and OpenCV verified (g++, make, libopencv-dev available)."
fi

# 2. Check Hailo PCIe dependencies
echo "[2/3] Checking Hailo PCIe runtime..."
if [ -d "/usr/include/hailort" ]; then
    echo " -> HailoRT C++ headers found at /usr/include/hailort"
else
    echo " -> Notice: /usr/include/hailort not found. Compiling in portable fallback mode."
fi

# 3. Compile native binary via Makefile
echo "[3/3] Compiling native binary bin/bharat_qc_native.bin..."
make clean
make -j"$(nproc)"

echo ""
echo "=================================================================="
echo " Binary Build Verification:"
ls -lh bin/bharat_qc_native.bin
file bin/bharat_qc_native.bin || true
echo "=================================================================="
echo " Launch the native engine:"
echo "     ./bin/bharat_qc_native.bin"
echo "=================================================================="
