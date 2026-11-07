#!/usr/bin/env bash
# ==============================================================================
# Bharat AI-SoC PS5: Native ARM64 ExecuTorch Llama Runner Builder for RPi 5
# Architecture: arm64 / aarch64 (Raspberry Pi 5 Cortex-A76 Quad Core)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo " Building Native ARM64 ExecuTorch Llama Runner for Raspberry Pi 5"
echo " Architecture: $(uname -m)"
echo "=================================================================="

# 1. Verify ARM64 architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    echo "Notice: Detected architecture is '$ARCH' (expected aarch64 on RPi 5)."
    echo "Continuing build for host architecture..."
fi

# 2. Install build dependencies
echo "[1/4] Checking build dependencies (cmake, g++, git, python3-dev)..."
sudo apt-get update -y
sudo apt-get install -y cmake build-essential python3-dev git libgomp1

# 3. Clone or update ExecuTorch
EXECUTORCH_DIR="$HOME/executorch"
if [ ! -d "$EXECUTORCH_DIR" ]; then
    echo "[2/4] Cloning ExecuTorch into $EXECUTORCH_DIR..."
    git clone --recurse-submodules -b main https://github.com/pytorch/executorch.git "$EXECUTORCH_DIR"
else
    echo "[2/4] Using existing ExecuTorch directory at $EXECUTORCH_DIR..."
fi

cd "$EXECUTORCH_DIR"

# 4. Install requirements
echo "[3/4] Installing ExecuTorch prerequisites..."
./install_requirements.sh || true

# 5. Build llama_main with XNNPACK ARM NEON optimizations
echo "[4/4] Compiling llama_main for ARM Cortex-A76 with 4 CPU threads..."
cmake -DPYTHON_EXECUTABLE=python3 \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
      -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
      -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
      -Bcmake-out .

cmake --build cmake-out -j4 --target llama_main

# 6. Copy binary to deployment folder
LLAMA_BIN="$EXECUTORCH_DIR/cmake-out/examples/models/llama/llama_main"
if [ -f "$LLAMA_BIN" ]; then
    cp "$LLAMA_BIN" "$SCRIPT_DIR/llama_main"
    chmod +x "$SCRIPT_DIR/llama_main"
    echo ""
    echo "=================================================================="
    echo " SUCCESS: Native ARM64 llama_main successfully built & installed!"
    echo " Installed to: $SCRIPT_DIR/llama_main"
    echo " Size: $(ls -lh "$SCRIPT_DIR/llama_main" | awk '{print $5}')"
    echo "=================================================================="
else
    echo "ERROR: Build completed but binary was not found at $LLAMA_BIN"
    exit 1
fi
