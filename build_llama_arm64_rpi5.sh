#!/usr/bin/env bash
# ==============================================================================
# Bharat AI-SoC PS5: Native ExecuTorch Llama Runner Builder
# Compatible with Raspberry Pi 5 (ARM64 Cortex-A76) and Ubuntu 22.04 / 24.04
# Auto-configures Python >= 3.11 and CMake >= 3.24
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo " Building Native ExecuTorch Llama Runner"
echo " Host Architecture: $(uname -m)"
echo " OS: $(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')"
echo "=================================================================="

# 1. Architecture Check
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo " Target: Native ARM64 (Raspberry Pi 5 Cortex-A76)"
else
    echo " Notice: Host is '$ARCH' (Not ARM64). The resulting binary will be $ARCH."
    echo " If building for Raspberry Pi 5, run this script directly on the Pi 5."
fi

# 2. Check and Resolve Python Version (ExecuTorch requires Python >= 3.11)
echo ""
echo "[1/5] Checking Python version..."
PY_CMD=""
for cand in python3.12 python3.11 python3; do
    if command -v "$cand" &>/dev/null; then
        V_OK=$("$cand" -c 'import sys; print(1 if (sys.version_info.major == 3 and sys.version_info.minor >= 11 and sys.version_info.minor < 16) else 0)' 2>/dev/null || echo 0)
        if [ "$V_OK" = "1" ]; then
            PY_CMD="$cand"
            break
        fi
    fi
done

if [ -z "$PY_CMD" ]; then
    echo " Current system Python is older than 3.11 (ExecuTorch requires >=3.11,<3.16)."
    echo " Installing Python 3.11 via deadsnakes PPA for Ubuntu..."
    sudo apt-get update -y
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
    PY_CMD="python3.11"
fi
echo "Using Python: $($PY_CMD --version)"

# 3. Base Build Dependencies
echo ""
echo "[2/5] Checking system build tools (build-essential, git, libgomp1)..."
sudo apt-get update -y
sudo apt-get install -y build-essential git libgomp1

# 4. Setup Dedicated Build Virtualenv with modern CMake (>= 3.28)
echo ""
echo "[3/5] Setting up build environment with modern CMake (>= 3.24 required)..."
BUILD_VENV="$SCRIPT_DIR/.build_venv"
if [ ! -d "$BUILD_VENV" ]; then
    "$PY_CMD" -m venv "$BUILD_VENV"
fi

source "$BUILD_VENV/bin/activate"
pip install --upgrade pip wheel setuptools
pip install "cmake>=3.28"

export PATH="$BUILD_VENV/bin:$PATH"
export PYTHON_EXECUTABLE="$(which python3)"
echo "Using CMake: $(cmake --version | head -n 1)"

# 5. Clone or Update ExecuTorch
echo ""
echo "[4/5] Preparing ExecuTorch repository..."
EXECUTORCH_DIR="$HOME/executorch"
if [ ! -d "$EXECUTORCH_DIR" ]; then
    echo "Cloning ExecuTorch into $EXECUTORCH_DIR..."
    git clone --recurse-submodules -b main https://github.com/pytorch/executorch.git "$EXECUTORCH_DIR"
else
    echo "Using existing ExecuTorch directory at $EXECUTORCH_DIR"
fi

cd "$EXECUTORCH_DIR"

# Install ExecuTorch python requirements using our Python 3.11+ venv
echo "Installing ExecuTorch prerequisites..."
./install_requirements.sh || true

# 6. Compile llama_main with XNNPACK optimizations
echo ""
echo "[5/5] Compiling llama_main with XNNPACK optimized quantized kernels..."
NUM_JOBS=$(nproc 2>/dev/null || echo 4)
cmake -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
      -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
      -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
      -Bcmake-out .

cmake --build cmake-out -j"$NUM_JOBS" --target llama_main

# 7. Copy Binary to Deployment Directory
LLAMA_BIN="$EXECUTORCH_DIR/cmake-out/examples/models/llama/llama_main"
if [ -f "$LLAMA_BIN" ]; then
    cp "$LLAMA_BIN" "$SCRIPT_DIR/llama_main"
    chmod +x "$SCRIPT_DIR/llama_main"
    echo ""
    echo "=================================================================="
    echo " SUCCESS: Native llama_main successfully built & installed!"
    echo " Binary Location: $SCRIPT_DIR/llama_main"
    echo " Architecture:    $ARCH"
    echo " Binary Info:      $(file "$SCRIPT_DIR/llama_main" 2>/dev/null || ls -lh "$SCRIPT_DIR/llama_main")"
    echo "=================================================================="
else
    echo "ERROR: Build completed but binary was not found at $LLAMA_BIN"
    exit 1
fi
