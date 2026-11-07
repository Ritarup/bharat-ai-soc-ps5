#!/usr/bin/env bash
# ==============================================================================
# Bharat AI-SoC PS5: Native ExecuTorch Llama Runner Builder
# Compatible with Raspberry Pi 5 (ARM64 Cortex-A76) and Ubuntu 22.04 / 24.04
# Auto-configures Python 3.12 / modern CMake and builds llama_main
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo " Building Native ExecuTorch Llama Runner (llama_main)"
echo " Host Architecture: $(uname -m)"
echo " OS: $(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')"
echo "=================================================================="

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo " Target: Native ARM64 (Raspberry Pi 5 Cortex-A76)"
else
    echo " Notice: Host is '$ARCH' (Not ARM64). The resulting binary will be $ARCH."
fi

# 1. Resolve Python (ExecuTorch requires final release >= 3.11, < 3.16; avoids 3.11.0rc1)
echo ""
echo "[1/5] Checking Python interpreter..."
PY_CMD=""
for cand in python3.12 python3.11 python3; do
    if command -v "$cand" &>/dev/null; then
        V_OK=$("$cand" -c 'import sys; print(1 if (sys.version_info >= (3, 11, 0) and sys.version_info < (3, 16, 0) and "rc" not in sys.version) else 0)' 2>/dev/null || echo 0)
        if [ "$V_OK" = "1" ]; then
            PY_CMD="$cand"
            break
        fi
    fi
done

if [ -z "$PY_CMD" ]; then
    echo " Installing Python 3.12 to guarantee clean ExecuTorch compatibility..."
    sudo apt-get update -y
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    PY_CMD="python3.12"
fi
echo "Using Python: $($PY_CMD --version)"

# 2. System build tools
echo ""
echo "[2/5] Checking system build tools (build-essential, git, libgomp1)..."
sudo apt-get update -y
sudo apt-get install -y build-essential git libgomp1

# 3. Dedicated build virtual environment
echo ""
echo "[3/5] Setting up build environment with modern CMake & PyTorch headers..."
BUILD_VENV="$SCRIPT_DIR/.build_venv"
if [ ! -d "$BUILD_VENV" ]; then
    "$PY_CMD" -m venv "$BUILD_VENV"
fi

source "$BUILD_VENV/bin/activate"
pip install --upgrade pip wheel setuptools
pip install "cmake>=3.28" torch

export PATH="$BUILD_VENV/bin:$PATH"
export PYTHON_EXECUTABLE="$(which python3)"
echo "Using CMake: $(cmake --version | head -n 1)"
echo "Using PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"

# 4. Clone or update ExecuTorch
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

# Try installing prerequisites if needed
./install_requirements.sh || true

# 5. Compile with complete LLM preset & dependencies
echo ""
echo "[5/5] Compiling ExecuTorch and llama_main..."
NUM_JOBS=$(nproc 2>/dev/null || echo 4)

# Stage 5A: Configure and build core libraries with LLM preset
cmake -Bcmake-out \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      -DEXECUTORCH_BUILD_PRESET_FILE=tools/cmake/preset/llm.cmake \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
      -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
      -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
      -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
      -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
      -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
      -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
      .

echo "Building and installing core libraries..."
cmake --build cmake-out -j"$NUM_JOBS" --target install

# Stage 5B: Build llama_main runner
echo "Building llama_main binary..."
if [ -d "examples/models/llama" ]; then
    cmake -Bcmake-out/examples/models/llama \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_FIND_ROOT_PATH="$EXECUTORCH_DIR/cmake-out" \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          examples/models/llama
    cmake --build cmake-out/examples/models/llama -j"$NUM_JOBS" --target llama_main
else
    cmake --build cmake-out -j"$NUM_JOBS" --target llama_main
fi

# 6. Copy Binary to Deployment Directory
LLAMA_BIN=""
for cand in \
    "$EXECUTORCH_DIR/cmake-out/examples/models/llama/llama_main" \
    "$EXECUTORCH_DIR/cmake-out/llama_main" \
    "$EXECUTORCH_DIR/cmake-out/bin/llama_main"; do
    if [ -f "$cand" ]; then
        LLAMA_BIN="$cand"
        break
    fi
done

if [ -n "$LLAMA_BIN" ] && [ -f "$LLAMA_BIN" ]; then
    cp "$LLAMA_BIN" "$SCRIPT_DIR/llama_main"
    chmod +x "$SCRIPT_DIR/llama_main"
    echo ""
    echo "=================================================================="
    echo " SUCCESS: Native llama_main successfully built & installed!"
    echo " Binary Location: $SCRIPT_DIR/llama_main"
    echo " Architecture:    $ARCH"
    echo " Binary Info:     $(file "$SCRIPT_DIR/llama_main" 2>/dev/null || ls -lh "$SCRIPT_DIR/llama_main")"
    echo "=================================================================="
else
    echo "ERROR: Build completed but llama_main binary was not located."
    exit 1
fi
