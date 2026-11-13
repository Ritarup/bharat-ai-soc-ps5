#!/usr/bin/env bash
# ==============================================================================
# Bharat AI-SoC PS5: Native ExecuTorch Llama Daemon Builder (ARM64 / aarch64)
# Builds both `bin/llama_daemon` and `bin/libllama_runner.so` natively for the
# host architecture (Raspberry Pi 5 Cortex-A76 / Azure ARM64 / x86_64).
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

ARCH="$(uname -m)"
echo "=================================================================="
echo " Building Native ExecuTorch Resident Daemon (llama_daemon)"
echo " Host Architecture: $ARCH"
echo "=================================================================="

# 1. Resolve Python (>= 3.11, < 3.16 required by ExecuTorch)
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
    echo "[1/5] Installing Python 3.12 for ExecuTorch build..."
    sudo apt-get update -y
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    PY_CMD="python3.12"
fi
echo "[1/5] Using Python: $($PY_CMD --version)"

# 2. System build tools
echo "[2/5] Installing build dependencies..."
sudo apt-get update -y
sudo apt-get install -y build-essential git libgomp1 ninja-build

# 3. Build venv
echo "[3/5] Preparing Python build environment..."
BUILD_VENV="$HOME/.llama_build_venv"
if [ ! -d "$BUILD_VENV" ]; then
    "$PY_CMD" -m venv "$BUILD_VENV"
fi
source "$BUILD_VENV/bin/activate"
pip install --upgrade pip wheel setuptools
pip install "cmake>=3.28" torch

export PATH="$BUILD_VENV/bin:$PATH"
export PYTHON_EXECUTABLE="$(which python3)"

# 4. Clone ExecuTorch at verified commit
EXECUTORCH_DIR="$HOME/executorch"
VERIFIED_COMMIT="d07c72ba19e9f6adc2e186c4c5f322db690a0251"
if [ ! -d "$EXECUTORCH_DIR" ]; then
    echo "[4/5] Cloning ExecuTorch into $EXECUTORCH_DIR..."
    git clone https://github.com/pytorch/executorch.git "$EXECUTORCH_DIR"
    cd "$EXECUTORCH_DIR"
    git checkout "$VERIFIED_COMMIT"
    git submodule sync --recursive
    git submodule update --init --recursive
else
    echo "[4/5] Using existing ExecuTorch at $EXECUTORCH_DIR"
    cd "$EXECUTORCH_DIR"
fi

./install_requirements.sh || true

# Copy daemon.cpp and wire CMake target
cp "$SCRIPT_DIR/scripts/llama_daemon.cpp" "$EXECUTORCH_DIR/examples/models/llama/daemon.cpp"
if ! grep -q "add_executable(llama_daemon" "$EXECUTORCH_DIR/examples/models/llama/CMakeLists.txt"; then
    cat <<'EOF' >> "$EXECUTORCH_DIR/examples/models/llama/CMakeLists.txt"

add_executable(llama_daemon daemon.cpp)
target_include_directories(llama_daemon PUBLIC ${_common_include_directories})
target_link_libraries(llama_daemon PUBLIC llama_runner ${link_libraries})
target_compile_options(llama_daemon PUBLIC ${_common_compile_options})
if(UNIX)
  set_target_properties(llama_daemon PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'")
endif()
EOF
fi

# 5. Build core ExecuTorch + llama_daemon
echo "[5/5] Compiling ExecuTorch core & llama_daemon (this may take several minutes)..."
NUM_JOBS="$(nproc 2>/dev/null || echo 2)"

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

cmake --build cmake-out -j"$NUM_JOBS" --target install

cmake -Bcmake-out/examples/models/llama \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_FIND_ROOT_PATH="$EXECUTORCH_DIR/cmake-out" \
      -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      examples/models/llama

cmake --build cmake-out/examples/models/llama -j"$NUM_JOBS" --target llama_daemon

mkdir -p "$SCRIPT_DIR/bin"
cp cmake-out/examples/models/llama/llama_daemon "$SCRIPT_DIR/bin/llama_daemon"
chmod +x "$SCRIPT_DIR/bin/llama_daemon"

# Copy libllama_runner.so if built as a shared library
LIB_RUNNER=$(find cmake-out/examples/models/llama -name "libllama_runner.so*" | head -n 1 || true)
if [ -n "$LIB_RUNNER" ] && [ -f "$LIB_RUNNER" ]; then
    cp "$LIB_RUNNER" "$SCRIPT_DIR/bin/libllama_runner.so"
    chmod +x "$SCRIPT_DIR/bin/libllama_runner.so"
else
    rm -f "$SCRIPT_DIR/bin/libllama_runner.so"
fi

echo ""
echo "=================================================================="
echo " SUCCESS: Native llama_daemon built for $ARCH!"
file "$SCRIPT_DIR/bin/llama_daemon" || true
ls -lh "$SCRIPT_DIR/bin/"
echo "=================================================================="
