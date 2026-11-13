# ==============================================================================
# BHARAT AI-SOC CHALLENGE 2026-27 | PROBLEM STATEMENT 5
# NATIVE AHEAD-OF-TIME (AOT) C++ ENGINE MAKEFILE
# Target Platform: Raspberry Pi 5 (Quad-Core Cortex-A76 @ 2.4 GHz, Ubuntu 24.04 ARM64)
# ==============================================================================

CXX      := g++
ARCH     := $(shell uname -m)

# Base compiler flags with C++17, OpenMP and includes
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -pthread -fopenmp \
            -Iinclude -Ithird_party -Ithird_party/onnxruntime/include \
            -I/usr/include/opencv4

# ARM Cortex-A76 Hardware SIMD NEON flags for Raspberry Pi 5
ifeq ($(ARCH), aarch64)
    CXXFLAGS += -march=armv8.2-a+fp16 -mtune=cortex-a76 -D__ARM_NEON
    ORT_DIR := third_party/onnxruntime-linux-aarch64-1.20.1
else ifeq ($(ARCH), arm64)
    CXXFLAGS += -march=armv8.2-a+fp16 -mtune=cortex-a76 -D__ARM_NEON
    ORT_DIR := third_party/onnxruntime-linux-aarch64-1.20.1
else
    $(info [Notice] Host architecture is $(ARCH). Compiling with host optimizations.)
    ORT_DIR := third_party/onnxruntime-linux-x64-1.20.1
endif
ORT_LIB_DIR := $(ORT_DIR)/lib

# Check for HailoRT installation on host (Raspberry Pi 5 with AI HAT)
ifneq ($(wildcard /usr/include/hailort/*),)
    CXXFLAGS += -I/usr/include/hailort -DHAVE_HAILORT
    LDFLAGS  += -lhailort
endif

# OpenCV Libraries
LDFLAGS  += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn -lopencv_videoio

# ONNX Runtime Library
LDFLAGS  += -L$(ORT_LIB_DIR) -lonnxruntime \
            -Wl,-rpath,'$$ORIGIN/../$(ORT_LIB_DIR)' \
            -Wl,-rpath,'$$ORIGIN/$(ORT_LIB_DIR)'

LDFLAGS  += -lpthread -lrt

BIN_DIR  := bin
OBJ_DIR  := obj
SRC_DIR  := src

TARGET   := $(BIN_DIR)/bharat_qc_native.bin

SRCS     := $(wildcard $(SRC_DIR)/*.cpp)
OBJS     := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Default target
all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	@echo "=================================================================="
	@echo " Linking Ahead-Of-Time Native Binary: $(TARGET)"
	@echo " Architecture: $(ARCH)"
	@echo "=================================================================="
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)
	@chmod +x $(TARGET)
	@echo ""
	@echo ">>> SUCCESS: Native binary compiled to $(TARGET) <<<"
	@echo "To launch native engine:"
	@echo "    ./$(TARGET)"
	@echo "=================================================================="

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo " [CXX] $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

clean:
	@echo "Cleaning native build artifacts..."
	@rm -rf $(OBJ_DIR) $(BIN_DIR)/bharat_qc_native.bin

run: $(TARGET)
	@./$(TARGET)

install: $(TARGET)
	@cp $(TARGET) /usr/local/bin/bharat_qc_native.bin 2>/dev/null || true
	@echo "Installed to /usr/local/bin/bharat_qc_native.bin"

.PHONY: all clean run install
