# Bharat AI-SoC PS5: Edge Native C++ Inference Engine

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205%20(ARM64)-C51A4A)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![NPU Accelerator](https://img.shields.io/badge/NPU-Hailo--8L%2013%20TOPS%20PCIe-007ACC)](https://hailo.ai/products/ai-accelerators/hailo-8l-ai-accelerator-entry-level-m-2/)
[![C++ Standard](https://img.shields.io/badge/Standard-C%2B%2B17%20Compiled-00599C)](https://isocpp.org/)
[![SIMD](https://img.shields.io/badge/SIMD-ARM%20NEON%20Intrinsics-ED7D31)](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)


A high-performance, ahead-of-time (AOT) compiled native C++ industrial edge defect detection and advisory engine designed for the **Raspberry Pi 5** running **Ubuntu 24.04 LTS arm64 (aarch64)**.

---

## 1. System Engineering & Performance Architecture

| Architectural Design Pillar | Native Implementation Details |
| :--- | :--- |
| **Pure Ahead-Of-Time (AOT) Compilation** | The entire system compiles into a single, self-contained ELF binary (`bin/bharat_qc_native.bin`) via GNU `Makefile` with `-O3 -march=armv8.2-a+fp16 -mtune=cortex-a76 -fopenmp -pthread`. **Zero Python runtime processes.** |
| **Deterministic Core Pinning (`pthread_setaffinity_np`)** | Explicit hardware thread mapping across physical Cortex-A76 Cores 0–3 eliminates OS context-switching overhead and cache invalidation. |
| **Warm In-Memory Resident Llama 3.2 1B** | On-device INT4 Llama 3.2 1B model loaded once into RAM at boot (~774 MiB flat RSS). Persistent stream interface eliminates per-request model reload overhead, yielding 18–22.4 tok/s in 2.1–3.4s. |
| **Dual-Pipeline Computer Vision Dispatch** | Autonomous Stage 0 routing (`stage0_selector_224.onnx`): routes steel strips to **Mode 1** (YOLO11n-416 + NEON Metallic Gate) and fastener components to **Mode 2** (EfficientAD-S + sub-type classifier). |
| **Integrated Web SCADA Workstation (Port 5000)** | Fully embedded web station powered by `cpp-httplib` and `nlohmann/json`. Serves the dark-mode SCADA kiosk, live 720p vector HUD stream, playlist navigation, and CSV audit logging. |

---

## 2. Hardware Core Topology on Raspberry Pi 5

```text
Raspberry Pi 5 Quad-Core ARM Cortex-A76 @ 2.4 GHz:
┌─────────────────────────────────────┬─────────────────────────────────────┐
│  CORE 0 (Affinity Mask: 0x01)       │  CORE 1 (Affinity Mask: 0x02)       │
│  - Thread: "scada_http_c0"          │  - Thread: "vision_worker_c1"       │
│  - cpp-httplib Web Server (Port 5000│  - Stage 0 Routing (224x224 ONNX)   │
│  - Live 720p JPEG Frame Streamer    │  - Mode 1: YOLO11n-416 (Hailo / CPU)│
│  - Interactive API & Playlist State │  - Mode 2: EfficientAD-S (ONNX)     │
│  - SCADA CSV Audit Logbook I/O      │  - ARM NEON SIMD Metallic Gate      │
├─────────────────────────────────────┼─────────────────────────────────────┤
│  CORE 2 (Affinity Mask: 0x04)       │  CORE 3 (Affinity Mask: 0x08)       │
│  - Thread: "llama_resident"         │  - Thread: "llama_resident" (Worker)│
│  - Resident Llama Leader Thread     │  - OpenMP SIMD Tensor Math          │
│  - Warm Memory Flatbuffer (~774 MiB)│  - INT4 Matrix-Vector Multiply      │
│  - Token Streaming & Observation    │  - Dynamic Context Decoding         │
└─────────────────────────────────────┴─────────────────────────────────────┘
```

---

## 3. Directory Layout

```text
native/
├── Makefile                     # Production GNU Makefile targeting Cortex-A76 NEON
├── bin/                         # Compiled target binary location
│   ├── bharat_qc_native.bin     # Main pure AOT executable (served on port 5000)
│   ├── llama_daemon             # ExecuTorch resident runner binary
│   └── libllama_runner.so       # Optimized Llama 3.2 1B runtime shared library
├── config/
│   └── edge_qc_config.json      # Camera indices, thresholds, and core topology map
├── include/
│   ├── cpu_affinity.hpp         # POSIX thread core pinning (Cores 0, 1, 2, 3)
│   ├── embedded_ui.hpp          # Build-time embedded SCADA Web UI header
│   ├── hailo_vision_engine.hpp  # Dual-mode vision pipeline (Hailo-8L / ORT)
│   ├── ipc_server.hpp           # Native C++ HTTP Web Station (Port 5000)
│   ├── llama_resident.hpp       # Warm in-memory resident ExecuTorch engine
│   ├── metallic_gate_neon.hpp   # 128-bit ARM NEON SIMD metallic surface gate
│   ├── qc_rules.hpp             # Industrial severity & root-cause scoring rules
│   └── ring_buffer.hpp          # Zero-copy circular frame buffer queue
├── models/                      # Vision HEF, ONNX models, and tokenizer
│   ├── Llama-3.2-1B-UnifiedQC-FT-INT4.pte
│   ├── mvtec_efficientad_s.onnx
│   ├── stage0_selector_224.onnx
│   ├── tokenizer.model
│   ├── yolo11n_416_v6.hef
│   └── yolo11n_416_v6.onnx
├── sample_test_images/          # Official flat test suite (15 test cases)
│   ├── test_00_clean_steel.jpg
│   ├── test_bent_1.png
│   ├── test_bent_2.png
│   ├── test_color_1.png
│   ├── test_color_2.png
│   ├── test_crazing.jpg
│   ├── test_flip_1.png
│   ├── test_good_1.png
│   ├── test_good_2.png
│   ├── test_inclusion.jpg
│   ├── test_patches.jpg
│   ├── test_pitted_surface.jpg
│   ├── test_rolled-in_scale.jpg
│   ├── test_scratch_1.png
│   └── test_scratches.jpg
├── scripts/
│   ├── bharat-qc.service        # systemd production daemon unit
│   ├── build.sh                 # 1-click build script
│   └── run_native.sh            # 1-click launcher with LD_LIBRARY_PATH
├── src/
│   ├── cpu_affinity.cpp         # Hardware core binding implementations
│   ├── embedded_ui.cpp          # Byte-for-byte embedded index.html string
│   ├── hailo_vision_engine.cpp  # Dual-mode inference, NMS, and 720p HUD drawing
│   ├── ipc_server.cpp           # HTTP server endpoints and state serializer
│   ├── llama_resident.cpp       # In-process resident daemon management
│   ├── main.cpp                 # Multi-core orchestrator and telemetry loop
│   ├── metallic_gate_neon.cpp   # Vectorized HSV saturation calculations
│   └── qc_rules.cpp             # Severity classification rules
├── templates/
│   └── index.html               # Industrial SCADA UI template
├── third_party/
│   ├── httplib.h                # Header-only C++ HTTP/HTTPS server
│   ├── json.hpp                 # Header-only JSON serializer
│   └── onnxruntime/             # Prebuilt ONNX Runtime headers & libs (ARM64 / x86)
├── .gitignore                   # Git exclusion rules
├── README.md                    # Technical documentation
└── rpi5_qc_inspection_log.csv   # Real-time CSV audit inspection logbook
```

---

## 4. Quickstart Guide (Raspberry Pi 5 / Ubuntu 24.04 LTS arm64)

### Prerequisites
```bash
sudo apt-get update
sudo apt-get install -y build-essential libopencv-dev
```

### Step 1: Compile Native Binary
```bash
cd native
make -j$(nproc)
```
*(Produces the ahead-of-time binary `bin/bharat_qc_native.bin`)*

### Step 2: Run Native Edge Engine
```bash
bash scripts/run_native.sh
```
Or run directly:
```bash
LD_LIBRARY_PATH=bin:third_party/onnxruntime/lib:$LD_LIBRARY_PATH ./bin/bharat_qc_native.bin
```

### Step 3: Access Web SCADA Station
Open your web browser at:
**`http://localhost:5000`** *(or `http://<raspberry-pi-ip>:5000` from any local machine)*.

---

## 5. Web SCADA API Endpoints

All 13 standard endpoints are implemented natively with zero external server dependencies:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the embedded dark SCADA control-room interface |
| `GET` | `/api/state` | Returns live JSON telemetry (defect, confidence, area, boxes, tokens, tok/s, hardware metrics) |
| `GET` | `/api/frame.jpg` | Returns the latest annotated 720p HUD JPEG frame |
| `GET` | `/api/stream` | Continuous MJPEG stream endpoint |
| `GET` | `/api/select_item?path=...` | Selects a specific test sample from the playlist |
| `GET` | `/api/next` / `/api/prev` | Playlist stepping navigation |
| `GET` | `/api/switch_mode?mode=...` | Switches modes: `DEMO`, `IMAGE`, `VIDEO`, `CAMERA`, `LOG` |
| `GET` | `/api/set_pipeline_mode?mode=...` | Sets pipeline: `AUTO`, `MODE_1`, `MODE_2` |
| `GET` | `/api/toggle_pause` | Pauses / Resumes conveyor or media streaming |
| `POST`| `/api/upload` | Multipart file upload for instant evaluation |
| `POST`| `/api/save_frame` | Saves the annotated frame to `snapshots/` |
| `GET` | `/api/logs` | Fetches JSON array of historical inspection records |
| `POST`| `/api/clear_logs` | Clears historical records and resets CSV |
| `GET` | `/api/download_csv` | Downloads `rpi5_qc_inspection_log.csv` attachment |

---

## 6. Production Deployment (systemd)

To configure the engine as a persistent system service with auto-restart on boot:
```bash
sudo cp scripts/bharat-qc.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bharat-qc.service
```
