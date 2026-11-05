# Bharat AI-SoC Challenge 2026–27 | Problem Statement 5
# Unified Multi-Mode Industrial Edge Quality Control (QC) Assistant on Raspberry Pi 5 + 13 TOPS AI HAT (Hailo-8L)

[![Target Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205%20(8GB)-C51A4A?logo=raspberry-pi&logoColor=white)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![AI Accelerator](https://img.shields.io/badge/NPU-Hailo--8L%2013%20TOPS%20PCIe-007ACC)](https://hailo.ai/products/ai-accelerators/hailo-8l-ai-accelerator-entry-level-m-2/)
[![Vision Core](https://img.shields.io/badge/Vision-YOLO11n%20%7C%20EfficientAD--S-2ECC71)](https://docs.ultralytics.com/models/yolo11/)
[![Edge LLM](https://img.shields.io/badge/GenAI-Llama%203.2%201B%20INT4%20(ExecuTorch)-0467DF)](https://pytorch.org/executorch/)
[![Operating System](https://img.shields.io/badge/OS-Raspberry%20Pi%20OS%20%2F%20Ubuntu%2024.04-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com/download/raspberry-pi)
[![Interface](https://img.shields.io/badge/SCADA-Dark%20Industrial%20Web%20Station-161A1F)](http://localhost:5000)

An end-to-end, multi-mode heterogeneous Edge AI quality control station engineered for the **Raspberry Pi 5** accelerated by the **13 TOPS Hailo-8L AI HAT**. The system pairs ultra-low-latency computer vision (**YOLO11n** compiled to `.hef` on NPU + **EfficientAD-S** on ARM NEON CPU) with on-device Generative AI reasoning (**quantized Llama 3.2 1B INT4 on PyTorch ExecuTorch**) to detect defects in real time and synthesize deterministic, domain-accurate operator remediation advisories without any cloud connectivity.

---

## 🌟 Key Highlights & System Capabilities

- **Heterogeneous Edge Architecture:**
  - Offloads 416×416 **YOLO11n** object detection to the **13 TOPS Hailo-8L AI HAT** over PCIe Gen 3 at **100+ FPS (<10ms latency)** with **~0% CPU load**.
  - Runs **EfficientAD-S** (256×256) on 4-core Cortex-A76 ARM NEON CPU cores at **~40 FPS**, eliminating PCIe context switching overhead.
  - Hosts on-device **Llama 3.2 1B INT4** via **PyTorch ExecuTorch**, generating actionable operator advisories at **~13.8 tok/s** with streaming BPE tokens and live cursor `▌`.
  - **Zero-Crash Fallback:** If the Hailo AI HAT is detached, the pipeline automatically and seamlessly falls back to CPU execution without throwing exceptions.
- **Stage 0 Tiny Classifier Router:**
  - Ultra-fast 224×224 classification network (`stage0_selector_224.onnx`) operating in **1.2 ms** to automatically inspect incoming frames and route between:
    - **Mode 1:** Hot/Cold Rolled Steel Strip Surface Defects (NEU-DET dataset).
    - **Mode 2:** Precision Metal Nut Fastener Anomalies (MVTec AD dataset).
- **Zero-Flicker 4-Frame Temporal Consistency Lock:**
  - In video, live camera, and demo streams, an anomaly must persist for 4 consecutive frames before locking and triggering advisory generation, preventing false-positive conveyor vibrations.
  - In static image mode, locks instantaneously on frame 1 with a single-run analysis latch.
- **Metallic Surface Rejector Gate:**
  - HSV saturation and chromatic filter automatically screens incoming frames to reject human hands, skin tones, faces, and factory background clutter before inference.
- **High-DPI Dark Industrial SCADA Web Station (`localhost:5000`):**
  - Responsive dark SCADA telemetry dashboard with guaranteed viewport headroom clearance.
  - Rotating SVG/CSS preloader spinners next to "Possible Cause" and "Recommended Action" headers during LLM synthesis.
  - Real-time token streaming with live cursor `▌`.
  - Searchable full-screen CSV Inspection Logbook with export and real-time filtering.

---

## 🏗️ Technical Pipeline Architecture

```text
               FACTORY INSPECTION FEED (Camera / Video / Image / Conveyor)
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │ Stage 0: Tiny Classifier (yolo11n-cls 224x224 ONNX) │
             │ Automatically routes steel strips vs mechanical nuts│
             └──────────────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
    ┌───────────────────────────────┐       ┌───────────────────────────────┐
    │     MODE 1: STEEL DEFECTS     │       │    MODE 2: COMPONENT ANOMALY  │
    │          (NEU-DET)            │       │          (MVTec AD)           │
    │   YOLO11n 416x416 Detection   │       │     EfficientAD-S 256x256     │
    │   - Crazing, Inclusion,       │       │   - Bent, Color, Flip,        │
    │     Patches, Pitted Surface,  │       │     Scratch Anomalies         │
    │     Rolled-in Scale, Scratches│       │   - Dual-condition threshold  │
    │   - Dominance Ranking Formula │       │     (T_IMG = 0.35, T_PIX=0.16)│
    │   - 13 TOPS Hailo-8L / CPU    │       │   - ARM NEON CPU Optimized   │
    │   - Metallic Surface Gate     │       │   - Clean Nut Pass Filtering  │
    └───────────────┬───────────────┘       └───────────────┬───────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │ 4-Frame Temporal Lock (or Instant in Image Mode)
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │       STAGE 2: ON-DEVICE LLM ADVISORY ENGINE        │
             │   Llama 3.2 1B INT4 LoRA Fine-Tuned via ExecuTorch  │
             │   - 100% Parametric Knowledge (Zero Cloud / Zero KB)│
             │   - Synthesizes 2-clause industrial advisories:     │
             │     "Possible Cause: ... | Recommended Action: ..." │
             │   - Natural BPE token streaming at ~13.8 tok/s       │
             │   - Writes prompt to current_prompt.txt for audit   │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │    SCADA OPERATOR HUD & CSV INSPECTION LOGBOOK      │
             │ Real-Time MJPEG Stream · Live Metrics · Audit Trail │
             └─────────────────────────────────────────────────────┘
```

---

## 🛠️ Hardware Requirements & AI HAT Installation

### Bill of Materials (BOM)
1. **SBC:** [Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/) (8GB recommended).
2. **NPU:** [Raspberry Pi AI HAT](https://www.raspberrypi.com/products/ai-hat/) with Hailo-8L M.2 module (13 TOPS, PCIe Gen 2/3).
3. **Cooling:** Raspberry Pi Active Cooler (essential for sustained CPU+NPU workloads).
4. **Power Supply:** Official Raspberry Pi 27W USB-C Power Delivery (PD) Power Supply.
5. **Camera (Optional):** Raspberry Pi Camera Module 3 or standard USB UVC Industrial Webcam.

### Enabling PCIe Gen 3 on Raspberry Pi 5
To unlock full 13 TOPS Hailo-8L throughput (PCIe Gen 3 ×1 link at 8 Gbps):

1. Edit the boot firmware configuration:
   ```bash
   sudo nano /boot/firmware/config.txt
   ```
2. Append the following lines under `[all]`:
   ```ini
   # Enable PCIe interface
   dtparam=pciex1
   # Force PCIe Gen 3 speeds (8 GT/s)
   dtparam=pciex1_gen=3
   ```
3. Save and reboot:
   ```bash
   sudo reboot
   ```
4. Verify PCIe link status:
   ```bash
   lspci -nn
   # Output should display:
   # 0000:01:00.0 Co-processor [0b40]: Hailo Technologies Ltd. Hailo-8 AI Processor [1e60:2864] (rev 02)
   ```

---

## ⚡ 1-Click Quickstart Guide

### 1. Automated Setup
Clone this repository and run the setup script:

```bash
git clone https://github.com/your-username/bharat-ai-soc-ps5.git
cd bharat-ai-soc-ps5/bharat_rpi5_deploy_halio

chmod +x setup_hailo_rpi5.sh setup_env.sh start_web_station.sh start_desktop_hud.sh
./setup_hailo_rpi5.sh
```

`setup_hailo_rpi5.sh` will:
- Install the Hailo PCIe kernel driver and software stack (`sudo apt install hailo-all`).
- Verify Hailo device node exists at `/dev/hailo0`.
- Create a Python virtual environment and install lightweight arm64 dependencies (`hailort`, `opencv-python-headless`, `onnxruntime`, `flask`, `numpy`).

### 2. Launch the SCADA Web Station (Recommended)
Start the multi-threaded inspection server:

```bash
./start_web_station.sh
```

Open your browser:
- **On Raspberry Pi 5:** [`http://localhost:5000`](http://localhost:5000)
- **From another PC / Tablet on the Factory LAN:** `http://<raspberry-pi-ip>:5000`

The HUD status pill in the top-right corner will show:
`● HAILO-8L (13 TOPS NPU) · LLAMA 3.2 1B`

### 3. Launch Standalone Desktop OpenCV HUD
For direct HDMI touchscreen or monitor display:

```bash
./start_desktop_hud.sh
```

---

## 📂 Package Directory Structure

```text
bharat_rpi5_deploy_halio/
├── config/
│   └── edge_qc_config.json      # Dual-mode calibrated thresholds, surface gate & LLM params
├── models/
│   ├── stage0_selector_224.onnx # Stage 0 Tiny Classifier ONNX model (224x224)
│   ├── yolo11n_416_v6.onnx      # Mode 1 Steel Defect YOLO11n ONNX fallback (416x416)
│   ├── yolo11n_416_v6.hef       # Mode 1 Hailo-8L 13 TOPS compiled binary (10.5 MB)
│   ├── mvtec_efficientad_s.onnx # Mode 2 Metal Nut Anomaly EfficientAD-S model (256x256)
│   ├── Llama-3.2-1B-UnifiedQC-FT-INT4.pte # On-device ExecuTorch INT4 LLM model (1.5 GB)
│   ├── tokenizer.model          # Meta Llama 3.2 BPE tokenizer
│   └── params.json              # Model hyperparameters
├── sample_test_images/          # Verified factory test images (clean & defect steel / nut samples)
├── templates/
│   └── index.html               # High-DPI dark industrial SCADA web interface
├── rpi5_qc_web_station.py       # Core multi-threaded web station service & REST API
├── rpi5_qc_assistant.py         # Standalone desktop OpenCV HUD application
├── rpi5_qc_inspection_log.csv   # Real-time CSV quality control audit logbook
├── sample_conveyor_video.mp4    # Industrial conveyor belt simulation video
├── requirements_rpi5.txt        # Lightweight Raspberry Pi 5 Python dependencies
├── setup_hailo_rpi5.sh          # One-click Hailo PCIe stack & environment installer
├── setup_env.sh                 # Generic Python virtual environment setup
├── start_web_station.sh         # One-click launcher for SCADA Web Station (Port 5000)
├── start_desktop_hud.sh         # One-click launcher for Desktop OpenCV HUD
└── .gitignore                   # Excludes heavy .pte files and runtime temp logs
```

---

## 📊 Benchmarks & Performance Metrics

| Pipeline Component | Hardware Target | Resolution / Size | Latency | Throughput | CPU Usage |
|---|---|---|---|---|---|
| **Stage 0 Classifier** | Cortex-A76 (ARM NEON) | 224 × 224 | **1.2 ms** | **833 FPS** | < 4% |
| **Mode 1: YOLO11n (NPU)** | Hailo-8L (13 TOPS AI HAT) | 416 × 416 | **9.1 ms** | **110 FPS** | **~0%** |
| **Mode 1: YOLO11n (CPU)** | Cortex-A76 (ARM NEON) | 416 × 416 | **31.4 ms** | **31.8 FPS** | 68% |
| **Mode 2: EfficientAD-S** | Cortex-A76 (ARM NEON) | 256 × 256 | **24.5 ms** | **40.8 FPS** | 52% |
| **Stage 2: Llama 3.2 1B** | Cortex-A76 (ExecuTorch INT4) | Telemetry Prompt | **72 ms / tok** | **13.8 tok/s** | 85% (single thread) |

---

## ⚙️ Calibration & Threshold Mathematics

### Mode 1: Multi-Defect Dominance Scoring
When multiple defects are detected in a single steel sheet frame, the primary defect is selected via:
$$\text{Dominance} = \text{Confidence} \times \left(1.0 + \frac{\text{BoundingBoxAreaPct}}{50.0}\right)$$

### Mode 2: EfficientAD Dual-Condition Anomaly Gate
To completely eliminate false rejects on normal parts while catching subtle hairline scratches:
- Image Anomaly Threshold: $T_{\text{IMG}} = 0.35$
- Pixel Anomaly Threshold: $T_{\text{PIX}} = 0.1676$
- Decision Rule:
  $$\text{IsAnomaly} = (\text{Score} \ge 0.35) \lor (\text{Count}(\text{Boxes}) \ge 1 \land \text{AreaPct} \ge 0.30\% \land \text{Score} \ge 0.28)$$
- Clean Parts: $\text{Score} \le 0.1949, \text{Area} \le 0.01\% \implies \text{CLEAN NUT PASS}$
- Defect Parts: Bent ($\approx 0.81$), Flip ($\approx 2.06$), Color ($\approx 0.45$), Scratch ($\ge 0.38$)

---

## ⌨️ Operator Shortcuts (Keyboard Controls)

| Key | Action | Description |
|:---:|---|---|
| `1` | **Demo Mode** | Simulates factory conveyor belt with jitter |
| `2` | **Image Mode** | Static image inspection with playlist dropdown |
| `3` | **Video Mode** | Continuous video stream inspection |
| `4` | **Live Mode** | Real-time USB Webcam or CSI camera feed |
| `5` | **Logbook** | Opens full-screen CSV Inspection Logbook |
| `Space` | **Pause / Resume** | Freezes conveyor feed for visual inspection |
| `N` / `P` | **Next / Prev** | Navigates through test media playlist |
| `S` | **Save Frame** | Saves current annotated inspection snapshot to disk |
| `O` | **Upload File** | Uploads new factory image or conveyor video |

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/state` | `GET` | Returns full JSON telemetry (mode, defect, confidence, boxes, lock status, LLM stream). |
| `/video_feed` | `GET` | High-DPI multipart MJPEG video stream with bounding boxes and heatmaps. |
| `/api/mode` | `POST` | Switches operating mode (`demo`, `image`, `video`, `live`, `log`). |
| `/api/image_index` | `POST` | Selects image from playlist dropdown (`{"index": N}`). |
| `/api/upload` | `POST` | Uploads an image or video file directly to the QC station. |
| `/api/trigger_llama` | `POST` | Manually re-analyzes current locked frame with Llama 3.2 1B. |
| `/api/logs` | `GET` | Fetches real-time CSV audit logbook records. |

---

## 🌐 Official Documentation & Technical Links

- **Challenge:**
  - [Bharat AI-SoC Challenge Portal (Arm Developer Labs)](https://arm-education.github.io/Arm-Developer-Labs/)
- **Hardware & Accelerators:**
  - [Raspberry Pi 5 Product Specs](https://www.raspberrypi.com/products/raspberry-pi-5/)
  - [Raspberry Pi AI HAT Documentation](https://www.raspberrypi.com/documentation/accessories/ai-hat.html)
  - [Hailo-8L AI Accelerator Overview](https://hailo.ai/products/ai-accelerators/hailo-8l-ai-accelerator-entry-level-m-2/)
  - [Hailo Dataflow Compiler & Software Suite](https://hailo.ai/developer-zone/software-downloads/)
- **Vision & LLM Frameworks:**
  - [PyTorch ExecuTorch Documentation](https://pytorch.org/executorch/)
  - [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
  - [Meta Llama 3.2 Model Card & Prompt Formats](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)
  - [MVTec Anomaly Detection Benchmark](https://www.mvtec.com/company/research/datasets/mvtec-ad)
  - [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)

---

## 📜 Citation & Acknowledgements

- **Bharat AI-SoC Challenge 2026–27:** Developed for Problem Statement 5 (Unified Multi-Mode Industrial Edge Quality Control Assistant).
- **Datasets:** NEU Surface Defect Database (Northeastern University) and MVTec Anomaly Detection (MVTec Software GmbH).
- **Edge AI Frameworks:** PyTorch ExecuTorch, Hailo Dataflow Compiler, Ultralytics YOLO11, and Meta Llama 3.2.
