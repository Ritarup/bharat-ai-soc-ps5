# Bharat AI-SoC PS5: Real-Time Industrial Defect Detection & QC Assistant

An on-device visual inspection and quality-control pipeline optimized for edge hardware (Raspberry Pi 5 and Android) to detect steel surface defects and deliver deterministic operator advisories.

---

## 🏗️ System Architecture & Pipeline

1. **Vision Engine (`YOLO11n`):** Custom-trained over multiple ablation iterations on the NEU-DET steel surface defect dataset (`416x416` resolution), exported to **ONNX**, **NCNN**, and **LiteRT (.tflite)**.
2. **OpenCV Preprocessing:** Unsharp masking and Gaussian blur stabilization to handle factory lighting variations.
3. **Temporal Consistency Lock-In:** A **4-frame ring buffer** to eliminate false-positive flicker on moving conveyor belts.
4. **Calibrated Union-Area Severity:** Polygon union-area calculation (preventing double-counting of overlapping bounding boxes) using NEU-DET dataset-calibrated thresholds:
   - **Medium Severity:** ≥ 22.1% union area
   - **High Severity:** ≥ 45.8% union area *(Crazing structural cracks always default to High)*
5. **LLM Advisory Engine (`Llama-3.2-1B-Instruct`):** Runs on-device via **PyTorch ExecuTorch (SpinQuant INT4)** paired with a **Dynamic Micro-RAG Knowledge Base** and a strict regex guardrail enforcing zero-hallucination operator advisories.
6. **Telemetry & Logging:** Real-time HUD overlay and automated CSV inspection logging (`rpi5_qc_inspection_log.csv`).

---

## 📂 Repository Structure

```text
bharat-ai-soc-ps5/
├── models/
│   ├── yolo11n_416_v6.onnx            # Vision model for ONNX Runtime (Laptop / RPi5)
│   ├── yolo11n_416_v6_ncnn_model/     # NCNN model optimized for Arm CPU
│   ├── tokenizer.model                # Llama 3.2 tokenizer
│   └── params.json                    # Model architecture parameters
├── config/
│   └── edge_qc_config.json            # Calibrated thresholds, KB, and prompt rules
├── sample_test_images/
│   ├── test_00_clean_steel.jpg        # Clean baseline sample
│   ├── test_crazing.jpg               # Crazing defect sample
│   ├── test_inclusion.jpg             # Inclusion defect sample
│   ├── test_patches.jpg               # Patches defect sample
│   ├── test_pitted_surface.jpg        # Pitted surface defect sample
│   ├── test_rolled-in_scale.jpg       # Rolled-in scale defect sample
│   └── test_scratches.jpg             # Scratches defect sample
├── rpi5_qc_assistant.py               # Unified edge pipeline script (Conveyor Sim + Live Camera)
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## 🚀 Quick Start (PC Simulation & RPi 5 Testing)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the QC Assistant Pipeline

The script automatically simulates a factory conveyor belt cycling through the bundled clean and defective steel test images, and allows switching to a live webcam feed by pressing `w`:

```bash
python rpi5_qc_assistant.py
```

### 3. Download the ExecuTorch `.pte` Model (Raspberry Pi 5)

To enable on-device Llama 3.2 1B generation on your Raspberry Pi 5 via ExecuTorch, download the pre-compiled INT4 model into your `models/` directory:

```bash
wget -O models/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte "https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET/resolve/main/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"
```

---

## 📋 License & Attribution

- **Built with Llama:** This project utilizes `Llama-3.2-1B-Instruct` quantized with SpinQuant INT4. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
