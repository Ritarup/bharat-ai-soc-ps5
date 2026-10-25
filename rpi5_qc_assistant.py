#!/usr/bin/env python3
import csv, json, os, re, subprocess, time
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = Path("models")
CONFIG_JSON = Path("config/edge_qc_config.json")
SAMPLES_DIR = Path("sample_test_images")
YOLO_ONNX = MODELS_DIR / "yolo11n_416_v6.onnx"
TOKENIZER_BIN = MODELS_DIR / "tokenizer.model"
LLAMA_PTE = MODELS_DIR / "Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"
EXECUTORCH_RUNNER = Path("./llama_main")
CSV_LOG = Path("rpi5_qc_inspection_log.csv")

with open(CONFIG_JSON, "r", encoding="utf-8") as f:
    CFG = json.load(f)

T_MED = float(CFG["T_MED"])
T_HIGH = float(CFG["T_HIGH"])
CLASSES = CFG["CLASSES"]
KB = CFG["KB"]

CLASS_COLORS = {
    "crazing": (0, 0, 255), "inclusion": (0, 165, 255), "patches": (0, 255, 255),
    "pitted_surface": (255, 0, 255), "rolled-in_scale": (0, 128, 255), "scratches": (255, 255, 0)
}
LINE_RE = re.compile(r"Likely Cause:\s*(.+?)\s*\|\s*Recommended Action:\s*(.+)", re.IGNORECASE)

def union_area_pct(boxes, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))] = 1
    return 100.0 * float(mask.mean())

def compute_severity(defect, area_pct, n_boxes=1):
    if defect == "crazing" or area_pct >= T_HIGH:
        return "High"
    if area_pct >= T_MED:
        return "High" if n_boxes >= 3 else "Medium"
    return "Medium" if n_boxes >= 3 else "Low"

def query_llama_executorch(defect, area_pct, severity):
    ref_c = KB[defect]["cause"]
    ref_a = KB[defect]["actions"][severity]
    t0 = time.perf_counter()
    if EXECUTORCH_RUNNER.exists() and LLAMA_PTE.exists() and TOKENIZER_BIN.exists():
        try:
            prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                + CFG["SYSTEM_PROMPT"] + "<|eot_id|>"
                + "<|start_header_id|>user<|end_header_id|>\n\n"
                + CFG["FEWSHOT_USER"] + "<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                + CFG["FEWSHOT_ASSISTANT"] + "<|eot_id|>"
                + "<|start_header_id|>user<|end_header_id|>\n\n"
                + f"Defect: {defect} | Area: {area_pct:.1f}% | Severity: {severity}\n"
                + f"Reference cause: {ref_c}\nReference action: {ref_a}<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n\nLikely Cause:"
            )
            cmd = [str(EXECUTORCH_RUNNER), f"--model_path={LLAMA_PTE}", f"--tokenizer_path={TOKENIZER_BIN}", f"--prompt={prompt}", "--temperature=0.0", "--seq_len=256"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=6.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            raw = "Likely Cause:" + res.stdout.split("Likely Cause:")[-1]
            first_line = raw.strip().splitlines()[0] if raw.strip() else ""
            m = LINE_RE.search(first_line)
            if m:
                c, a = m.group(1).strip().rstrip("."), m.group(2).strip().rstrip(".")
                if not re.search(r"\d|°|operator", f"{c} {a}".lower()):
                    return c, a, "ExecuTorch-SpinQuant-INT4 (.pte)", elapsed_ms
        except Exception:
            pass
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return ref_c, ref_a, "KB-Grounded-Advisory", elapsed_ms

class RPi5QCAssistant:
    def __init__(self):
        print(f"Loading YOLO11n Vision Model: {YOLO_ONNX}...")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(YOLO_ONNX), sess_options=sess_opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.frame_history = deque(maxlen=4)
        self.last_locked_defect = None
        self.latest_severity = "NONE"
        self.latest_defect = "CLEAN STEEL SURFACE (0 Defects)"
        self.latest_cause = "Surface within normal quality tolerances"
        self.latest_action = "Pass component to next assembly stage"
        with open(CSV_LOG, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Timestamp", "Defect_Class", "Confidence_%", "Union_Area_%", "Box_Count", "Severity", "Likely_Cause", "Recommended_Action", "Advisory_Source", "LLM_Latency_ms"])

    def detect(self, frame_bgr, conf_thres=0.35, iou_thres=0.55):
        h_orig, w_orig = frame_bgr.shape[:2]
        blur = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=1.2)
        sharp = cv2.addWeighted(frame_bgr, 1.35, blur, -0.35, 0)
        resized = cv2.resize(sharp, (416, 416))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)
        t0 = time.perf_counter()
        raw_out = self.session.run(None, {self.input_name: tensor})[0]
        cv_ms = (time.perf_counter() - t0) * 1000.0
        preds = np.squeeze(raw_out).T
        boxes_xywh, scores, class_ids = [], [], []
        sx, sy = w_orig / 416.0, h_orig / 416.0
        for row in preds:
            cls_id = int(np.argmax(row[4:]))
            conf = float(row[4 + cls_id])
            if conf >= conf_thres:
                xc, yc, bw, bh = row[0], row[1], row[2], row[3]
                boxes_xywh.append([int((xc - bw / 2.0) * sx), int((yc - bh / 2.0) * sy), int(bw * sx), int(bh * sy)])
                scores.append(conf)
                class_ids.append(cls_id)
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_thres, iou_thres)
        dets = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, wb, hb = boxes_xywh[idx]
                dets.append({"class_name": CLASSES[class_ids[idx]], "conf": scores[idx], "box": (max(0, x), max(0, y), min(w_orig, x + wb), min(h_orig, y + hb))})
        return dets, cv_ms

    def step(self, frame_bgr, mode_tag="CONVEYOR SIM"):
        canvas = cv2.resize(frame_bgr, (640, 480))
        h, w = canvas.shape[:2]
        dets, cv_ms = self.detect(canvas)
        fps = 1000.0 / max(1.0, cv_ms)
        primary_defect, best_conf = None, 0.0
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            c_name = d["class_name"]
            color = CLASS_COLORS.get(c_name, (0, 255, 0))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{c_name} {d['conf']*100:.0f}%", (x1, max(52, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if d["conf"] > best_conf:
                best_conf, primary_defect = d["conf"], c_name
        primary_dets = [d for d in dets if d["class_name"] == primary_defect]
        area_pct = union_area_pct([d["box"] for d in primary_dets], w, h) if primary_dets else 0.0
        self.frame_history.append(primary_defect)
        lock_cnt = sum(1 for x in self.frame_history if x == primary_defect and x is not None)
        is_locked = (primary_defect is not None and len(self.frame_history) == 4 and lock_cnt == 4)
        if is_locked and primary_defect != self.last_locked_defect:
            self.last_locked_defect = primary_defect
            sev = compute_severity(primary_defect, area_pct, len(primary_dets))
            cause, action, src, llm_ms = query_llama_executorch(primary_defect, area_pct, sev)
            self.latest_severity = sev
            self.latest_defect = f"{primary_defect.upper()} ({best_conf*100:.1f}% | Area: {area_pct:.1f}%)"
            self.latest_cause, self.latest_action = cause, action
            with open(CSV_LOG, mode="a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), primary_defect, f"{best_conf*100:.1f}", f"{area_pct:.2f}", len(primary_dets), sev, cause, action, src, f"{llm_ms:.1f}"])
            print(f"[LOCKED] {self.latest_defect} | Severity: {sev} | Action: {action}")
        elif primary_defect is None and all(x is None for x in self.frame_history):
            self.last_locked_defect = None
            self.latest_severity = "NONE"
            self.latest_defect = "CLEAN STEEL SURFACE (0 Defects)"
            self.latest_cause = "Surface within normal quality tolerances"
            self.latest_action = "Pass component to next assembly stage"
        cv2.rectangle(canvas, (0, 0), (w, 40), (20, 20, 20), -1)
        lock_str = "LOCKED!" if is_locked else f"{lock_cnt}/4 Frames"
        cv2.putText(canvas, f"[{mode_tag}] CV: {fps:.1f} FPS ({cv_ms:.1f}ms) | Lock: {lock_str}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 128), 2)
        cv2.rectangle(canvas, (0, h - 90), (w, h), (15, 15, 15), -1)
        s_col = (0, 0, 255) if self.latest_severity == "High" else ((0, 165, 255) if self.latest_severity == "Medium" else (0, 255, 128))
        cv2.putText(canvas, f"STATUS: {self.latest_defect} | SEVERITY: {self.latest_severity}", (12, h - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, s_col, 2)
        cv2.putText(canvas, f"Cause: {self.latest_cause[:78]}", (12, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1)
        cv2.putText(canvas, f"Action: {self.latest_action[:78]}", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 255, 100), 1)
        return canvas

if __name__ == "__main__":
    assistant = RPi5QCAssistant()
    sample_imgs = sorted(SAMPLES_DIR.glob("*.jpg"))
    print("Controls: Press w for Live Camera | Press q to Quit.")
    live_cam = False
    while not live_cam:
        for img_p in sample_imgs:
            frame = cv2.imread(str(img_p))
            if frame is None: continue
            for _ in range(8):
                M = np.float32([[1, 0, np.random.randint(-2, 3)], [0, 1, np.random.randint(-2, 3)]])
                jittered = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
                cv2.imshow("Bharat AI-SoC PS5 - RPi5 / Laptop QC Station", assistant.step(jittered, mode_tag="CONVEYOR SIM"))
                key = cv2.waitKey(250) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    raise SystemExit()
                elif key == ord("w"):
                    live_cam = True; break
            if live_cam: break
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Bharat AI-SoC PS5 - RPi5 / Laptop QC Station", assistant.step(frame, mode_tag="LIVE CAMERA"))
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release()
    cv2.destroyAllWindows()
