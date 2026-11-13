#include "cpu_affinity.hpp"
#include "metallic_gate_neon.hpp"
#include "hailo_vision_engine.hpp"
#include "llama_resident.hpp"
#include "ipc_server.hpp"
#include "qc_rules.hpp"

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

static std::atomic<bool> g_shutdown{false};

void signal_handler(int sig) {
    (void)sig;
    std::cout << "\n[Bharat QC Native] Received shutdown signal. Terminating cleanly..." << std::endl;
    g_shutdown = true;
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "==================================================================" << std::endl;
    std::cout << "\033[1;36m BHARAT AI-SOC PS5: AHEAD-OF-TIME NATIVE QC ENGINE (.bin) \033[0m" << std::endl;
    std::cout << " Platform: Raspberry Pi 5 · Quad-Core Cortex-A76 @ 2.4 GHz" << std::endl;
    std::cout << " Target OS: Ubuntu 24.04 LTS arm64 (aarch64) · Zero Python Runtime" << std::endl;
    std::cout << " Acceleration: 13 TOPS Hailo-8L NPU (PCIe) + Llama 3.2 1B INT4" << std::endl;
    std::cout << "==================================================================" << std::endl;

    // Name Main Process Orchestrator (pin Core 0 after ONNX Runtime worker pools spawn)
    qc::hardware::set_thread_name("main_orchestrator");

    std::cout << "[Hardware Topology] Host CPU Cores detected: " << qc::hardware::get_cpu_core_count() << std::endl;
    std::cout << "  -> Core 0: I/O, Hailo PCIe DMA interrupts, and Web SCADA Server (:5000)" << std::endl;
    std::cout << "  -> Core 1: Vision Ingestion, YOLO11n / Stage0 / EfficientAD & NMS" << std::endl;
    std::cout << "  -> Cores 2 & 3: Warm Resident ExecuTorch Llama 3.2 1B INT4 Engine" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;

    // Locate models directory dynamically
    fs::path base_path = fs::current_path();
    std::vector<fs::path> candidate_dirs = {
        base_path / "models",
        base_path / ".." / "bharat_rpi5_deploy_halio" / "models",
        base_path.parent_path() / "bharat_rpi5_deploy_halio" / "models",
        base_path / ".." / "models",
        base_path / ".." / "bharat_rpi5_deploy_main" / "models"
    };
    fs::path models_path = base_path / "models";
    for (const auto& cand : candidate_dirs) {
        if (fs::exists(cand)) {
            models_path = cand;
            break;
        }
    }

    std::string hef_file = (models_path / "yolo11n_416_v6.hef").string();
    std::string yolo_onnx = (models_path / "yolo11n_416_v6.onnx").string();
    std::string stage0_onnx = (models_path / "stage0_selector_224.onnx").string();
    std::string mode2_onnx = (models_path / "mvtec_efficientad_s.onnx").string();
    std::string pte_file = (models_path / "Llama-3.2-1B-UnifiedQC-FT-INT4.pte").string();
    std::string tok_file = (models_path / "tokenizer.model").string();

    std::string runner_file = "bin/llama_daemon";

    std::cout << "[Paths] Models Directory: " << models_path << std::endl;
    std::cout << "  -> YOLO ONNX: " << (fs::exists(yolo_onnx) ? "Found" : "Missing") << std::endl;
    std::cout << "  -> Stage0 ONNX: " << (fs::exists(stage0_onnx) ? "Found" : "Missing") << std::endl;
    std::cout << "  -> Mode2 ONNX: " << (fs::exists(mode2_onnx) ? "Found" : "Missing") << std::endl;
    std::cout << "  -> Llama PTE: " << (fs::exists(pte_file) ? "Found" : "Missing") << std::endl;

    // 1. Initialize Hailo Vision Engine
    std::cout << "\n[1/3] Initializing Vision Engine (Hailo-8L / CPU ONNX Fallback)..." << std::endl;
    qc::vision::HailoVisionEngine vision_engine;
    vision_engine.init(hef_file, yolo_onnx, stage0_onnx, mode2_onnx);

    // 2. Initialize Resident Llama 3.2 1B Engine
    std::cout << "\n[2/3] Initializing Warm Resident Llama 3.2 1B Engine on Cores 2 & 3..." << std::endl;
    qc::reasoning::LlamaResidentEngine llama_engine;
    llama_engine.init(pte_file, tok_file, runner_file);

    // 3. Start Native Pure C++ Web SCADA Server on Port 5000 (Core 0)
    std::cout << "\n[3/3] Starting Native Pure C++ Web SCADA Server on port 5000 (Core 0)..." << std::endl;
    qc::hardware::pin_current_thread(qc::hardware::CORE_IO_IPC);
    qc::ipc::IPCServer ipc_server(5000);
    ipc_server.start();

    std::cout << "\n\033[1;32m>>> ALL ENGINES ONLINE · PURE C++ REAL-TIME INSPECTION ACTIVE <<<\033[0m" << std::endl;
    std::cout << "\033[1;36m>>> Web SCADA Workstation URL: http://localhost:5000 <<<\033[0m\n" << std::endl;

    // Launch Dedicated Vision Processing Thread on Core 1
    std::thread vision_thread([&]() {
        qc::hardware::set_thread_name("vision_worker_c1");
        qc::hardware::pin_current_thread(qc::hardware::CORE_VISION);

        std::string last_defect_locked = "";
        int stability_lock_counter = 0;
        std::string last_advisory_defect = "INITIAL_UNSET";
        int last_advisory_idx = -1;
        int demo_step_cnt = 0;
        bool current_image_analyzed = false;
        uint64_t last_processed_gen_id = 0;
        const int REQUIRED_LOCK_FRAMES = 4;

        cv::VideoCapture video_cap;
        std::string active_capture_kind = "";
        std::string active_video_path = "";

        auto render_offline_banner = [&](const std::string& m, uint64_t gen_id, int idx) {
            cv::Mat banner(480, 640, CV_8UC3, cv::Scalar(18, 18, 18));
            const bool is_cam = (m == "CAMERA");
            std::string title = is_cam ? "Camera Offline" : "No Media Loaded";
            std::string sub = is_cam ? "Connect USB Webcam or CSI Camera" : "Drop image/video or click Upload";
            cv::putText(banner, title, cv::Point(195, 230), cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(160, 140, 120), 2);
            cv::putText(banner, sub, cv::Point(155, 270), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(120, 105, 90), 1);

            std::vector<uint8_t> jpeg;
            cv::imencode(".jpg", banner, jpeg, {cv::IMWRITE_JPEG_QUALITY, 88});
            if (!jpeg.empty()) {
                ipc_server.update_frame_jpeg(jpeg);
            }

            qc::ipc::LiveTelemetryState st = ipc_server.get_state();
            st.active_idx = idx;
            st.defect = is_cam ? "CAMERA OFFLINE" : "NO MEDIA LOADED";
            st.display_title = st.defect;
            st.severity = "NONE";
            st.conf = 0.0f;
            st.score_ratio = 0.0f;
            st.area = 0.0f;
            st.boxes = 0;
            st.regions = 0;
            st.spread = "-";
            st.cause = is_cam ? "No camera hardware detected (/dev/video0 or CSI)" : "No image or video loaded";
            st.action = is_cam ? "Connect USB Webcam or Raspberry Pi Camera Module" : "Select image or video from toolbar";
            st.engine_source = "Hardware Standby";
            st.cv_ms = 1.0f;
            st.llm_ms = 0.0f;
            st.llm_busy = false;
            st.is_locked = false;
            st.lock_cnt = 0;
            ipc_server.update_state_for_generation(gen_id, st);
        };

        while (!g_shutdown) {
            if (ipc_server.is_paused()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            std::string mode = ipc_server.get_active_mode();
            std::string pipe_mode = ipc_server.get_pipeline_mode();
            int cur_idx = ipc_server.get_active_index();
            uint64_t cur_gen_id = ipc_server.get_sample_generation_id();

            if (mode == "LOG") {
                if (video_cap.isOpened()) {
                    video_cap.release();
                    active_capture_kind.clear();
                    active_video_path.clear();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(80));
                continue;
            }

            // In IMAGE mode, if current selected image is already analyzed, idle efficiently
            if (mode == "IMAGE" && current_image_analyzed && cur_gen_id == last_processed_gen_id) {
                std::this_thread::sleep_for(std::chrono::milliseconds(40));
                continue;
            }

            if (cur_gen_id != last_processed_gen_id) {
                last_processed_gen_id = cur_gen_id;
                current_image_analyzed = false;
                stability_lock_counter = 0;
                last_defect_locked = "";
            }

            cv::Mat frame;
            if (mode == "VIDEO") {
                std::string vid_file = ipc_server.get_current_video_path();
                if (vid_file.empty() || !fs::exists(vid_file)) {
                    if (video_cap.isOpened()) video_cap.release();
                    active_capture_kind.clear();
                    active_video_path.clear();
                    render_offline_banner("VIDEO", cur_gen_id, cur_idx);
                    std::this_thread::sleep_for(std::chrono::milliseconds(120));
                    continue;
                }
                if (active_capture_kind != "VIDEO" || active_video_path != vid_file || !video_cap.isOpened()) {
                    if (video_cap.isOpened()) video_cap.release();
                    video_cap.open(vid_file);
                    active_capture_kind = "VIDEO";
                    active_video_path = vid_file;
                }
                if (video_cap.isOpened()) {
                    video_cap >> frame;
                    if (frame.empty()) {
                        video_cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                        video_cap >> frame;
                    }
                }
                if (frame.empty()) {
                    render_offline_banner("VIDEO", cur_gen_id, cur_idx);
                    std::this_thread::sleep_for(std::chrono::milliseconds(120));
                    continue;
                }
            } else if (mode == "CAMERA") {
                if (active_capture_kind != "CAMERA") {
                    if (video_cap.isOpened()) video_cap.release();
                    active_video_path.clear();
                    active_capture_kind = "CAMERA";
                    if (fs::exists("/dev/video0") || fs::exists("/dev/video1")) {
                        video_cap.open(0);
                    }
                }
                if (video_cap.isOpened()) {
                    video_cap >> frame;
                }
                if (frame.empty()) {
                    render_offline_banner("CAMERA", cur_gen_id, cur_idx);
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                    continue;
                }
            } else if (mode == "DEMO") {
                if (video_cap.isOpened()) video_cap.release();
                active_capture_kind.clear();
                active_video_path.clear();
                std::string img_p = ipc_server.get_current_image_path();
                if (!img_p.empty() && fs::exists(img_p)) {
                    cv::Mat base = cv::imread(img_p);
                    if (!base.empty()) {
                        // Synthetic conveyor jitter (-2 to +2 px translation)
                        int dx = (std::rand() % 5) - 2;
                        int dy = (std::rand() % 5) - 2;
                        cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
                        cv::warpAffine(base, frame, M, base.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
                    }
                }
            } else {
                // IMAGE mode
                if (video_cap.isOpened()) video_cap.release();
                active_capture_kind.clear();
                active_video_path.clear();
                std::string img_p = ipc_server.get_current_image_path();
                if (!img_p.empty() && fs::exists(img_p)) {
                    frame = cv::imread(img_p);
                }
            }

            if (frame.empty()) {
                frame = cv::Mat(480, 640, CV_8UC3, cv::Scalar(40, 40, 40));
            }

            // Stage 0 Routing
            auto s0 = vision_engine.classify_stage0(frame);
            std::string active_pipe = pipe_mode;
            if (pipe_mode == "AUTO") {
                active_pipe = s0.route;
            }

            // Run Real Vision Inference
            qc::vision::VisionInferenceResult v_res;
            if (active_pipe == "MODE_2") {
                v_res = vision_engine.infer_mode2(frame);
            } else {
                v_res = vision_engine.infer_mode1(frame);
            }

            // If another sample was requested while vision model was running, discard stale pass
            if (ipc_server.get_sample_generation_id() != cur_gen_id) {
                continue;
            }

            // Update JPEG frame buffer for web streaming
            if (!v_res.encoded_jpeg.empty()) {
                ipc_server.update_frame_jpeg(v_res.encoded_jpeg);
            }

            // 4-Frame Stability Lock-In Logic
            std::string current_defect = v_res.has_defects ? v_res.primary_defect : "NO DEFECT DETECTED";
            if (mode == "IMAGE") {
                stability_lock_counter = 4;
            } else {
                if (current_defect == last_defect_locked) {
                    stability_lock_counter++;
                } else {
                    last_defect_locked = current_defect;
                    stability_lock_counter = 1;
                }
            }

            bool is_locked = (stability_lock_counter >= REQUIRED_LOCK_FRAMES);

            // Prepare Live Telemetry State
            qc::ipc::LiveTelemetryState cur = ipc_server.get_state();
            cur.active_idx = cur_idx;
            cur.stage0_route = s0.class_name;
            cur.stage0_conf = s0.confidence;
            cur.active_pipeline_mode = active_pipe;
            cur.is_mode2 = v_res.is_mode2;
            cur.defect = v_res.has_defects ? v_res.display_title : (v_res.is_mode2 ? "CLEAN NUT PASS" : "NO DEFECT DETECTED");
            cur.display_title = cur.defect;
            cur.conf = v_res.highest_confidence;
            cur.score_ratio = v_res.score_ratio;
            cur.area = v_res.union_area_pct;
            cur.boxes = v_res.total_boxes;
            cur.regions = v_res.regions;
            cur.spread = v_res.spread;
            cur.severity = v_res.severity;
            cur.cv_ms = v_res.cv_latency_ms;
            cur.vision_hardware = v_res.hardware_device;
            cur.is_locked = is_locked;
            cur.lock_cnt = std::min(REQUIRED_LOCK_FRAMES, stability_lock_counter);

            // Fetch Live Advisory when locked defect changes or when active item changes
            bool need_advisory = false;
            if (mode == "IMAGE") {
                if (!current_image_analyzed && is_locked) {
                    need_advisory = true;
                    current_image_analyzed = true;
                }
            } else {
                if (is_locked && (current_defect != last_advisory_defect || cur_idx != last_advisory_idx)) {
                    need_advisory = true;
                    last_advisory_defect = current_defect;
                    last_advisory_idx = cur_idx;
                }
            }

            if (need_advisory) {
                if (v_res.has_defects) {
                    cur.llm_busy = true;
                    cur.cause = "Executing on-device Llama 3.2 1B...";
                    cur.action = "Generating root-cause & SOP remediation...";
                    cur.engine_source = "Llama 3.2 1B (Running...)";
                    cur.lock_cnt = 4;
                    cur.is_locked = true;
                    ipc_server.update_state_for_generation(cur_gen_id, cur);

                    std::string defect_for_llama = v_res.is_mode2 ? v_res.sub_type : v_res.primary_defect;
                    auto adv = llama_engine.request_advisory_sync(
                        defect_for_llama,
                        v_res.severity,
                        v_res.highest_confidence,
                        v_res.union_area_pct,
                        v_res.total_boxes,
                        v_res.is_mode2
                    );

                    cur.cause = adv.possible_cause;
                    cur.action = adv.recommended_action;
                    cur.engine_source = adv.advisory_source;
                    cur.llm_ms = adv.llm_latency_ms;
                    cur.tokens = adv.tokens_generated;
                    cur.tok_speed = adv.tokens_per_sec;
                    cur.llm_busy = false;

                    // Log to SCADA CSV
                    qc::ipc::InspectionRecord rec;
                    auto now_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                    std::stringstream t_ss;
                    t_ss << std::put_time(std::localtime(&now_t), "%Y-%m-%d %H:%M:%S");
                    rec.timestamp = t_ss.str();
                    rec.mode = v_res.is_mode2 ? "MODE_2" : "MODE_1";
                    rec.part = v_res.is_mode2 ? "metal_nut" : "steel_sheet";
                    rec.defect = v_res.display_title;
                    {
                        char buf[32];
                        if (v_res.is_mode2) {
                            rec.score_conf = "N/A";
                            std::snprintf(buf, sizeof(buf), "%.2fx", v_res.score_ratio);
                            rec.score_ratio = buf;
                        } else {
                            std::snprintf(buf, sizeof(buf), "%.1f%%", v_res.highest_confidence * 100.0f);
                            rec.score_conf = buf;
                            rec.score_ratio = "-";
                        }
                    }
                    rec.area_pct = v_res.union_area_pct;
                    rec.regions_boxes = v_res.is_mode2 ? v_res.regions : v_res.total_boxes;
                    rec.spread = v_res.spread.empty() ? "-" : v_res.spread;
                    rec.severity = v_res.severity;
                    rec.cause = cur.cause;
                    rec.action = cur.action;
                    rec.source = cur.engine_source;
                    rec.latency_ms = cur.llm_ms;
                    if (ipc_server.get_sample_generation_id() == cur_gen_id) {
                        ipc_server.append_log(rec);
                    }
                } else {
                    if (v_res.is_mode2) {
                        cur.cause = "No anomaly detected on component";
                        cur.action = "Pass component to assembly line";
                    } else {
                        cur.cause = "No defect detected";
                        cur.action = "Pass component to next stage";
                    }
                    cur.engine_source = "System Ready";
                    cur.llm_ms = 0.0f;
                    cur.tokens = 0;
                    cur.tok_speed = 0.0f;
                    cur.llm_busy = false;
                }
            } else if (!is_locked && !cur.llm_busy) {
                if (v_res.has_defects) {
                    if (v_res.is_mode2) {
                        cur.cause = "Anomaly detected · Verifying stability...";
                        cur.action = "Awaiting 4 consistent frames before advisory";
                        cur.engine_source = "Temporal Filter (Mode 2)";
                    } else {
                        cur.cause = "Defect detected · Verifying stability...";
                        cur.action = "Awaiting temporal lock before advisory synthesis";
                        cur.engine_source = "Temporal Filter";
                    }
                } else {
                    if (v_res.is_mode2) {
                        cur.cause = "No anomaly detected on component";
                        cur.action = "Pass component to assembly line";
                    } else {
                        cur.cause = "No defect detected";
                        cur.action = "Pass component to next stage";
                    }
                    cur.engine_source = "System Ready";
                }
            }

            // Advance conveyor in DEMO mode if not paused and not busy
            if (mode == "DEMO" && !ipc_server.is_paused() && !cur.llm_busy) {
                demo_step_cnt++;
                if (demo_step_cnt >= 12) {
                    demo_step_cnt = 0;
                    int total_s = static_cast<int>(ipc_server.get_playlist_size());
                    if (total_s > 0) {
                        int next_i = (ipc_server.get_active_index() + 1) % total_s;
                        ipc_server.set_active_index(next_i);
                    }
                }
            }

            ipc_server.update_state_for_generation(cur_gen_id, cur);

            // Regulate conveyor frame rate (~30 FPS or step pause)
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    });

    // Main Orchestrator Loop & Console Telemetry HUD
    auto last_hud_time = std::chrono::steady_clock::now();
    while (!g_shutdown) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<float>(now - last_hud_time).count() >= 1.0f) {
            last_hud_time = now;
            auto s = ipc_server.get_state();
            float live_fps = 1000.0f / std::max(1.0f, s.cv_ms);
            std::cout << "\r[Telemetric HUD] "
                      << "FPS: " << std::fixed << std::setprecision(1) << live_fps
                      << " | CV: " << s.cv_ms << "ms"
                      << " | LLM: " << std::fixed << std::setprecision(0) << s.llm_ms << "ms"
                      << " | Defect: " << s.display_title
                      << " (" << std::fixed << std::setprecision(0) << (s.conf * 100.0f) << "%)"
                      << " | Sev: " << s.severity
                      << " | Mode: " << s.active_pipeline_mode
                      << " | Cores: Pinned (C0-C3)   " << std::flush;
        }
    }

    if (vision_thread.joinable()) {
        vision_thread.join();
    }

    llama_engine.shutdown();
    ipc_server.stop();

    std::cout << "\n[Bharat QC Native] Clean shutdown completed. Goodbye!" << std::endl;
    return 0;
}
