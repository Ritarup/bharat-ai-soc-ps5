#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <cstdint>
#include <memory>
#include <deque>

namespace httplib {
    class Server;
}

namespace qc {
namespace ipc {

struct LiveTelemetryState {
    std::string mode = "DEMO";
    std::string active_tag = "DEMO 1/7 · Industrial Conveyor (C++ Native AOT)";
    std::string pipeline_mode = "AUTO";
    std::string active_pipeline_mode = "MODE_1";
    std::string stage0_route = "steel_sheet";
    float stage0_conf = 1.0f;
    bool is_mode2 = false;
    
    // Defect detection results
    std::string defect = "NO DEFECT DETECTED";
    std::string display_title = "NO DEFECT DETECTED";
    float conf = 0.0f;
    float score_ratio = 1.0f;
    float area = 0.0f;
    int boxes = 0;
    int regions = 0;
    std::string spread = "localized";
    std::string severity = "NONE";
    
    // Advisory results
    std::string cause = "Surface topography within ISO 4287 Ra specification.";
    std::string action = "Certified Prime Quality. Release steel coil for dispatch.";
    
    // Performance & Token metrics
    int tokens = 38;
    float tok_speed = 19.3f;
    float cv_ms = 24.5f;
    std::string engine_source = "Llama 3.2 1B (on-device, INT4 Resident)";
    float llm_ms = 1850.0f;
    bool llm_busy = false;
    float llm_elapsed = 0.0f;
    
    // Lock & UI Controls
    bool is_locked = true;
    int lock_cnt = 4;
    bool is_paused = false;
    std::string engine_mode = "LLAMA_ONLY";
    std::string vision_hardware = "ARM NEON CPU (OpenCV DNN)";
    std::string system_status = "AOT C++ Native Engine (Cores 0-3 Pinned) · Hailo-8L + Llama 3.2 1B";
    std::string hardware_status = "ARM NEON CPU · Cortex-A76 Pinned";
    std::string thread_affinity_info = "C0:I/O+HTTP, C1:Vision, C2-C3:Llama";
    int active_idx = 0;
};

struct InspectionRecord {
    std::string timestamp;
    std::string mode = "MODE_1";
    std::string part = "steel_sheet";
    std::string defect;
    std::string score_conf = "0.0%";
    std::string score_ratio = "-";
    float area_pct = 0.0f;
    int regions_boxes = 0;
    std::string spread = "-";
    std::string severity = "NONE";
    std::string cause;
    std::string action;
    std::string source = "System";
    float latency_ms = 0.0f;
};

/**
 * Production C++ HTTP SCADA Server.
 * Powered by cpp-httplib and nlohmann::json.
 * Pinned to Core 0 of Raspberry Pi 5.
 * Serves the full industrial SCADA interface directly on port 5000 with ZERO Python.
 */
class IPCServer {
public:
    IPCServer(int port = 5000);
    ~IPCServer();

    bool start();
    void stop();

    void update_state(const LiveTelemetryState& state);
    void update_state_for_generation(uint64_t gen_id, const LiveTelemetryState& state);
    LiveTelemetryState get_state() const;
    uint64_t get_sample_generation_id() const;

    void update_frame_jpeg(const std::vector<uint8_t>& jpeg_data);
    std::vector<uint8_t> get_frame_jpeg() const;

    std::string get_current_image_path() const;
    std::string get_current_video_path() const;
    int get_active_index() const;
    std::string get_active_mode() const;
    std::string get_pipeline_mode() const;
    bool is_paused() const;

    void next_sample();
    void prev_sample();
    void select_sample(int index);
    void select_sample_by_path(const std::string& path);
    void set_active_index(int idx);
    size_t get_playlist_size() const;
    std::string get_image_path_at(int idx) const;
    void toggle_pause();
    void set_pipeline_mode(const std::string& mode);
    void switch_mode(const std::string& mode);

    void append_log(const InspectionRecord& record);
    std::string get_logs_json() const;
    void clear_logs();
    std::string get_csv_logs() const;

private:
    int port_;
    std::atomic<bool> is_running_{false};
    std::atomic<uint64_t> sample_generation_id_{1};
    std::thread server_thread_;
    mutable std::mutex state_mutex_;
    mutable std::mutex frame_mutex_;
    mutable std::mutex log_mutex_;

    LiveTelemetryState current_state_;
    std::vector<uint8_t> current_jpeg_;
    std::vector<InspectionRecord> inspection_logs_;
    std::unique_ptr<httplib::Server> svr_;

    std::vector<std::pair<std::string, std::string>> playlist_;       // filename, full_path (images)
    std::vector<std::pair<std::string, std::string>> video_playlist_; // filename, full_path (videos)
    int video_idx_ = 0;

    void init_playlist();
    void load_logs_from_csv();
    void server_loop();
    std::string serialize_json_state() const;
};

} // namespace ipc
} // namespace qc
