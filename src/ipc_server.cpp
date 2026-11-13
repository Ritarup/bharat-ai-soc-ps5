#include "ipc_server.hpp"
#include "cpu_affinity.hpp"
#include "embedded_ui.hpp"

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#include "httplib.h"
#include "json.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cstdio>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {
constexpr const char* CSV_LOG_PATH = "rpi5_qc_inspection_log.csv";
constexpr const char* CSV_HEADER_LINE =
    "Timestamp,Mode,Part,DefectClass,ScoreConf,ScoreRatio,AreaPct,RegionsBoxes,Spread,Severity,PossibleCause,RecommendedAction,AdvisorySource,LatencyMs";

bool is_video_extension(std::string ext) {
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv" || ext == ".webm";
}

bool is_image_extension(std::string ext) {
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp";
}

std::string escape_csv(const std::string& val) {
    std::string out;
    out.reserve(val.size() + 4);
    for (char c : val) {
        if (c == '"') {
            out += "\"\"";
        } else if (c == '\r' || c == '\n') {
            out += ' ';
        } else {
            out += c;
        }
    }
    return "\"" + out + "\"";
}

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur += '"';
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur += c;
            }
        } else {
            if (c == '"') {
                in_quotes = true;
            } else if (c == ',') {
                fields.push_back(cur);
                cur.clear();
            } else if (c != '\r') {
                cur += c;
            }
        }
    }
    fields.push_back(cur);
    return fields;
}

std::string format_one_decimal(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", v);
    return std::string(buf);
}

void reset_analyzing_state(qc::ipc::LiveTelemetryState& st) {
    st.is_locked = false;
    st.lock_cnt = 0;
    st.llm_busy = false;
    st.defect = "ANALYZING...";
    st.display_title = "ANALYZING...";
    st.cause = "Evaluating image on neural network...";
    st.action = "Awaiting inference...";
}
} // namespace

namespace qc {
namespace ipc {

IPCServer::IPCServer(int port) : port_(port), svr_(std::make_unique<httplib::Server>()) {
    init_playlist();
    load_logs_from_csv();
}

IPCServer::~IPCServer() {
    stop();
}

void IPCServer::init_playlist() {
    playlist_.clear();
    video_playlist_.clear();

    // Check candidate sample image directories
    std::vector<fs::path> dirs = {
        "sample_test_images",
        "../sample_test_images",
        "native/sample_test_images"
    };

    fs::path chosen_dir;
    for (const auto& d : dirs) {
        if (fs::exists(d) && fs::is_directory(d)) {
            chosen_dir = d;
            break;
        }
    }

    if (!chosen_dir.empty()) {
        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(chosen_dir)) {
            if (entry.is_regular_file() && is_image_extension(entry.path().extension().string())) {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
        for (const auto& p : files) {
            playlist_.push_back({p.filename().string(), p.string()});
        }
    }

    if (playlist_.empty()) {
        playlist_.push_back({"test_00_clean_steel.jpg", "sample_test_images/test_00_clean_steel.jpg"});
        playlist_.push_back({"test_crazing.jpg", "sample_test_images/test_crazing.jpg"});
        playlist_.push_back({"test_inclusion.jpg", "sample_test_images/test_inclusion.jpg"});
        playlist_.push_back({"test_patches.jpg", "sample_test_images/test_patches.jpg"});
        playlist_.push_back({"test_pitted_surface.jpg", "sample_test_images/test_pitted_surface.jpg"});
        playlist_.push_back({"test_rolled-in_scale.jpg", "sample_test_images/test_rolled-in_scale.jpg"});
        playlist_.push_back({"test_scratches.jpg", "sample_test_images/test_scratches.jpg"});
        playlist_.push_back({"test_bent_1.png", "sample_test_images/test_bent_1.png"});
        playlist_.push_back({"test_color_1.png", "sample_test_images/test_color_1.png"});
        playlist_.push_back({"test_flip_1.png", "sample_test_images/test_flip_1.png"});
        playlist_.push_back({"test_good_1.png", "sample_test_images/test_good_1.png"});
        playlist_.push_back({"test_scratch_1.png", "sample_test_images/test_scratch_1.png"});
    }

    // Also include any previously uploaded images
    if (fs::exists("uploads") && fs::is_directory("uploads")) {
        std::vector<fs::path> up_imgs, up_vids;
        for (const auto& entry : fs::directory_iterator("uploads")) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            if (is_image_extension(ext)) up_imgs.push_back(entry.path());
            else if (is_video_extension(ext)) up_vids.push_back(entry.path());
        }
        std::sort(up_imgs.begin(), up_imgs.end());
        for (const auto& p : up_imgs) {
            playlist_.push_back({p.filename().string(), p.string()});
        }
        std::sort(up_vids.begin(), up_vids.end());
        for (const auto& p : up_vids) {
            video_playlist_.push_back({p.filename().string(), p.string()});
        }
    }

    // Discover sample videos in current / parent directories
    for (const fs::path& dir_cand : {fs::path("."), fs::path(".."), fs::path("sample_videos"), fs::path("../sample_videos")}) {
        if (!fs::exists(dir_cand) || !fs::is_directory(dir_cand)) continue;
        std::vector<fs::path> vids;
        for (const auto& entry : fs::directory_iterator(dir_cand)) {
            if (entry.is_regular_file() && is_video_extension(entry.path().extension().string())) {
                vids.push_back(entry.path());
            }
        }
        std::sort(vids.begin(), vids.end());
        for (const auto& vp : vids) {
            std::string fname = vp.filename().string();
            bool dup = false;
            for (const auto& existing : video_playlist_) {
                if (existing.first == fname) { dup = true; break; }
            }
            if (!dup) {
                video_playlist_.push_back({fname, vp.string()});
            }
        }
    }

    current_state_.active_idx = 0;
    video_idx_ = 0;
    current_state_.active_tag = "DEMO 1/" + std::to_string(playlist_.size()) + " · " + playlist_[0].first;
}

std::string IPCServer::get_current_image_path() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (playlist_.empty()) return "";
    int sz = static_cast<int>(playlist_.size());
    int idx = (current_state_.active_idx % sz + sz) % sz;
    return playlist_[idx].second;
}

std::string IPCServer::get_current_video_path() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (video_playlist_.empty()) return "";
    int sz = static_cast<int>(video_playlist_.size());
    int idx = (video_idx_ % sz + sz) % sz;
    return video_playlist_[idx].second;
}

int IPCServer::get_active_index() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (current_state_.mode == "VIDEO") return video_idx_;
    return current_state_.active_idx;
}

std::string IPCServer::get_active_mode() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_.mode;
}

std::string IPCServer::get_pipeline_mode() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_.pipeline_mode;
}

bool IPCServer::is_paused() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_.is_paused;
}

void IPCServer::update_frame_jpeg(const std::vector<uint8_t>& jpeg_data) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    current_jpeg_ = jpeg_data;
}

std::vector<uint8_t> IPCServer::get_frame_jpeg() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return current_jpeg_;
}

void IPCServer::next_sample() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (current_state_.mode == "VIDEO") {
        if (video_playlist_.empty()) return;
        sample_generation_id_++;
        video_idx_ = (video_idx_ + 1) % static_cast<int>(video_playlist_.size());
        reset_analyzing_state(current_state_);
        current_state_.active_tag = "VIDEO [" + std::to_string(video_idx_ + 1) + "/" +
                                    std::to_string(video_playlist_.size()) + "]";
        return;
    }
    if (playlist_.empty()) return;
    sample_generation_id_++;
    current_state_.active_idx = (current_state_.active_idx + 1) % static_cast<int>(playlist_.size());
    reset_analyzing_state(current_state_);
    current_state_.active_tag = current_state_.mode + " " + std::to_string(current_state_.active_idx + 1) + "/" +
                                std::to_string(playlist_.size()) + " · " + playlist_[current_state_.active_idx].first;
}

void IPCServer::prev_sample() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (current_state_.mode == "VIDEO") {
        int vsz = static_cast<int>(video_playlist_.size());
        if (vsz == 0) return;
        sample_generation_id_++;
        video_idx_ = (video_idx_ - 1 + vsz) % vsz;
        reset_analyzing_state(current_state_);
        current_state_.active_tag = "VIDEO [" + std::to_string(video_idx_ + 1) + "/" +
                                    std::to_string(vsz) + "]";
        return;
    }
    if (playlist_.empty()) return;
    sample_generation_id_++;
    int sz = static_cast<int>(playlist_.size());
    current_state_.active_idx = (current_state_.active_idx - 1 + sz) % sz;
    reset_analyzing_state(current_state_);
    current_state_.active_tag = current_state_.mode + " " + std::to_string(current_state_.active_idx + 1) + "/" +
                                std::to_string(sz) + " · " + playlist_[current_state_.active_idx].first;
}

void IPCServer::select_sample(int index) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (current_state_.mode == "VIDEO") {
        int vsz = static_cast<int>(video_playlist_.size());
        if (vsz == 0) return;
        sample_generation_id_++;
        video_idx_ = std::max(0, std::min(index, vsz - 1));
        reset_analyzing_state(current_state_);
        current_state_.active_tag = "VIDEO [" + std::to_string(video_idx_ + 1) + "/" +
                                    std::to_string(vsz) + "]";
        return;
    }
    int sz = static_cast<int>(playlist_.size());
    if (sz == 0) return;
    sample_generation_id_++;
    current_state_.active_idx = std::max(0, std::min(index, sz - 1));
    reset_analyzing_state(current_state_);
    current_state_.active_tag = current_state_.mode + " " + std::to_string(current_state_.active_idx + 1) + "/" +
                                std::to_string(sz) + " · " + playlist_[current_state_.active_idx].first;
}

void IPCServer::select_sample_by_path(const std::string& path) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    fs::path target_p(path);
    std::string fname = target_p.filename().string();
    if (current_state_.mode == "VIDEO") {
        for (size_t i = 0; i < video_playlist_.size(); ++i) {
            if (video_playlist_[i].first == fname || video_playlist_[i].second == path) {
                sample_generation_id_++;
                video_idx_ = static_cast<int>(i);
                reset_analyzing_state(current_state_);
                current_state_.active_tag = "VIDEO [" + std::to_string(i + 1) + "/" +
                                            std::to_string(video_playlist_.size()) + "]";
                return;
            }
        }
    }
    for (size_t i = 0; i < playlist_.size(); ++i) {
        if (playlist_[i].first == fname || playlist_[i].second == path) {
            sample_generation_id_++;
            current_state_.active_idx = static_cast<int>(i);
            reset_analyzing_state(current_state_);
            if (current_state_.mode == "IMAGE") {
                current_state_.active_tag = "IMAGE [" + std::to_string(i + 1) + "/" +
                                            std::to_string(playlist_.size()) + "]";
            } else {
                current_state_.active_tag = current_state_.mode + " " + std::to_string(i + 1) + "/" +
                                            std::to_string(playlist_.size()) + " · " + playlist_[i].first;
            }
            break;
        }
    }
}

void IPCServer::set_active_index(int idx) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    int sz = static_cast<int>(playlist_.size());
    if (sz == 0) return;
    sample_generation_id_++;
    current_state_.active_idx = (idx % sz + sz) % sz;
    reset_analyzing_state(current_state_);
    if (current_state_.mode == "DEMO") {
        current_state_.active_tag = "DEMO " + std::to_string(current_state_.active_idx + 1) + "/" +
                                    std::to_string(sz) + " · Industrial Conveyor";
    } else if (current_state_.mode == "IMAGE") {
        current_state_.active_tag = "IMAGE [" + std::to_string(current_state_.active_idx + 1) + "/" +
                                    std::to_string(sz) + "]";
    } else {
        current_state_.active_tag = current_state_.mode + " " + std::to_string(current_state_.active_idx + 1) + "/" +
                                    std::to_string(sz) + " · " + playlist_[current_state_.active_idx].first;
    }
}

size_t IPCServer::get_playlist_size() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (current_state_.mode == "VIDEO") return video_playlist_.size();
    return playlist_.size();
}

std::string IPCServer::get_image_path_at(int idx) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (playlist_.empty()) return "";
    int sz = static_cast<int>(playlist_.size());
    int real_idx = (idx % sz + sz) % sz;
    return playlist_[real_idx].second;
}

void IPCServer::toggle_pause() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_state_.is_paused = !current_state_.is_paused;
}

void IPCServer::set_pipeline_mode(const std::string& mode) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_state_.pipeline_mode = mode;
    sample_generation_id_++;
    if (current_state_.mode != "LOG") {
        reset_analyzing_state(current_state_);
    }
    if (mode == "MODE_2") {
        current_state_.active_pipeline_mode = "MODE_2";
        current_state_.is_mode2 = true;
    } else if (mode == "MODE_1") {
        current_state_.active_pipeline_mode = "MODE_1";
        current_state_.is_mode2 = false;
    }
}

void IPCServer::switch_mode(const std::string& mode) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_state_.mode = mode;
    sample_generation_id_++;
    if (mode != "LOG") {
        current_state_.is_paused = false;
        reset_analyzing_state(current_state_);
    }
    if (mode == "DEMO" && !playlist_.empty()) {
        current_state_.active_tag = "DEMO " + std::to_string(current_state_.active_idx + 1) + "/" +
                                    std::to_string(playlist_.size()) + " · Industrial Conveyor";
    } else if (mode == "IMAGE" && !playlist_.empty()) {
        current_state_.active_tag = "IMAGE [" + std::to_string(current_state_.active_idx + 1) + "/" +
                                    std::to_string(playlist_.size()) + "]";
    } else if (mode == "VIDEO") {
        current_state_.active_tag = "VIDEO [" + std::to_string(video_playlist_.empty() ? 0 : video_idx_ + 1) + "/" +
                                    std::to_string(video_playlist_.size()) + "]";
    } else if (mode == "CAMERA") {
        current_state_.active_tag = "LIVE CAM · Stream";
    } else if (mode == "LOG") {
        current_state_.active_tag = "CSV LOGBOOK";
    }
}

uint64_t IPCServer::get_sample_generation_id() const {
    return sample_generation_id_.load();
}

void IPCServer::update_state_for_generation(uint64_t gen_id, const LiveTelemetryState& state) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (sample_generation_id_.load() != gen_id) {
        // Discard stale result: user has navigated to another item
        return;
    }
    std::string m = current_state_.mode;
    std::string pm = current_state_.pipeline_mode;
    int idx = current_state_.active_idx;
    bool paused = current_state_.is_paused;

    current_state_ = state;
    current_state_.mode = m;
    current_state_.pipeline_mode = pm;
    current_state_.active_idx = idx;
    current_state_.is_paused = paused;
}

void IPCServer::update_state(const LiveTelemetryState& state) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    // Keep user interactive selections
    std::string m = current_state_.mode;
    std::string pm = current_state_.pipeline_mode;
    int idx = current_state_.active_idx;
    bool paused = current_state_.is_paused;

    current_state_ = state;
    current_state_.mode = m;
    current_state_.pipeline_mode = pm;
    current_state_.active_idx = idx;
    current_state_.is_paused = paused;
}

LiveTelemetryState IPCServer::get_state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_;
}

void IPCServer::load_logs_from_csv() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    inspection_logs_.clear();
    if (!fs::exists(CSV_LOG_PATH)) {
        return;
    }
    std::ifstream in(CSV_LOG_PATH);
    if (!in.is_open()) return;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto cols = parse_csv_line(line);
        if (cols.empty() || cols[0] == "Timestamp" || cols[0] == "timestamp") continue;

        InspectionRecord rec;
        if (cols.size() >= 14) {
            rec.timestamp = cols[0];
            rec.mode = cols[1];
            rec.part = cols[2];
            rec.defect = cols[3];
            rec.score_conf = cols[4];
            rec.score_ratio = cols[5];
            try { rec.area_pct = std::stof(cols[6]); } catch (...) { rec.area_pct = 0.0f; }
            try { rec.regions_boxes = std::stoi(cols[7]); } catch (...) { rec.regions_boxes = 0; }
            rec.spread = cols[8];
            rec.severity = cols[9];
            rec.cause = cols[10];
            rec.action = cols[11];
            rec.source = cols[12];
            try { rec.latency_ms = std::stof(cols[13]); } catch (...) { rec.latency_ms = 0.0f; }
            inspection_logs_.push_back(std::move(rec));
        } else if (cols.size() >= 8) {
            // Backward compatibility with legacy 8-column CSV rows
            rec.timestamp = cols[0];
            rec.defect = cols[1];
            try {
                float c = std::stof(cols[2]);
                rec.score_conf = format_one_decimal(c <= 1.0f ? c * 100.0f : c) + "%";
            } catch (...) {
                rec.score_conf = cols[2];
            }
            rec.severity = cols[3];
            try { rec.area_pct = std::stof(cols[4]); } catch (...) { rec.area_pct = 0.0f; }
            try { rec.regions_boxes = std::stoi(cols[5]); } catch (...) { rec.regions_boxes = 1; }
            rec.cause = cols[6];
            rec.action = cols[7];
            inspection_logs_.push_back(std::move(rec));
        }
    }
    if (inspection_logs_.size() > 500) {
        inspection_logs_.erase(inspection_logs_.begin(), inspection_logs_.end() - 500);
    }
}

void IPCServer::append_log(const InspectionRecord& record) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    inspection_logs_.push_back(record);
    if (inspection_logs_.size() > 500) {
        inspection_logs_.erase(inspection_logs_.begin());
    }

    bool need_header = !fs::exists(CSV_LOG_PATH) || fs::file_size(CSV_LOG_PATH) == 0;
    std::ofstream csv(CSV_LOG_PATH, std::ios::app);
    if (csv.is_open()) {
        if (need_header) {
            csv << CSV_HEADER_LINE << "\n";
        }
        csv << escape_csv(record.timestamp) << ","
            << escape_csv(record.mode) << ","
            << escape_csv(record.part) << ","
            << escape_csv(record.defect) << ","
            << escape_csv(record.score_conf) << ","
            << escape_csv(record.score_ratio) << ","
            << std::fixed << std::setprecision(2) << record.area_pct << ","
            << record.regions_boxes << ","
            << escape_csv(record.spread) << ","
            << escape_csv(record.severity) << ","
            << escape_csv(record.cause) << ","
            << escape_csv(record.action) << ","
            << escape_csv(record.source) << ","
            << static_cast<long>(std::lround(record.latency_ms)) << "\n";
    }
}

std::string IPCServer::get_logs_json() const {
    std::lock_guard<std::mutex> lock(log_mutex_);
    json arr = json::array();
    for (auto it = inspection_logs_.rbegin(); it != inspection_logs_.rend(); ++it) {
        std::string sev_val = it->severity.empty() ? "NONE" : it->severity;
        arr.push_back({
            {"ts", it->timestamp},
            {"mode", it->mode},
            {"part", it->part},
            {"defect", it->defect},
            {"score_conf", it->score_conf},
            {"score_ratio", it->score_ratio},
            {"area_pct", format_one_decimal(it->area_pct)},
            {"regions_boxes", it->regions_boxes},
            {"spread", it->spread.empty() ? "-" : it->spread},
            {"sev", sev_val},
            {"cause", it->cause},
            {"action", it->action},
            {"src", it->source},
            {"ms", static_cast<long>(std::lround(it->latency_ms))}
        });
    }
    return arr.dump();
}

void IPCServer::clear_logs() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    inspection_logs_.clear();
    std::ofstream csv(CSV_LOG_PATH, std::ios::trunc);
    if (csv.is_open()) {
        csv << CSV_HEADER_LINE << "\n";
    }
}

std::string IPCServer::get_csv_logs() const {
    std::lock_guard<std::mutex> lock(log_mutex_);
    std::stringstream ss;
    ss << CSV_HEADER_LINE << "\n";
    for (const auto& r : inspection_logs_) {
        ss << escape_csv(r.timestamp) << ","
           << escape_csv(r.mode) << ","
           << escape_csv(r.part) << ","
           << escape_csv(r.defect) << ","
           << escape_csv(r.score_conf) << ","
           << escape_csv(r.score_ratio) << ","
           << std::fixed << std::setprecision(2) << r.area_pct << ","
           << r.regions_boxes << ","
           << escape_csv(r.spread) << ","
           << escape_csv(r.severity) << ","
           << escape_csv(r.cause) << ","
           << escape_csv(r.action) << ","
           << escape_csv(r.source) << ","
           << static_cast<long>(std::lround(r.latency_ms)) << "\n";
    }
    return ss.str();
}

std::string IPCServer::serialize_json_state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    json j;
    j["mode"] = current_state_.mode;

    int sz = static_cast<int>(playlist_.size());
    int idx = current_state_.active_idx;
    if (current_state_.mode == "DEMO") {
        j["active_tag"] = "DEMO " + std::to_string(idx + 1) + "/" + std::to_string(sz) + " · Industrial Conveyor";
    } else if (current_state_.mode == "IMAGE") {
        j["active_tag"] = "IMAGE [" + std::to_string(idx + 1) + "/" + std::to_string(sz) + "]";
    } else if (current_state_.mode == "VIDEO") {
        int vsz = static_cast<int>(video_playlist_.size());
        j["active_tag"] = "VIDEO [" + std::to_string(vsz > 0 ? video_idx_ + 1 : 0) + "/" + std::to_string(vsz) + "]";
    } else if (current_state_.mode == "CAMERA") {
        j["active_tag"] = "LIVE CAM · Stream";
    } else {
        j["active_tag"] = "CSV LOGBOOK";
    }

    j["pipeline_mode"] = current_state_.pipeline_mode;
    j["active_pipeline_mode"] = current_state_.active_pipeline_mode;
    j["stage0_route"] = current_state_.stage0_route;
    j["stage0_conf"] = current_state_.stage0_conf;
    j["is_mode2"] = current_state_.is_mode2;
    j["defect"] = current_state_.defect;
    j["conf"] = current_state_.conf;
    j["score_ratio"] = current_state_.score_ratio;
    j["area"] = current_state_.area;
    j["boxes"] = current_state_.boxes;
    j["regions"] = current_state_.regions;
    j["spread"] = current_state_.spread;
    j["severity"] = current_state_.severity;
    j["cause"] = current_state_.cause;
    j["action"] = current_state_.action;
    j["tokens"] = current_state_.tokens;
    j["tok_speed"] = current_state_.tok_speed;
    j["cv_ms"] = current_state_.cv_ms;
    j["engine_source"] = current_state_.engine_source;
    j["llm_ms"] = current_state_.llm_ms;
    j["llm_busy"] = current_state_.llm_busy;
    j["llm_elapsed"] = current_state_.llm_elapsed;
    j["is_locked"] = current_state_.is_locked;
    j["lock_cnt"] = current_state_.lock_cnt;
    j["is_paused"] = current_state_.is_paused;
    j["engine_mode"] = "LLAMA_ONLY";
    j["vision_hardware"] = current_state_.vision_hardware;
    j["system_status"] = current_state_.vision_hardware + " · LLAMA 3.2 1B";
    j["active_idx"] = (current_state_.mode == "VIDEO") ? video_idx_ : current_state_.active_idx;

    // In DEMO, CAMERA, and LOG modes, return empty playlist
    j["playlist"] = json::array();
    if (current_state_.mode == "IMAGE") {
        for (const auto& item : playlist_) {
            j["playlist"].push_back({{"name", item.first}, {"path", item.second}});
        }
    } else if (current_state_.mode == "VIDEO") {
        for (const auto& item : video_playlist_) {
            j["playlist"].push_back({{"name", item.first}, {"path", item.second}});
        }
    }

    return j.dump();
}

bool IPCServer::start() {
    is_running_ = true;
    server_thread_ = std::thread(&IPCServer::server_loop, this);
    return true;
}

void IPCServer::stop() {
    if (is_running_) {
        is_running_ = false;
        if (svr_) {
            svr_->stop();
        }
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }
}

void IPCServer::server_loop() {
    hardware::set_thread_name("scada_http_c0");
    hardware::pin_current_thread(hardware::CORE_IO_IPC);

    std::cout << "[IPCServer] Pure C++ Native SCADA Web Server listening on port " << port_
              << " (" << hardware::get_current_affinity_string() << ")" << std::endl;

    // 1. Embedded Web UI Routes
    svr_->Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(ui::get_index_html(), "text/html; charset=utf-8");
    });
    svr_->Get("/index.html", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(ui::get_index_html(), "text/html; charset=utf-8");
    });

    // 2. Real Telemetry State API
    svr_->Get("/api/state", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(serialize_json_state(), "application/json");
    });
    svr_->Get("/api/telemetry", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(serialize_json_state(), "application/json");
    });

    // 3. Image / Frame Stream
    svr_->Get("/api/frame.jpg", [this](const httplib::Request&, httplib::Response& res) {
        std::vector<uint8_t> frame = get_frame_jpeg();
        if (!frame.empty()) {
            res.set_header("Cache-Control", "no-cache, no-store, must-revalidate");
            res.set_content(reinterpret_cast<const char*>(frame.data()), frame.size(), "image/jpeg");
        } else {
            res.status = 404;
        }
    });
    svr_->Get("/api/stream", [this](const httplib::Request&, httplib::Response& res) {
        std::vector<uint8_t> frame = get_frame_jpeg();
        if (!frame.empty()) {
            res.set_header("Cache-Control", "no-cache, no-store, must-revalidate");
            res.set_content(reinterpret_cast<const char*>(frame.data()), frame.size(), "image/jpeg");
        } else {
            res.status = 404;
        }
    });

    // 4. Playlist & Controls
    svr_->Get("/api/next", [this](const httplib::Request&, httplib::Response& res) {
        next_sample();
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr_->Get("/api/prev", [this](const httplib::Request&, httplib::Response& res) {
        prev_sample();
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr_->Get("/api/select_item", [this](const httplib::Request& req, httplib::Response& res) {
        if (req.has_param("path")) {
            select_sample_by_path(req.get_param_value("path"));
        } else if (req.has_param("idx")) {
            select_sample(std::stoi(req.get_param_value("idx")));
        } else if (req.has_param("index")) {
            select_sample(std::stoi(req.get_param_value("index")));
        }
        res.set_content("{\"status\":\"ok\",\"success\":true}", "application/json");
    });
    svr_->Get("/api/switch_mode", [this](const httplib::Request& req, httplib::Response& res) {
        std::string mode = req.has_param("mode") ? req.get_param_value("mode") : "DEMO";
        switch_mode(mode);
        res.set_content("{\"mode\":\"" + mode + "\",\"success\":true}", "application/json");
    });
    svr_->Get("/api/set_pipeline_mode", [this](const httplib::Request& req, httplib::Response& res) {
        std::string mode = req.has_param("mode") ? req.get_param_value("mode") : "AUTO";
        set_pipeline_mode(mode);
        res.set_content("{\"success\":true,\"pipeline_mode\":\"" + mode + "\"}", "application/json");
    });
    svr_->Get("/api/toggle_pause", [this](const httplib::Request&, httplib::Response& res) {
        toggle_pause();
        bool paused = is_paused();
        res.set_content("{\"is_paused\":" + std::string(paused ? "true" : "false") + ",\"success\":true}", "application/json");
    });

    // 5. File Uploads (Multipart Form Data)
    svr_->Post("/api/upload", [this](const httplib::Request& req, httplib::Response& res) {
        if (!req.is_multipart_form_data() || req.files.empty()) {
            res.set_content("{\"success\":false,\"status\":\"error\",\"error\":\"No file uploaded\"}", "application/json");
            return;
        }
        fs::create_directories("uploads");
        std::string last_media_type = "IMAGE";
        std::string last_name;
        bool saved_any = false;

        for (const auto& file : req.files) {
            std::string clean_name = fs::path(file.second.filename).filename().string();
            if (clean_name.empty()) continue;

            std::string ext = fs::path(clean_name).extension().string();
            bool is_vid = is_video_extension(ext);
            bool is_img = is_image_extension(ext);
            if (!is_vid && !is_img) continue;

            std::string save_path = "uploads/" + clean_name;
            std::ofstream out(save_path, std::ios::binary);
            if (!out.is_open()) continue;
            out.write(file.second.content.data(), file.second.content.size());
            out.close();

            std::lock_guard<std::mutex> lock(state_mutex_);
            sample_generation_id_++;
            current_state_.is_paused = false;
            reset_analyzing_state(current_state_);
            if (is_vid) {
                auto it = std::find_if(video_playlist_.begin(), video_playlist_.end(),
                                       [&](const auto& p) { return p.second == save_path || p.first == clean_name; });
                if (it != video_playlist_.end()) video_playlist_.erase(it);
                video_playlist_.insert(video_playlist_.begin(), {clean_name, save_path});
                video_idx_ = 0;
                current_state_.mode = "VIDEO";
                current_state_.active_tag = "VIDEO [1/" + std::to_string(video_playlist_.size()) + "]";
                last_media_type = "VIDEO";
            } else {
                auto it = std::find_if(playlist_.begin(), playlist_.end(),
                                       [&](const auto& p) { return p.second == save_path || p.first == clean_name; });
                if (it != playlist_.end()) playlist_.erase(it);
                playlist_.insert(playlist_.begin(), {clean_name, save_path});
                current_state_.active_idx = 0;
                current_state_.mode = "IMAGE";
                current_state_.active_tag = "IMAGE [1/" + std::to_string(playlist_.size()) + "]";
                last_media_type = "IMAGE";
            }
            last_name = clean_name;
            saved_any = true;
        }

        if (!saved_any) {
            res.set_content("{\"success\":false,\"status\":\"error\",\"error\":\"Unsupported or empty upload file\"}", "application/json");
            return;
        }

        json ok_resp = {
            {"success", true},
            {"status", "ok"},
            {"media_type", last_media_type},
            {"name", last_name},
            {"message", "Uploaded successfully"}
        };
        res.set_content(ok_resp.dump(), "application/json");
    });

    // 6. Logs & Reports
    svr_->Get("/api/logs", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(get_logs_json(), "application/json");
    });
    svr_->Post("/api/clear_logs", [this](const httplib::Request&, httplib::Response& res) {
        clear_logs();
        res.set_content("{\"success\":true,\"status\":\"ok\",\"message\":\"Logs cleared\"}", "application/json");
    });
    svr_->Get("/api/download_csv", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Disposition", "attachment; filename=rpi5_qc_inspection_log.csv");
        res.set_content(get_csv_logs(), "text/csv");
    });
    svr_->Post("/api/save_frame", [this](const httplib::Request&, httplib::Response& res) {
        std::vector<uint8_t> frame = get_frame_jpeg();
        if (!frame.empty()) {
            fs::create_directories("snapshots");
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << "snapshots/snap_" << t << ".jpg";
            std::string snap_path = ss.str();
            std::ofstream out(snap_path, std::ios::binary);
            if (out.is_open()) {
                out.write(reinterpret_cast<const char*>(frame.data()), frame.size());
            }
            res.set_content("{\"success\":true,\"filename\":\"" + snap_path + "\"}", "application/json");
            return;
        }
        res.set_content("{\"success\":false}", "application/json");
    });

    // Start server listening
    svr_->listen("0.0.0.0", port_);
}

} // namespace ipc
} // namespace qc
