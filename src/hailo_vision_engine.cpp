#include "hailo_vision_engine.hpp"
#include "metallic_gate_neon.hpp"
#include "qc_rules.hpp"
#include "cpu_affinity.hpp"

#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <onnxruntime_cxx_api.h>

#ifdef HAVE_HAILORT
#include "hailo/hailort.hpp"
#endif

namespace qc {
namespace vision {

static const std::vector<std::string> STEEL_CLASSES = {
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"
};

const std::vector<std::string>& HailoVisionEngine::get_steel_classes() {
    return STEEL_CLASSES;
}

// Color palette matching Python SCADA Dark Theme
static const std::map<std::string, cv::Scalar> CLASS_COLORS_BGR = {
    {"crazing", cv::Scalar(77, 77, 255)},        // Coral Red
    {"inclusion", cv::Scalar(0, 160, 255)},      // Amber
    {"patches", cv::Scalar(255, 229, 0)},        // Cyan
    {"pitted_surface", cv::Scalar(251, 64, 224)},// Magenta
    {"rolled-in_scale", cv::Scalar(0, 145, 255)},// Orange
    {"scratches", cv::Scalar(0, 215, 255)},      // Gold Yellow
    {"anomaly", cv::Scalar(0, 70, 255)},         // Vibrant Red
    {"bent", cv::Scalar(0, 70, 255)},
    {"color", cv::Scalar(0, 160, 255)},
    {"flip", cv::Scalar(255, 145, 0)},
    {"scratch", cv::Scalar(0, 215, 255)}
};

namespace {

struct OrtSessionHolder {
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_cstrs;
    std::vector<const char*> output_cstrs;
    bool ready = false;

    bool load(Ort::Env& env, const Ort::SessionOptions& opts, const std::string& model_path, const char* tag) {
        if (model_path.empty()) return false;
        try {
            session = std::make_unique<Ort::Session>(env, model_path.c_str(), opts);
            Ort::AllocatorWithDefaultOptions allocator;

            size_t n_in = session->GetInputCount();
            input_names.reserve(n_in);
            input_cstrs.reserve(n_in);
            for (size_t i = 0; i < n_in; ++i) {
                auto name_alloc = session->GetInputNameAllocated(i, allocator);
                input_names.emplace_back(name_alloc.get());
            }
            for (const auto& s : input_names) input_cstrs.push_back(s.c_str());

            size_t n_out = session->GetOutputCount();
            output_names.reserve(n_out);
            output_cstrs.reserve(n_out);
            for (size_t i = 0; i < n_out; ++i) {
                auto name_alloc = session->GetOutputNameAllocated(i, allocator);
                output_names.emplace_back(name_alloc.get());
            }
            for (const auto& s : output_names) output_cstrs.push_back(s.c_str());

            ready = true;
            std::cout << "[HailoVisionEngine] Loaded " << tag << " (ONNX Runtime): " << model_path << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[HailoVisionEngine] Failed to load " << tag << " (" << model_path << "): " << e.what() << std::endl;
            ready = false;
            return false;
        }
    }
};

/**
 * Resize BGR frame to (target_w, target_h), convert to RGB float32 [0,1] in NCHW layout.
 */
void bgr_to_nchw_float(const cv::Mat& frame_bgr, int target_w, int target_h, std::vector<float>& out, cv::Mat* out_rgb = nullptr) {
    cv::Mat resized;
    if (frame_bgr.cols == target_w && frame_bgr.rows == target_h) {
        resized = frame_bgr;
    } else {
        cv::resize(frame_bgr, resized, cv::Size(target_w, target_h));
    }

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    if (out_rgb) {
        *out_rgb = rgb;
    }

    const int hw = target_w * target_h;
    out.resize(3 * hw);
    constexpr float inv255 = 1.0f / 255.0f;
    for (int y = 0; y < target_h; ++y) {
        const uint8_t* row = rgb.ptr<uint8_t>(y);
        int row_off = y * target_w;
        for (int x = 0; x < target_w; ++x) {
            int p = row_off + x;
            out[0 * hw + p] = static_cast<float>(row[x * 3 + 0]) * inv255;
            out[1 * hw + p] = static_cast<float>(row[x * 3 + 1]) * inv255;
            out[2 * hw + p] = static_cast<float>(row[x * 3 + 2]) * inv255;
        }
    }
}

} // namespace

struct HailoVisionEngine::Impl {
    // Unified ONNX Runtime Environment & Sessions (works identically across x86_64 & ARM64 NEON)
    std::unique_ptr<Ort::Env> ort_env;
    OrtSessionHolder yolo_ort;
    OrtSessionHolder stage0_ort;
    OrtSessionHolder mode2_ort;

#ifdef HAVE_HAILORT
    std::unique_ptr<hailort::VDevice> vdevice;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group;
    std::unique_ptr<hailort::InputVStream> input_vstream;
    std::vector<std::unique_ptr<hailort::OutputVStream>> output_vstreams;
#endif
    bool hailo_configured = false;
};

HailoVisionEngine::HailoVisionEngine() : impl_(std::make_unique<Impl>()) {}
HailoVisionEngine::~HailoVisionEngine() = default;

bool HailoVisionEngine::init(
    const std::string& hef_path,
    const std::string& yolo_onnx_path,
    const std::string& stage0_onnx_path,
    const std::string& mode2_onnx_path
) {
    hef_path_ = hef_path;
    yolo_onnx_path_ = yolo_onnx_path;
    stage0_onnx_path_ = stage0_onnx_path;
    mode2_onnx_path_ = mode2_onnx_path;
    is_hailo_ready_ = false;

    // 1. Hailo-8L PCIe Hardware Check
    std::ifstream hailo_node("/dev/hailo0");
    bool node_exists = hailo_node.is_open();

    if (node_exists && !hef_path_.empty()) {
#ifdef HAVE_HAILORT
        try {
            auto vdevice_exp = hailort::VDevice::create();
            if (vdevice_exp) {
                impl_->vdevice = vdevice_exp.release();
                auto hef_exp = hailort::Hef::create_from_file(hef_path_);
                if (hef_exp) {
                    hailort::Hef hef = hef_exp.release();
                    auto configure_params = impl_->vdevice->create_configure_params(hef);
                    if (configure_params) {
                        auto network_groups = impl_->vdevice->configure(hef, configure_params.value());
                        if (network_groups && !network_groups.value().empty()) {
                            impl_->network_group = network_groups.value()[0];
                            impl_->hailo_configured = true;
                            is_hailo_ready_ = true;
                            hardware_name_ = "Hailo-8L (13 TOPS AI HAT)";
                            std::cout << "[HailoVisionEngine] PCIe 13 TOPS AI HAT initialized with " << hef_path_ << "!" << std::endl;
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[HailoVisionEngine] Hailo init notice: " << e.what() << ". Using CPU vision." << std::endl;
        }
#endif
    }

    if (!is_hailo_ready_) {
        hardware_name_ = "ARM NEON CPU (ONNX Runtime)";
    }

    // 2. Initialize Unified ONNX Runtime for Stage 0, YOLO11n, and Mode 2 EfficientAD-S
    impl_->ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "BharatVisionEngine");
    Ort::SessionOptions opts;
    int threads = std::max(1, std::min(4, qc::hardware::get_cpu_core_count()));
    opts.SetIntraOpNumThreads(threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    impl_->yolo_ort.load(*impl_->ort_env, opts, yolo_onnx_path_, "YOLO11n Mode1");
    impl_->stage0_ort.load(*impl_->ort_env, opts, stage0_onnx_path_, "Stage0 Router");
    impl_->mode2_ort.load(*impl_->ort_env, opts, mode2_onnx_path_, "Mode2 EfficientAD-S");

    return impl_->yolo_ort.ready || impl_->mode2_ort.ready;
}

Stage0Result HailoVisionEngine::classify_stage0(const cv::Mat& frame_bgr) {
    Stage0Result res;
    if (frame_bgr.empty() || !impl_->stage0_ort.ready) {
        res.route = "MODE_1";
        res.class_name = "steel_sheet";
        res.confidence = 1.0f;
        res.latency_ms = 0.0f;
        return res;
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> input_tensor_values;
    bgr_to_nchw_float(frame_bgr, 224, 224, input_tensor_values);

    std::vector<int64_t> in_shape = {1, 3, 224, 224};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor_values.data(), input_tensor_values.size(),
        in_shape.data(), in_shape.size()
    );

    auto outputs = impl_->stage0_ort.session->Run(
        Ort::RunOptions{nullptr},
        impl_->stage0_ort.input_cstrs.data(), &in_tensor, 1,
        impl_->stage0_ort.output_cstrs.data(), impl_->stage0_ort.output_cstrs.size()
    );

    auto t1 = std::chrono::steady_clock::now();
    res.latency_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    const float* p = outputs[0].GetTensorData<float>();
    float p_part = p[0];
    float p_steel = p[1];
    float p_unclear = p[2];

    if (p_steel >= 0.50f && p_steel > p_part) {
        res.route = "MODE_1";
        res.class_name = "steel_sheet";
        res.confidence = p_steel;
    } else if (p_part >= 0.60f && p_part > p_steel) {
        res.route = "MODE_2";
        res.class_name = "part";
        res.confidence = p_part;
    } else {
        res.route = "unclear";
        res.class_name = "unclear";
        res.confidence = std::max({p_part, p_steel, p_unclear});
    }

    return res;
}

float HailoVisionEngine::compute_union_area(const std::vector<DetectionBox>& boxes, int w, int h) {
    if (boxes.empty() || w <= 0 || h <= 0) return 0.0f;
    cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
    for (const auto& b : boxes) {
        int x1 = std::max(0, std::min(w, b.x1));
        int y1 = std::max(0, std::min(h, b.y1));
        int x2 = std::max(0, std::min(w, b.x2));
        int y2 = std::max(0, std::min(h, b.y2));
        if (x2 > x1 && y2 > y1) {
            cv::rectangle(mask, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), -1);
        }
    }
    int nonzeros = cv::countNonZero(mask);
    return (static_cast<float>(nonzeros) / static_cast<float>(w * h)) * 100.0f;
}

void HailoVisionEngine::draw_mode1_overlay(
    cv::Mat& frame,
    const std::vector<DetectionBox>& boxes,
    const std::string& title,
    const std::string& sev
) {
    (void)title;
    (void)sev;
    int h_orig = frame.rows;
    int w_orig = frame.cols;
    int target_dim = 720;
    float scale = 1.0f;
    if (std::max(w_orig, h_orig) < target_dim) {
        scale = static_cast<float>(target_dim) / static_cast<float>(std::max(w_orig, h_orig));
        int disp_w = static_cast<int>(w_orig * scale);
        int disp_h = static_cast<int>(h_orig * scale);
        cv::resize(frame, frame, cv::Size(disp_w, disp_h), 0, 0, cv::INTER_LINEAR);
    }
    int disp_w = frame.cols;
    int disp_h = frame.rows;

    for (const auto& b : boxes) {
        int x1 = std::max(0, std::min(disp_w - 1, static_cast<int>(b.x1 * scale)));
        int y1 = std::max(0, std::min(disp_h - 1, static_cast<int>(b.y1 * scale)));
        int x2 = std::max(0, std::min(disp_w - 1, static_cast<int>(b.x2 * scale)));
        int y2 = std::max(0, std::min(disp_h - 1, static_cast<int>(b.y2 * scale)));

        cv::Scalar color = cv::Scalar(0, 255, 0);
        auto it = CLASS_COLORS_BGR.find(b.class_name);
        if (it != CLASS_COLORS_BGR.end()) color = it->second;

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2, cv::LINE_AA);

        std::string c_disp = b.class_name;
        for (auto& ch : c_disp) { if (ch == '_') ch = ' '; }
        if (!c_disp.empty()) c_disp[0] = std::toupper(c_disp[0]);
        std::string lbl_text = c_disp + " " + std::to_string(static_cast<int>(b.confidence * 100.0f)) + "%";

        int baseline = 0;
        cv::Size sz = cv::getTextSize(lbl_text, cv::FONT_HERSHEY_SIMPLEX, 0.50, 1, &baseline);
        int pill_w = sz.width + 18;
        int pill_h = sz.height + 10;
        int px1 = std::max(4, std::min(disp_w - pill_w - 4, x1));
        int py1 = (y1 >= pill_h + 4) ? (y1 - pill_h) : std::min(disp_h - pill_h - 4, y1 + 4);

        cv::rectangle(frame, cv::Rect(px1, py1, pill_w, pill_h), cv::Scalar(14, 16, 19), -1);
        cv::rectangle(frame, cv::Rect(px1, py1, pill_w, pill_h), color, 1, cv::LINE_AA);
        cv::rectangle(frame, cv::Rect(px1, py1, 4, pill_h), color, -1);
        cv::putText(frame, lbl_text, cv::Point(px1 + 8, py1 + sz.height + 3), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(240, 243, 246), 1, cv::LINE_AA);
    }
}

void HailoVisionEngine::draw_mode2_overlay(
    cv::Mat& frame,
    const cv::Mat& amap,
    const std::vector<DetectionBox>& boxes,
    const std::string& title,
    const std::string& sev,
    float score,
    float area_pct
) {
    (void)title;
    (void)sev;
    (void)area_pct;
    int h_orig = frame.rows;
    int w_orig = frame.cols;
    int target_dim = 720;
    float scale = 1.0f;
    if (std::max(w_orig, h_orig) < target_dim) {
        scale = static_cast<float>(target_dim) / static_cast<float>(std::max(w_orig, h_orig));
        int disp_w = static_cast<int>(w_orig * scale);
        int disp_h = static_cast<int>(h_orig * scale);
        cv::resize(frame, frame, cv::Size(disp_w, disp_h), 0, 0, cv::INTER_LINEAR);
    }
    int disp_w = frame.cols;
    int disp_h = frame.rows;

    if (!amap.empty()) {
        double min_v, max_v;
        cv::minMaxLoc(amap, &min_v, &max_v);
        cv::Mat norm_map;
        amap.convertTo(norm_map, CV_8UC1, 255.0 / (max_v > 1e-4 ? max_v : 1.0));
        cv::Mat heatmap;
        cv::applyColorMap(norm_map, heatmap, cv::COLORMAP_JET);
        cv::Mat resized_hm;
        cv::resize(heatmap, resized_hm, frame.size());
        cv::addWeighted(frame, 0.60, resized_hm, 0.40, 0, frame);
    }

    for (const auto& b : boxes) {
        int x1 = std::max(0, std::min(disp_w - 1, static_cast<int>(b.x1 * scale)));
        int y1 = std::max(0, std::min(disp_h - 1, static_cast<int>(b.y1 * scale)));
        int x2 = std::max(0, std::min(disp_w - 1, static_cast<int>(b.x2 * scale)));
        int y2 = std::max(0, std::min(disp_h - 1, static_cast<int>(b.y2 * scale)));

        cv::Scalar box_color(0, 77, 255);
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 2, cv::LINE_AA);

        std::string sub_upper = b.class_name;
        std::transform(sub_upper.begin(), sub_upper.end(), sub_upper.begin(), ::toupper);
        std::ostringstream ss_sc;
        ss_sc << std::fixed << std::setprecision(2) << score;
        std::string lbl_text = sub_upper + " " + ss_sc.str();

        int baseline = 0;
        cv::Size sz = cv::getTextSize(lbl_text, cv::FONT_HERSHEY_SIMPLEX, 0.50, 1, &baseline);
        int pill_w = sz.width + 18;
        int pill_h = sz.height + 10;
        int px1 = std::max(4, std::min(disp_w - pill_w - 4, x1));
        int py1 = (y1 >= pill_h + 4) ? (y1 - pill_h) : std::min(disp_h - pill_h - 4, y1 + 4);

        cv::rectangle(frame, cv::Rect(px1, py1, pill_w, pill_h), cv::Scalar(14, 16, 19), -1);
        cv::rectangle(frame, cv::Rect(px1, py1, pill_w, pill_h), box_color, 1, cv::LINE_AA);
        cv::rectangle(frame, cv::Rect(px1, py1, 4, pill_h), box_color, -1);
        cv::putText(frame, lbl_text, cv::Point(px1 + 8, py1 + sz.height + 3), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(240, 243, 246), 1, cv::LINE_AA);
    }
}

VisionInferenceResult HailoVisionEngine::infer_mode1(
    const cv::Mat& frame_bgr,
    float conf_thresh,
    float iou_thresh
) {
    VisionInferenceResult res;
    if (frame_bgr.empty() || !impl_->yolo_ort.ready) {
        res.has_defects = false;
        res.primary_defect = "NONE";
        res.display_title = "NO DEFECT DETECTED";
        res.severity = "NONE";
        res.hardware_device = hardware_name_;
        if (!frame_bgr.empty()) {
            res.annotated_frame = frame_bgr.clone();
            cv::imencode(".jpg", res.annotated_frame, res.encoded_jpeg);
        }
        return res;
    }

    auto t0 = std::chrono::steady_clock::now();
    int w_orig = frame_bgr.cols;
    int h_orig = frame_bgr.rows;

    // 1. Preprocessing matching Python:
    // GaussianBlur (1.2) -> addWeighted unsharp mask -> resize 416x416 -> RGB -> [0, 1] NCHW
    cv::Mat blur;
    cv::GaussianBlur(frame_bgr, blur, cv::Size(0, 0), 1.2);
    cv::Mat sharp;
    cv::addWeighted(frame_bgr, 1.35, blur, -0.35, 0, sharp);

    std::vector<float> input_tensor_values;
    bgr_to_nchw_float(sharp, 416, 416, input_tensor_values);

    std::vector<int64_t> in_shape = {1, 3, 416, 416};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor_values.data(), input_tensor_values.size(),
        in_shape.data(), in_shape.size()
    );

    auto outputs = impl_->yolo_ort.session->Run(
        Ort::RunOptions{nullptr},
        impl_->yolo_ort.input_cstrs.data(), &in_tensor, 1,
        impl_->yolo_ort.output_cstrs.data(), impl_->yolo_ort.output_cstrs.size()
    );

    auto t1 = std::chrono::steady_clock::now();
    res.cv_latency_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    res.hardware_device = hardware_name_;
    res.is_mode2 = false;

    // 2. Decode YOLO output [1, 10, num_preds]
    auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int num_preds = (out_shape.size() >= 3) ? static_cast<int>(out_shape[2]) : 3549;
    float sx = static_cast<float>(w_orig) / 416.0f;
    float sy = static_cast<float>(h_orig) / 416.0f;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    const float* data = outputs[0].GetTensorData<float>();
    for (int p = 0; p < num_preds; ++p) {
        float max_score = -1.0f;
        int best_cls = -1;
        for (int c = 0; c < 6; ++c) {
            float score = data[(4 + c) * num_preds + p];
            if (score > max_score) {
                max_score = score;
                best_cls = c;
            }
        }
        if (max_score >= conf_thresh) {
            float xc = data[0 * num_preds + p];
            float yc = data[1 * num_preds + p];
            float bw = data[2 * num_preds + p];
            float bh = data[3 * num_preds + p];

            int x = static_cast<int>((xc - bw * 0.5f) * sx);
            int y = static_cast<int>((yc - bh * 0.5f) * sy);
            int w = static_cast<int>(bw * sx);
            int h = static_cast<int>(bh * sy);

            boxes.push_back(cv::Rect(x, y, w, h));
            confidences.push_back(max_score);
            class_ids.push_back(best_cls);
        }
    }

    // 3. NMS Boxes
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, iou_thresh, nms_indices);

    // 4. Metallic Gate Validation
    std::vector<DetectionBox> accepted_boxes;
    for (int idx : nms_indices) {
        const auto& r = boxes[idx];
        int x1 = std::max(0, r.x);
        int y1 = std::max(0, r.y);
        int x2 = std::min(w_orig, r.x + r.width);
        int y2 = std::min(h_orig, r.y + r.height);

        auto gate = MetallicSurfaceGate::evaluate_roi_bgr(
            frame_bgr.data, w_orig, h_orig, frame_bgr.step,
            x1, y1, x2, y2
        );

        if (gate.is_metallic) {
            DetectionBox b;
            b.x1 = x1;
            b.y1 = y1;
            b.x2 = x2;
            b.y2 = y2;
            b.confidence = confidences[idx];
            b.class_id = class_ids[idx];
            b.class_name = STEEL_CLASSES[class_ids[idx]];
            accepted_boxes.push_back(b);
        }
    }

    res.detections = accepted_boxes;
    res.total_boxes = static_cast<int>(accepted_boxes.size());
    res.has_defects = !accepted_boxes.empty();

    if (res.has_defects) {
        res.union_area_pct = compute_union_area(accepted_boxes, w_orig, h_orig);

        // Multi-defect dominance calculation
        std::map<std::string, float> class_max_conf;
        std::map<std::string, int> class_box_cnt;
        for (const auto& b : accepted_boxes) {
            if (b.confidence > class_max_conf[b.class_name]) {
                class_max_conf[b.class_name] = b.confidence;
            }
            class_box_cnt[b.class_name]++;
        }

        std::string dominant_class = accepted_boxes[0].class_name;
        float max_dominance = -1.0f;
        for (const auto& pair : class_max_conf) {
            float dom = pair.second * (1.0f + res.union_area_pct / 50.0f);
            if (dom > max_dominance) {
                max_dominance = dom;
                dominant_class = pair.first;
            }
        }

        res.primary_defect = dominant_class;
        std::string title = dominant_class;
        std::replace(title.begin(), title.end(), '_', ' ');
        std::transform(title.begin(), title.end(), title.begin(), ::toupper);
        res.display_title = title;
        res.highest_confidence = class_max_conf[dominant_class];

        res.severity = qc::rules::calculate_unified_severity(
            dominant_class, res.union_area_pct, res.highest_confidence, res.total_boxes, false
        );
    } else {
        res.primary_defect = "NONE";
        res.display_title = "NO DEFECT DETECTED";
        res.highest_confidence = 0.0f;
        res.union_area_pct = 0.0f;
        res.severity = "NONE";
    }

    res.annotated_frame = frame_bgr.clone();
    draw_mode1_overlay(res.annotated_frame, res.detections, res.display_title, res.severity);
    cv::imencode(".jpg", res.annotated_frame, res.encoded_jpeg);

    return res;
}

VisionInferenceResult HailoVisionEngine::infer_mode2(
    const cv::Mat& frame_bgr,
    float t_img,
    float t_pix
) {
    VisionInferenceResult res;
    if (frame_bgr.empty() || !impl_->mode2_ort.ready) {
        return infer_mode1(frame_bgr);
    }

    auto t0 = std::chrono::steady_clock::now();
    int w_orig = frame_bgr.cols;
    int h_orig = frame_bgr.rows;

    cv::Mat rgb;
    std::vector<float> input_tensor_values;
    bgr_to_nchw_float(frame_bgr, 256, 256, input_tensor_values, &rgb);

    std::vector<int64_t> in_shape = {1, 3, 256, 256};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor_values.data(), input_tensor_values.size(),
        in_shape.data(), in_shape.size()
    );

    auto outputs = impl_->mode2_ort.session->Run(
        Ort::RunOptions{nullptr},
        impl_->mode2_ort.input_cstrs.data(), &in_tensor, 1,
        impl_->mode2_ort.output_cstrs.data(), impl_->mode2_ort.output_cstrs.size()
    );
    auto t1 = std::chrono::steady_clock::now();
    res.cv_latency_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    res.hardware_device = "ARM NEON CPU (ONNX Runtime)";
    res.is_mode2 = true;

    float* amap_ptr = outputs[0].GetTensorMutableData<float>();
    float score = *outputs[1].GetTensorMutableData<float>();
    bool is_anom = (score >= t_img);

    cv::Mat amap(256, 256, CV_32FC1, amap_ptr);
    cv::Mat bin_mask(256, 256, CV_8UC1);
    for (int i = 0; i < 256 * 256; ++i) {
        bin_mask.data[i] = (amap_ptr[i] >= t_pix) ? 255 : 0;
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(bin_mask, bin_mask, cv::MORPH_OPEN, kernel);

    int nonzeros = cv::countNonZero(bin_mask);
    float area_pct = (static_cast<float>(nonzeros) / (256.0f * 256.0f)) * 100.0f;

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(bin_mask, labels, stats, centroids);

    std::vector<DetectionBox> boxes;
    float sx = static_cast<float>(w_orig) / 256.0f;
    float sy = static_cast<float>(h_orig) / 256.0f;
    int min_pixels = static_cast<int>(0.002f * 256.0f * 256.0f);

    for (int i = 1; i < num_labels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= min_pixels) {
            int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            int y = stats.at<int>(i, cv::CC_STAT_TOP);
            int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

            DetectionBox b;
            b.x1 = static_cast<int>(x * sx);
            b.y1 = static_cast<int>(y * sy);
            b.x2 = static_cast<int>((x + w) * sx);
            b.y2 = static_cast<int>((y + h) * sy);
            b.confidence = score;
            b.class_name = "anomaly";
            boxes.push_back(b);
        }
    }

    res.regions = static_cast<int>(boxes.size());
    res.score_ratio = score / std::max(1e-4f, t_img);
    res.highest_confidence = score;
    res.union_area_pct = area_pct;
    res.detections = boxes;
    res.total_boxes = static_cast<int>(boxes.size());

    if (!is_anom && !boxes.empty() && area_pct >= 0.30f && score >= 0.28f) {
        is_anom = true;
    }

    std::string sub_type = "clean";
    std::string title = "CLEAN NUT PASS";
    if (!is_anom || boxes.empty()) {
        res.has_defects = false;
        res.severity = "NONE";
        res.primary_defect = "CLEAN NUT PASS";
        res.display_title = "CLEAN NUT PASS";
    } else {
        res.has_defects = true;
        if (area_pct >= 20.0f) {
            sub_type = "flip";
            title = "COMPONENT ANOMALY: FLIP";
        } else {
            // Profile primary bounding box in 256x256 space
            int bx1 = static_cast<int>(boxes[0].x1 / sx);
            int by1 = static_cast<int>(boxes[0].y1 / sy);
            int bx2 = static_cast<int>(boxes[0].x2 / sx);
            int by2 = static_cast<int>(boxes[0].y2 / sy);
            float cx = (bx1 + bx2) * 0.5f;
            float cy = (by1 + by2) * 0.5f;
            float dist_c = std::sqrt((cx - 128.0f) * (cx - 128.0f) + (cy - 128.0f) * (cy - 128.0f));
            bool is_outer_flange = (bx1 <= 28 || by1 <= 28 || bx2 >= 228 || by2 >= 228) && (dist_c >= 90.0f);

            cv::Mat hsv_nut;
            cv::cvtColor(rgb, hsv_nut, cv::COLOR_RGB2HSV);
            cv::Scalar mean_s = cv::mean(hsv_nut, bin_mask);

            if (is_outer_flange) {
                sub_type = "bent";
                title = "COMPONENT ANOMALY: BENT";
            } else if (mean_s[1] >= 35.0) {
                sub_type = "color";
                title = "COMPONENT ANOMALY: COLOR";
            } else {
                sub_type = "scratch";
                title = "COMPONENT ANOMALY: SCRATCH";
            }
        }

        res.sub_type = sub_type;
        res.primary_defect = title;
        res.display_title = title;
        res.severity = qc::rules::calculate_unified_severity(
            sub_type, area_pct, score, res.regions, true
        );
    }

    res.annotated_frame = frame_bgr.clone();
    draw_mode2_overlay(res.annotated_frame, amap, res.detections, res.display_title, res.severity, score, area_pct);
    cv::imencode(".jpg", res.annotated_frame, res.encoded_jpeg);

    return res;
}

VisionInferenceResult HailoVisionEngine::infer(
    const cv::Mat& frame_bgr,
    const std::string& pipeline_mode,
    float conf_thresh,
    float iou_thresh
) {
    if (pipeline_mode == "MODE_2") {
        return infer_mode2(frame_bgr);
    } else if (pipeline_mode == "MODE_1") {
        return infer_mode1(frame_bgr, conf_thresh, iou_thresh);
    } else {
        // AUTO Mode: Use Stage0 to route frame
        auto s0 = classify_stage0(frame_bgr);
        if (s0.route == "MODE_2") {
            return infer_mode2(frame_bgr);
        } else {
            return infer_mode1(frame_bgr, conf_thresh, iou_thresh);
        }
    }
}

} // namespace vision
} // namespace qc
