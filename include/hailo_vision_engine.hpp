#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <opencv2/opencv.hpp>

namespace qc {
namespace vision {

struct DetectionBox {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    float confidence = 0.0f;
    int class_id = 0;
    std::string class_name;
};

struct VisionInferenceResult {
    bool has_defects = false;
    std::vector<DetectionBox> detections;
    std::string primary_defect;
    std::string display_title;
    std::string severity; // Low, Medium, High, NONE
    float highest_confidence = 0.0f;
    float score_ratio = 0.0f;
    float union_area_pct = 0.0f;
    int total_boxes = 0;
    int regions = 0;
    std::string spread = "localized";
    float cv_latency_ms = 0.0f;
    std::string hardware_device; // e.g. "Hailo-8L (13 TOPS AI HAT)" or "ARM NEON CPU (OpenCV DNN)"
    bool is_mode2 = false;
    std::string sub_type;
    cv::Mat annotated_frame;
    std::vector<uint8_t> encoded_jpeg;
};

struct Stage0Result {
    std::string route;       // "MODE_1", "MODE_2", or "unclear"
    std::string class_name;  // "steel_sheet", "part", "unclear"
    float confidence = 0.0f;
    float latency_ms = 0.0f;
};

class HailoVisionEngine {
public:
    HailoVisionEngine();
    ~HailoVisionEngine();

    /**
     * Initializes the vision engine with models.
     */
    bool init(
        const std::string& hef_path,
        const std::string& yolo_onnx_path,
        const std::string& stage0_onnx_path = "",
        const std::string& mode2_onnx_path = ""
    );

    /**
     * Evaluates Stage 0 tiny classifier (224x224) to automatically route frames.
     */
    Stage0Result classify_stage0(const cv::Mat& frame_bgr);

    /**
     * Runs Mode 1 Steel Surface Defect Inspection (YOLO11n 416x416).
     */
    VisionInferenceResult infer_mode1(
        const cv::Mat& frame_bgr,
        float conf_thresh = 0.35f,
        float iou_thresh = 0.55f
    );

    /**
     * Runs Mode 2 Fastener / Nut Anomaly Inspection (EfficientAD-S 256x256).
     */
    VisionInferenceResult infer_mode2(
        const cv::Mat& frame_bgr,
        float t_img = 0.35f,
        float t_pix = 0.1676f
    );

    /**
     * Unified infer call based on active pipeline mode.
     */
    VisionInferenceResult infer(
        const cv::Mat& frame_bgr,
        const std::string& pipeline_mode = "AUTO",
        float conf_thresh = 0.35f,
        float iou_thresh = 0.55f
    );

    bool is_hailo_active() const { return is_hailo_ready_; }
    std::string get_hardware_name() const { return hardware_name_; }
    static const std::vector<std::string>& get_steel_classes();

private:
    bool is_hailo_ready_ = false;
    std::string hardware_name_ = "ARM NEON CPU";
    std::string hef_path_;
    std::string yolo_onnx_path_;
    std::string stage0_onnx_path_;
    std::string mode2_onnx_path_;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    float compute_union_area(const std::vector<DetectionBox>& boxes, int w, int h);
    void draw_mode1_overlay(cv::Mat& frame, const std::vector<DetectionBox>& boxes, const std::string& title, const std::string& sev);
    void draw_mode2_overlay(cv::Mat& frame, const cv::Mat& amap, const std::vector<DetectionBox>& boxes, const std::string& title, const std::string& sev, float score, float area_pct);
};

} // namespace vision
} // namespace qc
