#pragma once

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdint>

namespace qc {
namespace rules {

struct RawBox {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float conf = 0.0f;
    std::string class_name;
};

struct ClassSummary {
    std::string class_name;
    float conf_pct = 0.0f;
    float max_conf = 0.0f;
    float area_pct = 0.0f;
    int box_count = 0;
    std::string pattern; // "isolated" or "clustered"
    float dominance = 0.0f;
    std::vector<RawBox> boxes;
};

struct MultiDefectAnalysis {
    bool has_defects = false;
    std::vector<ClassSummary> classes;
    std::string primary_class = "NONE";
    std::string signature = "NONE";
    std::string display_title = "NO DEFECT DETECTED";
    float highest_conf = 0.0f;
    float total_area_pct = 0.0f;
    int total_boxes = 0;
    std::string overall_severity = "NONE";
};

/**
 * Computes exact union area percentage of bounding boxes over frame (w, h).
 */
float compute_union_area_pct(const std::vector<RawBox>& boxes, int w, int h);

/**
 * Unified Class-Weighted Severity Engine (Phase 1).
 * Maps class risk factor, calibrated area tertiles, and box counts to 'Low', 'Medium', or 'High'.
 */
std::string calculate_unified_severity(
    const std::string& class_name,
    float area_pct,
    float conf = 0.85f,
    int n_boxes = 1,
    bool is_mode2 = false
);

/**
 * Analyzes multi-defect detections: groups by class, computes individual & union areas,
 * dominance ranking, pattern evaluation, and overall severity.
 */
MultiDefectAnalysis analyze_detections(
    const std::vector<RawBox>& detections,
    int w_orig,
    int h_orig
);

/**
 * Builds the compact fine-tuned prompt for on-device Llama 3.2 1B INT4 inference.
 */
std::string build_llama_advisory_prompt(
    const std::string& defect_type,
    const std::string& severity,
    float confidence,
    float area_pct,
    int box_count,
    bool is_mode2 = false
);

} // namespace rules
} // namespace qc
