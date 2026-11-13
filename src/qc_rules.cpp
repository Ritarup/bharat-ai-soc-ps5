#include "qc_rules.hpp"

#include <cmath>
#include <sstream>
#include <iomanip>
#include <numeric>

namespace qc {
namespace rules {

static const std::map<std::string, int> CLASS_RISK_MAP = {
    {"crazing", 3},
    {"inclusion", 3},
    {"bent", 3},
    {"pitted_surface", 2},
    {"rolled_in_scale", 2},
    {"scratch", 2},
    {"scratches", 2},
    {"patches", 1},
    {"color", 1},
    {"flip", 0}
};

static const std::map<std::string, std::pair<float, float>> CLASS_AREA_THRESHOLDS = {
    {"crazing", {58.0f, 72.0f}},
    {"inclusion", {14.0f, 20.0f}},
    {"patches", {27.0f, 38.0f}},
    {"pitted_surface", {80.0f, 95.0f}},
    {"rolled_in_scale", {45.0f, 56.0f}},
    {"scratches", {14.0f, 23.0f}},
    {"bent", {2.0f, 3.5f}},
    {"color", {0.8f, 1.8f}},
    {"flip", {38.0f, 42.0f}},
    {"scratch", {1.5f, 2.8f}}
};

static std::string normalize_class_name(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    for (char& c : name) {
        if (c == ' ' || c == '-') c = '_';
    }
    // Strip prefixes if present
    const std::string prefix = "component_anomaly:";
    if (name.rfind(prefix, 0) == 0) {
        name = name.substr(prefix.length());
    }
    return name;
}

float compute_union_area_pct(const std::vector<RawBox>& boxes, int w, int h) {
    if (boxes.empty() || w <= 0 || h <= 0) {
        return 0.0f;
    }

    // Grid downsampling factor for high-speed sub-millisecond bitmask (scale by 4)
    int sw = (w + 3) / 4;
    int sh = (h + 3) / 4;
    std::vector<uint8_t> mask(sw * sh, 0);

    for (const auto& b : boxes) {
        int x1 = std::max(0, std::min(static_cast<int>(b.x1 / 4.0f), sw));
        int y1 = std::max(0, std::min(static_cast<int>(b.y1 / 4.0f), sh));
        int x2 = std::max(0, std::min(static_cast<int>(b.x2 / 4.0f + 0.5f), sw));
        int y2 = std::max(0, std::min(static_cast<int>(b.y2 / 4.0f + 0.5f), sh));

        for (int y = y1; y < y2; ++y) {
            std::fill_n(&mask[y * sw + x1], x2 - x1, 1);
        }
    }

    int active_pixels = 0;
    for (uint8_t val : mask) {
        if (val) active_pixels++;
    }

    return 100.0f * static_cast<float>(active_pixels) / static_cast<float>(sw * sh);
}

std::string calculate_unified_severity(
    const std::string& class_name,
    float area_pct,
    float conf,
    int n_boxes,
    bool is_mode2
) {
    (void)conf;
    (void)is_mode2;

    std::string c_norm = normalize_class_name(class_name);
    auto it_risk = CLASS_RISK_MAP.find(c_norm);
    if (it_risk == CLASS_RISK_MAP.end()) {
        if (area_pct >= 25.0f || n_boxes >= 4) return "High";
        if (area_pct >= 5.0f || n_boxes >= 2) return "Medium";
        return "Low";
    }

    int risk = it_risk->second;
    float t1 = 15.0f;
    float t2 = 35.0f;

    auto it_area = CLASS_AREA_THRESHOLDS.find(c_norm);
    if (it_area != CLASS_AREA_THRESHOLDS.end()) {
        t1 = it_area->second.first;
        t2 = it_area->second.second;
    }

    // Special feeder handling for mode 2 flip
    if (c_norm == "flip") {
        if (n_boxes >= 4 || area_pct >= t2) return "High";
        if (n_boxes >= 2 || area_pct >= t1) return "Medium";
        return "Low";
    }

    // Base severity from calibrated area tertiles
    std::string base_sev = "Low";
    if (area_pct < t1) {
        base_sev = "Low";
    } else if (area_pct < t2) {
        base_sev = "Medium";
    } else {
        base_sev = "High";
    }

    // Multi-defect escalation according to risk factor
    if (risk == 3) { // Critical
        if (n_boxes >= 4 && base_sev == "Medium") return "High";
        if (n_boxes >= 3 && base_sev == "Low") return "Medium";
    } else if (risk == 2) { // Moderate
        if (n_boxes >= 5 && base_sev == "Medium") return "High";
    }

    return base_sev;
}

MultiDefectAnalysis analyze_detections(
    const std::vector<RawBox>& detections,
    int w_orig,
    int h_orig
) {
    MultiDefectAnalysis res;
    if (detections.empty() || w_orig <= 0 || h_orig <= 0) {
        res.has_defects = false;
        res.signature = "NONE";
        res.display_title = "NO DEFECT DETECTED";
        res.overall_severity = "NONE";
        return res;
    }

    res.has_defects = true;
    res.total_boxes = static_cast<int>(detections.size());
    res.total_area_pct = compute_union_area_pct(detections, w_orig, h_orig);

    // Group by class
    std::map<std::string, std::vector<RawBox>> by_class;
    for (const auto& d : detections) {
        by_class[d.class_name].push_back(d);
    }

    for (const auto& pair : by_class) {
        ClassSummary summary;
        summary.class_name = pair.first;
        summary.boxes = pair.second;
        summary.box_count = static_cast<int>(pair.second.size());
        summary.area_pct = compute_union_area_pct(pair.second, w_orig, h_orig);

        float max_c = 0.0f;
        for (const auto& b : pair.second) {
            if (b.conf > max_c) max_c = b.conf;
        }
        summary.max_conf = max_c;
        summary.conf_pct = max_c * 100.0f;
        summary.pattern = (summary.box_count >= 2 || summary.area_pct >= 25.0f) ? "clustered" : "isolated";
        summary.dominance = summary.conf_pct * (1.0f + summary.area_pct / 50.0f);

        res.classes.push_back(summary);
    }

    // Sort descending by dominance score
    std::sort(res.classes.begin(), res.classes.end(), [](const ClassSummary& a, const ClassSummary& b) {
        return a.dominance > b.dominance;
    });

    if (!res.classes.empty()) {
        res.primary_class = res.classes[0].class_name;
        res.highest_conf = res.classes[0].max_conf;

        std::stringstream ss_sig, ss_title;
        for (size_t i = 0; i < res.classes.size(); ++i) {
            if (i > 0) {
                ss_sig << " + ";
                ss_title << " + ";
            }
            ss_sig << res.classes[i].class_name;

            std::string t = res.classes[i].class_name;
            for (char& c : t) {
                if (c == '_') c = ' ';
                else c = std::toupper(c);
            }
            ss_title << t;
        }
        res.signature = ss_sig.str();
        res.display_title = ss_title.str();

        res.overall_severity = calculate_unified_severity(
            res.primary_class, res.total_area_pct, res.highest_conf, res.total_boxes, false
        );
    }

    return res;
}

std::string build_llama_advisory_prompt(
    const std::string& defect_type,
    const std::string& severity,
    float confidence,
    float area_pct,
    int box_count,
    bool is_mode2
) {
    (void)box_count;
    std::string sys_prompt;
    std::string user_msg;

    std::string sev_short = "Med";
    std::string s_lower = severity;
    std::transform(s_lower.begin(), s_lower.end(), s_lower.begin(), ::tolower);
    if (s_lower.find("high") != std::string::npos) sev_short = "High";
    else if (s_lower.find("low") != std::string::npos) sev_short = "Low";

    if (is_mode2) {
        sys_prompt = "Fastener QC AI. Reply one short sentence: Possible Cause: <cause> | Recommended Action: <action>";
        std::stringstream ss;
        ss << "Defect: " << defect_type << " (score " << std::fixed << std::setprecision(2) << confidence
           << ") | Area: " << std::fixed << std::setprecision(1) << area_pct << "% | Sev: " << sev_short;
        user_msg = ss.str();
    } else {
        sys_prompt = "Steel QC AI. Reply one short sentence: Possible Cause: <cause> | Recommended Action: <action>";
        std::stringstream ss;
        ss << "Defect: " << defect_type << " (" << static_cast<int>(confidence * 100) << "%) | Sev: " << sev_short;
        user_msg = ss.str();
    }

    std::stringstream prompt;
    prompt << "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
           << sys_prompt << "<|eot_id|>"
           << "<|start_header_id|>user<|end_header_id|>\n\n"
           << user_msg << "<|eot_id|>"
           << "<|start_header_id|>assistant<|end_header_id|>\n\n";

    return prompt.str();
}

} // namespace rules
} // namespace qc
