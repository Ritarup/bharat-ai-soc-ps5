#pragma once

#include <cstdint>
#include <vector>

namespace qc {
namespace vision {

struct GateResult {
    bool is_metallic;
    float mean_saturation;
    float colorful_ratio;
};

/**
 * High-performance Metallic Surface Gate optimized with ARM NEON SIMD.
 * Evaluates bounding box ROI in steel defect detection:
 *  - Rejects human hands, technician skin, wooden benches, and colorful debris.
 *  - Uses 128-bit vector instructions (16 pixels/cycle) for sub-millisecond execution.
 */
class MetallicSurfaceGate {
public:
    static constexpr float DEFAULT_MAX_MEAN_S        = 42.0f;
    static constexpr float DEFAULT_MAX_COLORFUL_RATIO = 0.20f;
    static constexpr uint8_t SATURATION_THRESHOLD    = 65;

    /**
     * Evaluates a BGR image ROI.
     * Returns true if the region matches metallic steel surface characteristics.
     */
    static GateResult evaluate_roi_bgr(
        const uint8_t* bgr_data,
        int img_w, int img_h, int stride,
        int x1, int y1, int x2, int y2,
        float max_mean_s = DEFAULT_MAX_MEAN_S,
        float max_colorful_ratio = DEFAULT_MAX_COLORFUL_RATIO
    );

private:
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    static void compute_saturation_neon(
        const uint8_t* bgr_ptr, int count,
        uint64_t& sum_s, uint32_t& count_high_s
    );
#endif
    static void compute_saturation_scalar(
        const uint8_t* bgr_ptr, int count,
        uint64_t& sum_s, uint32_t& count_high_s
    );
};

} // namespace vision
} // namespace qc
