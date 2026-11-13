#include "metallic_gate_neon.hpp"
#include <algorithm>
#include <cmath>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace qc {
namespace vision {

GateResult MetallicSurfaceGate::evaluate_roi_bgr(
    const uint8_t* bgr_data,
    int img_w, int img_h, int stride,
    int x1, int y1, int x2, int y2,
    float max_mean_s,
    float max_colorful_ratio
) {
    // Clamp ROI bounding box coordinates
    x1 = std::max(0, std::min(img_w - 1, x1));
    y1 = std::max(0, std::min(img_h - 1, y1));
    x2 = std::max(0, std::min(img_w - 1, x2));
    y2 = std::max(0, std::min(img_h - 1, y2));

    int roi_w = x2 - x1;
    int roi_h = y2 - y1;

    if (roi_w <= 2 || roi_h <= 2 || !bgr_data) {
        return {true, 0.0f, 0.0f};
    }

    uint64_t total_s = 0;
    uint32_t count_high_s = 0;
    int total_pixels = roi_w * roi_h;

    for (int y = y1; y < y2; ++y) {
        const uint8_t* row_ptr = bgr_data + (y * stride) + (x1 * 3);
        int x = 0;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        int neon_chunk = (roi_w / 16) * 16;
        if (neon_chunk > 0) {
            compute_saturation_neon(row_ptr, neon_chunk, total_s, count_high_s);
            x += neon_chunk;
            row_ptr += (neon_chunk * 3);
        }
#endif
        int remainder = roi_w - x;
        if (remainder > 0) {
            compute_saturation_scalar(row_ptr, remainder, total_s, count_high_s);
        }
    }

    float mean_s = static_cast<float>(total_s) / static_cast<float>(total_pixels);
    float colorful_ratio = static_cast<float>(count_high_s) / static_cast<float>(total_pixels);

    bool is_metallic = (mean_s <= max_mean_s) && (colorful_ratio <= max_colorful_ratio);
    return {is_metallic, mean_s, colorful_ratio};
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
void MetallicSurfaceGate::compute_saturation_neon(
    const uint8_t* bgr_ptr, int count,
    uint64_t& sum_s, uint32_t& count_high_s
) {
    uint8x16_t thresh_vec = vdupq_n_u8(SATURATION_THRESHOLD);
    uint32_t local_high = 0;
    uint64_t local_sum = 0;

    for (int i = 0; i < count; i += 16) {
        // vld3q_u8 deinterleaves 16 BGR triplets into 3 distinct vector registers
        uint8x16x3_t bgr = vld3q_u8(bgr_ptr + (i * 3));
        uint8x16_t v_b = bgr.val[0];
        uint8x16_t v_g = bgr.val[1];
        uint8x16_t v_r = bgr.val[2];

        // Max and Min per pixel
        uint8x16_t max_val = vmaxq_u8(v_r, vmaxq_u8(v_g, v_b));
        uint8x16_t min_val = vminq_u8(v_r, vminq_u8(v_g, v_b));
        uint8x16_t delta   = vsubq_u8(max_val, min_val);

        // Approximate S = (delta * 255) / max_val
        // In industrial monochrome steel, delta > 20 indicates saturation
        uint8x16_t cmp_mask = vcgtq_u8(delta, thresh_vec);

        // Count colorful pixels
        uint8_t cmp_arr[16];
        uint8_t delta_arr[16];
        vst1q_u8(cmp_arr, cmp_mask);
        vst1q_u8(delta_arr, delta);

        for (int j = 0; j < 16; ++j) {
            local_sum += delta_arr[j];
            if (cmp_arr[j]) {
                local_high++;
            }
        }
    }

    sum_s += local_sum;
    count_high_s += local_high;
}
#endif

void MetallicSurfaceGate::compute_saturation_scalar(
    const uint8_t* bgr_ptr, int count,
    uint64_t& sum_s, uint32_t& count_high_s
) {
    for (int i = 0; i < count; ++i) {
        uint8_t b = bgr_ptr[i * 3 + 0];
        uint8_t g = bgr_ptr[i * 3 + 1];
        uint8_t r = bgr_ptr[i * 3 + 2];

        uint8_t max_val = std::max({r, g, b});
        uint8_t min_val = std::min({r, g, b});
        uint8_t delta = max_val - min_val;

        uint8_t s = 0;
        if (max_val > 0) {
            s = static_cast<uint8_t>((static_cast<uint32_t>(delta) * 255) / max_val);
        }

        sum_s += s;
        if (s > SATURATION_THRESHOLD) {
            count_high_s++;
        }
    }
}

} // namespace vision
} // namespace qc
