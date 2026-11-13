#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <chrono>

namespace qc {
namespace memory {

struct FramePacket {
    int width = 0;
    int height = 0;
    int channels = 3;
    uint64_t frame_id = 0;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<uint8_t> data; // Pre-allocated BGR/RGB buffer
};

/**
 * Thread-safe, pre-allocated Circular Ring Buffer for High-Speed Vision.
 * Decouples Camera Ingestion (Core 0) from Vision/NMS Processing (Core 1)
 * without triggering heap allocations per frame at 100+ FPS.
 */
class FrameRingBuffer {
public:
    explicit FrameRingBuffer(size_t capacity = 4, int default_w = 640, int default_h = 480, int channels = 3)
        : capacity_(capacity), head_(0), tail_(0), size_(0), is_shutdown_(false) {
        slots_.resize(capacity_);
        for (size_t i = 0; i < capacity_; ++i) {
            slots_[i].width = default_w;
            slots_[i].height = default_h;
            slots_[i].channels = channels;
            slots_[i].data.resize(default_w * default_h * channels);
        }
    }

    bool push(const uint8_t* raw_bytes, int w, int h, int c, uint64_t frame_id) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (is_shutdown_) return false;

        FramePacket& slot = slots_[head_];
        slot.width = w;
        slot.height = h;
        slot.channels = c;
        slot.frame_id = frame_id;
        slot.timestamp = std::chrono::steady_clock::now();

        size_t total_bytes = w * h * c;
        if (slot.data.size() < total_bytes) {
            slot.data.resize(total_bytes);
        }
        std::memcpy(slot.data.data(), raw_bytes, total_bytes);

        head_ = (head_ + 1) % capacity_;
        if (size_ < capacity_) {
            size_++;
        } else {
            // Overwriting oldest unread frame to maintain ultra-low latency real-time stream
            tail_ = (tail_ + 1) % capacity_;
        }

        not_empty_cv_.notify_one();
        return true;
    }

    bool pop(FramePacket& out_packet, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_empty_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
            return size_ > 0 || is_shutdown_;
        })) {
            return false;
        }

        if (is_shutdown_ && size_ == 0) return false;

        out_packet = slots_[tail_];
        tail_ = (tail_ + 1) % capacity_;
        size_--;
        return true;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_shutdown_ = true;
        not_empty_cv_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

private:
    const size_t capacity_;
    std::vector<FramePacket> slots_;
    size_t head_;
    size_t tail_;
    size_t size_;
    bool is_shutdown_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_cv_;
};

} // namespace memory
} // namespace qc
