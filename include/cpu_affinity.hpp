#pragma once

#include <string>
#include <vector>
#include <pthread.h>

namespace qc {
namespace hardware {

// Raspberry Pi 5 Quad-Core Cortex-A76 Hardware Core Topology
constexpr int CORE_IO_IPC       = 0; // Core 0: V4L2 Camera, PCIe DMA, IPC / Web Server
constexpr int CORE_VISION       = 1; // Core 1: Pre-processing, ARM NEON Metallic Gate, NMS
constexpr int CORE_LLAMA_LEADER = 2; // Core 2: Resident Llama 3.2 1B Primary Worker
constexpr int CORE_LLAMA_WORKER = 3; // Core 3: Resident Llama 3.2 1B OpenMP SIMD Worker

/**
 * Pins a given POSIX thread to specific physical CPU cores.
 * Optimizes cache locality (L1/L2 data/instruction cache) and eliminates thread migration latency.
 */
bool pin_thread_to_cores(pthread_t thread, const std::vector<int>& core_ids);

/**
 * Pins the currently executing thread to a specific single CPU core.
 */
bool pin_current_thread(int core_id);

/**
 * Pins the currently executing thread to a list of CPU cores (e.g. Cores 2 & 3 for LLM).
 */
bool pin_current_thread_mask(const std::vector<int>& core_ids);

/**
 * Assigns a human-readable thread name visible in top / htop / perf.
 */
void set_thread_name(const std::string& name);

/**
 * Sets thread scheduling policy (e.g., SCHED_FIFO for real-time vision, SCHED_OTHER for background).
 */
bool set_thread_priority(pthread_t thread, int policy, int priority);

/**
 * Returns number of online physical CPU cores on host.
 */
int get_cpu_core_count();

/**
 * Formats a debug string showing the CPU core mask for the current thread.
 */
std::string get_current_affinity_string();

} // namespace hardware
} // namespace qc
