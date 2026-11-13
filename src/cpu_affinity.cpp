#include "cpu_affinity.hpp"

#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <sched.h>

#ifdef __linux__
#include <sys/syscall.h>
#endif

namespace qc {
namespace hardware {

int get_cpu_core_count() {
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? static_cast<int>(count) : 4;
}

bool pin_thread_to_cores(pthread_t thread, const std::vector<int>& core_ids) {
#ifdef __linux__
    int ncores = std::max(1, get_cpu_core_count());
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int core : core_ids) {
        if (core >= 0) {
            int mapped = core % ncores;
            if (mapped < CPU_SETSIZE) {
                CPU_SET(mapped, &cpuset);
            }
        }
    }
    int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "[CPU Affinity] Warning: pthread_setaffinity_np failed with error code: " << rc << std::endl;
        return false;
    }
    return true;
#else
    (void)thread;
    (void)core_ids;
    return false;
#endif
}

bool pin_current_thread(int core_id) {
    return pin_thread_to_cores(pthread_self(), {core_id});
}

bool pin_current_thread_mask(const std::vector<int>& core_ids) {
    return pin_thread_to_cores(pthread_self(), core_ids);
}

void set_thread_name(const std::string& name) {
#ifdef __linux__
    // Linux pthread_setname_np limits name to 16 chars including null byte
    std::string truncated = name.substr(0, 15);
    pthread_setname_np(pthread_self(), truncated.c_str());
#else
    (void)name;
#endif
}

bool set_thread_priority(pthread_t thread, int policy, int priority) {
#ifdef __linux__
    struct sched_param param;
    param.sched_priority = priority;
    int rc = pthread_setschedparam(thread, policy, &param);
    return (rc == 0);
#else
    (void)thread;
    (void)policy;
    (void)priority;
    return false;
#endif
}

std::string get_current_affinity_string() {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0) {
        std::stringstream ss;
        ss << "Cores [";
        bool first = true;
        for (int i = 0; i < get_cpu_core_count(); ++i) {
            if (CPU_ISSET(i, &cpuset)) {
                if (!first) ss << ", ";
                ss << i;
                first = false;
            }
        }
        ss << "]";
        return ss.str();
    }
#endif
    return "Cores [All]";
}

} // namespace hardware
} // namespace qc
