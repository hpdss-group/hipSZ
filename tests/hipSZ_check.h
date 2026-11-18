#ifndef HIPSZ_EXAMPLES_CHECK_H
#define HIPSZ_EXAMPLES_CHECK_H

#include <hip/hip_runtime.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

// Every HIP call in the test programs is wrapped in this macro: a failed
// API call must abort the test with a non-zero exit code, not silently
// produce garbage that the error check may or may not catch.
#define HIP_CHECK(expr)                                                     \
    do {                                                                    \
        hipError_t _hip_status = (expr);                                    \
        if (_hip_status != hipSuccess) {                                    \
            fprintf(stderr, "HIP error %s (%d) at %s:%d: %s\n",             \
                    hipGetErrorString(_hip_status), (int)_hip_status,       \
                    __FILE__, __LINE__, #expr);                             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// Wall-clock timer for the full-API view (scratch allocation, memset,
// compressed-size readback and the implicit syncs inside the entry points),
// complementing the HIP-event kernel-only view.
class WallTimer {
    std::chrono::steady_clock::time_point t0;
public:
    void start() { t0 = std::chrono::steady_clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    }
};

#endif // HIPSZ_EXAMPLES_CHECK_H
