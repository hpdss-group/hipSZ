#ifndef HIPSZ_INCLUDE_HIPSZ_HIPSZ_TIMER_H
#define HIPSZ_INCLUDE_HIPSZ_HIPSZ_TIMER_H

#include <hip/hip_runtime.h>

struct PrivateTimingGPU {
    hipEvent_t start;
    hipEvent_t stop;
};

class TimingGPU
{
    private:
        PrivateTimingGPU *privateTimingGPU;

    public:
        TimingGPU();
        ~TimingGPU();
        // Records events on the stream the measured work is submitted to;
        // passing 0 (default stream) reproduces the historical behavior.
        void StartCounter(hipStream_t stream = 0);
        void StartCounterFlags(hipStream_t stream = 0);
        float GetCounter(hipStream_t stream = 0);

};

#endif // HIPSZ_INCLUDE_HIPSZ_HIPSZ_TIMER_H
