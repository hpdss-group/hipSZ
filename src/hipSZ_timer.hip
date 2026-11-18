#include "hipSZ_timer.h"

/** ************************************************************************
 * @brief           HIP event timer for measuring GPU kernel execution.
 *
 *                  Events are recorded on the caller-provided stream so the
 *                  measurement brackets exactly the work submitted there;
 *                  they are destroyed once the elapsed time is read, so a
 *                  repeated Start/Get cycle does not leak event objects.
 * *********************************************************************** */
TimingGPU::TimingGPU() { privateTimingGPU = new PrivateTimingGPU;  }
TimingGPU::~TimingGPU() { delete privateTimingGPU; }

/** ************************************************************************
 * @brief           Start timer.
 * *********************************************************************** */
void TimingGPU::StartCounter(hipStream_t stream)
{
    hipEventCreate(&((*privateTimingGPU).start));
    hipEventCreate(&((*privateTimingGPU).stop));
    hipEventRecord((*privateTimingGPU).start, stream);
}

/** ************************************************************************
 * @brief           Start timer with flags.
 * *********************************************************************** */
void TimingGPU::StartCounterFlags(hipStream_t stream)
{
    int eventflags = hipEventBlockingSync;

    hipEventCreateWithFlags(&((*privateTimingGPU).start), eventflags);
    hipEventCreateWithFlags(&((*privateTimingGPU).stop), eventflags);
    hipEventRecord((*privateTimingGPU).start, stream);
}

/** ************************************************************************
 * @brief           End timer, get count in ms.
 * *********************************************************************** */
float TimingGPU::GetCounter(hipStream_t stream)
{
    float time;
    hipEventRecord((*privateTimingGPU).stop, stream);
    hipEventSynchronize((*privateTimingGPU).stop);
    hipEventElapsedTime(&time, (*privateTimingGPU).start, (*privateTimingGPU).stop);
    hipEventDestroy((*privateTimingGPU).start);
    hipEventDestroy((*privateTimingGPU).stop);
    return time;
}
