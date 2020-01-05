#include "core/cuda/utils.h"

using cuimage::DevPtr;

void cu_ThetaToHarmonics(DevPtr<float4> harmonics,
                         const DevPtr<float3> theta,
                         const float fx, const float fy,
                         const float cx, const float cy);