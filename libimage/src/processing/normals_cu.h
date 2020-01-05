#include "core/cuda.h"

void cu_NormalsToHarmonics(DevPtr<float4> output, 
                           const DevPtr<float3> input);