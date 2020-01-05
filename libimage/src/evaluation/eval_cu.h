#include "core/cuda.h"

#include <cuimage/cuda/devptr.h>

using cuimage::DevPtr;

void cu_PixelwiseError(DevPtr<float> error, 
                       const DevPtr<float3> normals,
                       const DevPtr<float3> normals_opt);