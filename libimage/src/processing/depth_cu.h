#include "core/cuda.h"

using cuimage::DevPtr;

void cu_DepthToNormals(DevPtr<float3> normals,
                       const DevPtr<float> depth,
                       const float fx, const float fy,
                       const float cx, const float cy);

void cu_DepthToHarmonics(DevPtr<float4> harmonics,
                         const DevPtr<float> depth,
                         const float fx, const float fy,
                         const float cx, const float cy);

void cu_DepthToTheta(DevPtr<float3> theta,
                    const DevPtr<float> depth);

void cu_NormalsToShading(DevPtr<float3> shading,
                         const DevPtr<float3> normals,
                         const DevPtr<float3> light);

void cu_Gradient(DevPtr<float> dx,
                 DevPtr<float> dy,
                 const DevPtr<float> depth,
                 const std::string type);

void cu_NormalDistribution(DevPtr<float> output,
                           const DevPtr<float> depth,
                           const float mean,
                           const float var);

void cu_GaussianSmooth(DevPtr<float> depth,
                       const DevPtr<uchar> mask,
                       const int radius,
                       const float sigma);

void cu_BilateralSmooth(DevPtr<float> depth,
                        const DevPtr<uchar> mask,
                        const int radius,
                        const float sigma_space,
                        const float sigma_color);