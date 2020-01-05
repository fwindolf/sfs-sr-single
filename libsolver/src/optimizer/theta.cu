#include "solver/optimizer/theta_cu.h"

#include "core/cuda_kernel.h"

using namespace core;

__global__ void g_ThetaToHarmonics(DevPtr<float4> harmonics, 
                                   const DevPtr<float3> theta, 
                                   const float fx, const float fy,
                                   const float cx, const float cy)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = theta.width;
    const int height = theta.height;

    if (pos.x >= width || pos.y >= height)
        return;

    // Calculate normals from theta
    float3 t = theta(pos.x, pos.y);
    float3 n = d_normals(t.x, t.y, t.z, pos.x, pos.y, fx, fy, cx, cy);

    harmonics(pos.x, pos.y) = make_float4(n.x, n.y, n.z, 1.f);
}


void cu_ThetaToHarmonics(DevPtr<float4> harmonics,
                        const DevPtr<float3> theta,
                        const float fx, const float fy,
                        const float cx, const float cy)
{
    assert(harmonics.width == theta.width);
    assert(harmonics.height == theta.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(theta.width, theta.height, block);

    g_ThetaToHarmonics <<< grid, block >>> (harmonics, theta, fx, fy, cx, cy);
    
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}