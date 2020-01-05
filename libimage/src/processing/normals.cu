#include "normals_cu.h"
#include "core/cuda/utils.h"


__global__ void g_Normals2Harmonics(DevPtr<float4> harmonics, 
                                    const DevPtr<float3> normals)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = normals.width;
    const int height = normals.height;

    if (pos.x >= width || pos.y >= height)
        return;

    float3 n = normals(pos.x, pos.y);
    harmonics(pos.x, pos.y) = make_float4(n.x, n.y, n.z, 1.f);
}

void cu_NormalsToHarmonics(DevPtr<float4> harmonics, 
                           const DevPtr<float3> normals)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(normals.width, normals.height, block);

    g_Normals2Harmonics <<< grid, block >>>(harmonics, normals);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}