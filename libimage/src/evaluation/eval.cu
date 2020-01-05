#include "eval_cu.h"

#include "core/cuda_kernel.h"

#include <cublas_v2.h>

using namespace core;

#define PI 3.141592654f

__global__ void g_PixelwiseError(DevPtr<float> error, 
                                 const DevPtr<float3> normals,
                                 const DevPtr<float3> normals_opt)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = normals.width;
    const int height = normals.height;

    if (pos.x >= width || pos.y >= height)
        return;

    float3 n_star = normals(pos.x, pos.y);
    float3 n_opt = normals_opt(pos.x, pos.y);

    if (n_star == n_opt)
    {
        error(pos.x, pos.y) = 0;
        return;
    }

    error(pos.x, pos.y) = atan2f(sqrtf( sum( cross(n_star, n_opt) * cross(n_star, n_opt) ) ), dot(n_star, n_opt) ) * 180 / PI;
}


void cu_PixelwiseError(DevPtr<float> error,
                       const DevPtr<float3> normals,
                       const DevPtr<float3> normals_opt)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(normals.width, normals.height, block);

    assert(normals.width == normals_opt.width);
    assert(normals.height == normals_opt.height);

    g_PixelwiseError <<< grid, block >>> (error, normals, normals_opt);    

    cudaCheckLastCall();
    //cudaSafeCall(cudaDeviceSynchronize());
}