#include "core/cuda_kernel.h"
#include "depth_cu.h"

using namespace core;

#define PI 3.14159265359f

#define BLOCKSIZE 32
__constant__ float smooth_Kernel[2 * BLOCKSIZE + 1]; // max supported size

inline __device__ __host__ float d_gaussian(const int x, const float sigma)
{
    return expf(-1.f * powf(x, 2.f) / (2 * powf(sigma, 2.f)))
        / (2 * PI * powf(sigma, 2.f));
}


__global__ void g_GaussianSmoothX(DevPtr<float> output,
    const DevPtr<float> depth, const DevPtr<uchar> mask, const int radius)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = depth.width;
    const int height = depth.height;

    // halos + image area for each row
    __shared__ float sh_data[3 * BLOCKSIZE * BLOCKSIZE];
    DevPtr<float> smem(sh_data, 3 * BLOCKSIZE, BLOCKSIZE);

    // Populate the shared memory region from depth
    fill_x(smem, depth, mask, threadIdx, pos, BLOCKSIZE);

    // Exit early just now, as we need to fill the shared memory no matter what
    if (pos.x >= width || pos.y >= height)
        return;

    if (!mask(pos.x, pos.y))
    {
        output(pos.x, pos.y) = nanf("");
        return;
    }        

    float sum = 0.f;
    float out = 0.f;

    // Do actual convolution
    for (int r = -radius; r <= radius; r++)
    {
        // The blocks' middle area begins at BLOCKSIZE
        int sx = threadIdx.x + BLOCKSIZE;
        int sy = threadIdx.y;

        // Offset by the thread index and then take into account radius
        const float d = smem(sx + r, sy);

        if (isnan(d) || d <= 0.f)
            continue;

        // The middle of the kernel is at BLOCKSIZE
        const float k = smooth_Kernel[BLOCKSIZE + r];

        sum += k;
        out += k * d;
    }

    // Normalize
    if (sum > 0)
        out /= sum;
    else
        out = nanf("");

    output(pos.x, pos.y) = out;
}

__global__ void g_GaussianSmoothY(DevPtr<float> output,
    const DevPtr<float> depth, const DevPtr<uchar> mask, const int radius)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = depth.width;
    const int height = depth.height;

    // halos + image area for each row
    __shared__ float sh_data[3 * BLOCKSIZE * BLOCKSIZE];
    DevPtr<float> smem(sh_data, BLOCKSIZE, 3 * BLOCKSIZE);

    // Populate the shared memory region from depth
    fill_y(smem, depth, mask, threadIdx, pos, BLOCKSIZE);
    
    // Exit early just now, as we need to fill the shared memory no matter what
    if (pos.x >= width || pos.y >= height)
        return;

    if (!mask(pos.x, pos.y))
    {
        output(pos.x, pos.y) = nanf("");
        return;
    }      

    float sum = 0.f;
    float out = 0.f;

    // Do actual convolution
    for (int r = -radius; r <= radius; r++)
    {
        // The blocks' middle area begins at BLOCKSIZE
        int sx = threadIdx.x;
        int sy = threadIdx.y + BLOCKSIZE;

        // Offset by the thread index and then take into account radius
        const float d = smem(sx, sy + r);

        if (isnan(d) || d <= 0.f)
            continue;

        // The middle of the kernel is at BLOCKSIZE
        const float k = smooth_Kernel[BLOCKSIZE + r];

        sum += k;
        out += k * d;
    }

    // Normalize
    if (sum > 0)
        out /= sum;
    else
        out = nanf("");

    output(pos.x, pos.y) = out;
}

void cu_GaussianSmooth(DevPtr<float> depth, const DevPtr<uchar> mask,
    const int radius, const float sigma)
{
    assert(radius < BLOCKSIZE);

    dim3 block = block2D(BLOCKSIZE);
    dim3 grid = grid2D(depth.width, depth.height, block);

    // Generate convolution kernel, centered at BLOCKSIZE
    float h_kernel[2 * BLOCKSIZE + 1] = {0.f};
    const float k0 = d_gaussian(0, sigma); // for normalization to 1
    for (int i = -radius; i <= radius; i++)
    {
        h_kernel[BLOCKSIZE + i] = d_gaussian(i, sigma) / k0;
    }

    cudaSafeCall(cudaMemcpyToSymbol(
        smooth_Kernel, h_kernel, (2 * BLOCKSIZE + 1) * sizeof(float)));

    // Temporary image for intermediate result
    DevPtr<float> tmp(depth.width, depth.height);

    // Run smoothing for rows
    g_GaussianSmoothX<<<grid, block>>>(tmp, depth, mask, radius);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());

    // Run smoothing for columns
    g_GaussianSmoothY<<<grid, block>>>(depth, tmp, mask, radius);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());

    tmp.free();
}