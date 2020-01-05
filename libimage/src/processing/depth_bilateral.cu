#include "core/cuda_kernel.h"
#include "depth_cu.h"

using namespace core;

#define PI 3.14159265359f
#define BLOCKSIZE 32

__constant__ float bilateral_Kernel_S[2 * BLOCKSIZE + 1]; // max supported size
__constant__ float bilateral_Kernel_C[2 * BLOCKSIZE + 1]; // max supported size

inline __device__ __host__ float d_gaussian(const int x, const float sigma)
{
    return expf(-1.f * powf(x, 2.f) / (2 * powf(sigma, 2.f)))
        / (2 * PI * powf(sigma, 2.f));
}

__global__ void g_BilteralSmooth(DevPtr<float> output,
    const DevPtr<float> depth, const DevPtr<uchar> mask, const int radius)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    // image area + halos on every connected block area
    __shared__ float sh_data[9 * BLOCKSIZE * BLOCKSIZE];
    DevPtr<float> smem(sh_data, 3 * BLOCKSIZE, 3 * BLOCKSIZE);

    // Populate smem with data from depth
    fill(smem, depth, mask, threadIdx, pos, BLOCKSIZE);

    // Exit early just now, as we need to fill the shared memory no matter what
    if (pos.x >= depth.width || pos.y >= depth.height)
        return;

    if (!mask(pos.x, pos.y))
    {
        output(pos.x, pos.y) = nanf("");
        return;
    }

    float sum = 0.f;
    float out = 0.f;

    // Center pixel (smem has halo of BLOCKSIZE)
    const int sx = threadIdx.x + BLOCKSIZE;
    const int sy = threadIdx.y + BLOCKSIZE;

    // Central depth
    const float d0 = smem(sx, sy);
    
    for (int rx = -radius; rx <= radius; rx++)
    {
        for (int ry = -radius; ry <= radius; ry++)
        {
            // From image region use offset of radius in x, y direction
            const float d = smem(sx + rx, sy + ry);
            
            // Either depth undefined or outside image area
            if (isnan(d) || d <= 0.f)
                continue;
            
            // Spatial distance: More influence of closer pixels
            const float ks_x = bilateral_Kernel_S[BLOCKSIZE + rx];
            const float ks_y = bilateral_Kernel_S[BLOCKSIZE + ry];
            const float k_s = ks_x * ks_y;

            // Color distance: More influence of pixels with similar value
            const int dist = min(2 * BLOCKSIZE, static_cast<int>(abs(d - d0)));
            const float k_c = bilateral_Kernel_C[dist];

            sum += k_c * k_s;
            out += k_c * k_s * d;
        }
    }

    // Normalize
    if (sum > 0)
        out /= sum;
    else
        out = nanf("");
    
    output(pos.x, pos.y) = out;
}

void cu_BilateralSmooth(DevPtr<float> depth, const DevPtr<uchar> mask,
    const int radius, const float sigma_space, const float sigma_color)
{
    assert(radius < BLOCKSIZE);

    dim3 block = block2D(BLOCKSIZE);
    dim3 grid = grid2D(depth.width, depth.height, block);

    // Create gaussian kernels
    float h_kernel_s[2 * BLOCKSIZE + 1] = {0.f};
    float h_kernel_c[2 * BLOCKSIZE + 1] = {0.f};
    const float k0_space = d_gaussian(0, sigma_space);
    const float k0_color = d_gaussian(0, sigma_color);

    for (int i = -BLOCKSIZE; i <= BLOCKSIZE; i++)
    {
        // Middle: BLOCKSIZE
        h_kernel_s[BLOCKSIZE + i] = d_gaussian(i, sigma_space) / k0_space;

        // From Left (only positive)
        h_kernel_c[BLOCKSIZE + i] = d_gaussian(BLOCKSIZE + i, sigma_color) / k0_color;
    }

    cudaSafeCall(cudaMemcpyToSymbol(
        bilateral_Kernel_S, h_kernel_s, (2 * BLOCKSIZE + 1) * sizeof(float)));
    cudaSafeCall(cudaMemcpyToSymbol(
        bilateral_Kernel_C, h_kernel_c, (2 * BLOCKSIZE + 1) * sizeof(float)));

    // Temporary image for intermediate result
    DevPtr<float> tmp(depth.width, depth.height);

    // Run smoothing for rows
    g_BilteralSmooth<<<grid, block>>>(tmp, depth, mask, radius);
    
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());

    depth = tmp;
    tmp.free();
}
