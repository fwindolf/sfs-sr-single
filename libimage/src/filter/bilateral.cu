#include "image/filter/filter_cu.h"

#include "core/cuda_kernel.h"

using namespace image;
using namespace cuimage;

template <typename T>
__device__ T d_normal(const T v, const float sigma);

template <>
inline __device__ float d_normal(const float v, const float sigma)
{
    return exp(- (v * v) / (2 * sigma * sigma));
}

template <>
inline __device__ float1 d_normal(const float1 v, const float sigma)
{
    return make_float1(exp(- (v.x * v.x) / (2 * sigma * sigma)));
}

template <>
inline __device__ float2 d_normal(const float2 v, const float sigma)
{
    float d = 1.f / (2 * sigma * sigma);
    return make_float2(exp(-(v.x * v.x) * d), exp(-(v.y * v.y) * d));
}

template <>
inline __device__ float3 d_normal(const float3 v, const float sigma)
{
    float d = 1.f / (2 * sigma * sigma);
    return make_float3(exp(-(v.x * v.x) * d), exp(-(v.y * v.y) * d), exp(-(v.z * v.z) * d));
}

template <>
inline __device__ float4 d_normal(const float4 v, const float sigma)
{
    float d = 1.f / (2 * sigma * sigma);
    return make_float4(exp(-(v.x * v.x) * d), exp(-(v.y * v.y) * d), exp(-(v.z * v.z) * d), exp(-(v.w * v.w) * d));
}


/**
 * Normal only implemented for float, so restrict template to that
 */
template <typename T, typename std::enable_if<is_float_type<T>::value, T>::type* = nullptr>
__global__ void g_BilateralFilter(DevPtr<T> output, const DevPtr<T> input, const int r, const float sigma_s, const float sigma_r)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if(pos.x >= output.width || pos.y >= output.height)
        return;

    T sum = make<T>(0.f);
    T out = make<T>(0.f);

    const T val = input(pos.x, pos.y);

    for (int ry = -r; ry <= r; ry++)
    {
        const int py = pos.y + ry;
        if (py < 0 || py >= input.height)
            continue;

        for (int rx = -r; rx <= r; rx++)
        {
            const int px = pos.x + rx;
            if (px < 0 || px >= input.width)
                continue;

            // Check if pixel of neighbor is invalid   
            const T val_n = input(px, py);
            if (!isvalid(val_n) && val_n <= make<T>(0.f))
                continue;    

            // Do magic for pixel (x, y) -> neighbors (px, py)
            T d_space = d_normal(ry + r, sigma_s) * d_normal(rx + r, sigma_s);
            T d_range = d_normal(val - val_n, sigma_r); // val - val_n is euclidean norm in 1D

            T factor = d_space * d_range;
            sum += factor;
            out += factor * val; 
        }
    }

    output(pos.x, pos.y) = out / sum;
}

template <typename T>
void cu_BilateralFilter(DevPtr<T>& output, const cuimage::DevPtr<T>& input, const int radius, const float sigmaS, const float sigmaR)
{
    assert(output.width == input.width);
    assert(output.height == input.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);

    g_BilateralFilter <<< grid, block >>> (output, input, radius, sigmaS, sigmaR);

    cudaCheckLastCall();
    cudaDeviceSynchronize();
}