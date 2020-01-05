#include "solver/optimizer/lighting_cu.h"

#include "core/cuda_kernel.h"

#include "cublas_v2.h"

using namespace core;

template <typename T>
__global__ void g_CalculateA(DevPtr<float> output,
                             const DevPtr<T> albedo,
                             const DevPtr<float4> harmonics,
                             const DevPtr<uchar> mask);


template <>
__global__ void g_CalculateA(DevPtr<float> output,
                             const DevPtr<float> albedo,
                             const DevPtr<float4> harmonics,
                             const DevPtr<uchar> mask)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = albedo.width;
    const int height = albedo.height;

    if (pos.x >= width || pos.y >= height)
        return;

    // Albedo is WxHx1
    float  a = albedo(pos.x, pos.y);
    // SHarm is WxHx4
    float4 l = harmonics(pos.x, pos.y);

    // Output is WHCx4x1 in col major format
    assert(output.width == 4);
    assert(output.height == width * height);
    
    const int idx = pos.x * height + pos.y;

    // Masked pixel are 0, else a .* l
    if (mask(pos.x, pos.y))
    {
        output(0, idx) = l.x * a;
        output(1, idx) = l.y * a;
        output(2, idx) = l.z * a;
        output(3, idx) = l.w * a;
    }
    else
    {
        output(0, idx) = 0.f;
        output(1, idx) = 0.f;
        output(2, idx) = 0.f;
        output(3, idx) = 0.f;
    }
}

template <>
__global__ void g_CalculateA(DevPtr<float> output,
                             const DevPtr<float3> albedo,
                             const DevPtr<float4> harmonics,
                             const DevPtr<uchar> mask)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = albedo.width;
    const int height = albedo.height;

    if (pos.x >= width || pos.y >= height)
        return;

    // Albedo is WHx3
    float3 a = albedo(pos.x, pos.y);
    // SHarm is WHx4
    float4 l = harmonics(pos.x, pos.y);

    // Output is WHCx4x1 in col major format
    assert(output.width == 4);
    assert(output.height == 3 * width * height);

    const int idx = pos.x * height + pos.y; // x, y inverted to make col major
    const int off = width * height;

    if (mask(pos.x, pos.y))
    {
        output(0, 0 * off + idx) = l.x * a.x;
        output(1, 0 * off + idx) = l.y * a.x;
        output(2, 0 * off + idx) = l.z * a.x;
        output(3, 0 * off + idx) = l.w * a.x;

        output(0, 1 * off + idx) = l.x * a.y;
        output(1, 1 * off + idx) = l.y * a.y;
        output(2, 1 * off + idx) = l.z * a.y;
        output(3, 1 * off + idx) = l.w * a.y;

        output(0, 2 * off + idx) = l.x * a.z;
        output(1, 2 * off + idx) = l.y * a.z;
        output(2, 2 * off + idx) = l.z * a.z;
        output(3, 2 * off + idx) = l.w * a.z;
    }
    else
    {
        output(0, 0 * off + idx) = 0.f;
        output(1, 0 * off + idx) = 0.f;
        output(2, 0 * off + idx) = 0.f;
        output(3, 0 * off + idx) = 0.f;

        output(0, 1 * off + idx) = 0.f;
        output(1, 1 * off + idx) = 0.f;
        output(2, 1 * off + idx) = 0.f;
        output(3, 1 * off + idx) = 0.f;

        output(0, 2 * off + idx) = 0.f;
        output(1, 2 * off + idx) = 0.f;
        output(2, 2 * off + idx) = 0.f;
        output(3, 2 * off + idx) = 0.f;
    }
}

template <typename T>
__global__ void g_CalculateB(DevPtr<float> output,
                             const DevPtr<T> image,
                             const DevPtr<uchar> mask);

template <>
__global__ void g_CalculateB(DevPtr<float> output,
                             const DevPtr<float> image,
                             const DevPtr<uchar> mask)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = image.width;
    const int height = image.height;

    if (pos.x >= width || pos.y >= height)
        return;

    // Output is WHCx1x1 in col major format
    assert(output.width == 1);
    assert(output.height == width * height);

    const int idx = pos.x * height + pos.y; // x, y inverted to make col major
    assert(idx < output.height);

    if (mask(pos.x, pos.y))
        output(0, idx) = image(pos.x, pos.y);
    else
        output(0, idx) = 0.f;
}

template <>
__global__ void g_CalculateB(DevPtr<float> output, 
                             const DevPtr<float3> image,
                             const DevPtr<uchar> mask)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = image.width;
    const int height = image.height;

    if (pos.x >= width || pos.y >= height)
        return;

    // Output is WHCx1x1 in col major format
    assert(output.width == 1);
    assert(output.height == 3 * width * height);

    const float3 img = image(pos.x, pos.y);

    const int idx = pos.x * height + pos.y; // x, y inverted to make col major
    const int off = width * height;

    if (mask(pos.x, pos.y))
    {
        output(0, 0 * off + idx) = img.x;
        output(0, 1 * off + idx) = img.y;
        output(0, 2 * off + idx) = img.z;
    }
    else
    {
        output(0, 0 * off + idx) = 0.f;
        output(0, 1 * off + idx) = 0.f;
        output(0, 2 * off + idx) = 0.f;
    }
}


template <typename T>
__device__ T d_applyLight(float4 a, T l1, T l2, T l3, T l4);

template <>
__device__ float3 d_applyLight(float4 a, float3 l0, float3 l1, float3 l2, float3 l3)
{
    float3 out;
    out.x = (a.x * l0.x + a.y * l1.x + a.z * l2.x + a.w * l3.x);
    out.y = (a.x * l0.y + a.y * l1.y + a.z * l2.y + a.w * l3.y);
    out.z = (a.x * l0.z + a.y * l1.z + a.z * l2.z + a.w * l3.z);
    //return clamp(out, 0.f, 1.f);
    return out;
}

template <>
__device__ float d_applyLight(float4 a, float l0, float l1, float l2, float l3)
{
    float out = a.x * l0 + a.y * l1 + a.z * l2 + a.w * l3;
    //return clamp(out, 0.f, 1.f);
    return out;    
}

template <typename T>
__global__ void g_CalculateShading(DevPtr<T> output, 
                                   const DevPtr<float4> harmonics,
                                   const DevPtr<T> light,
                                   const DevPtr<uchar> mask)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = harmonics.width;
    const int height = harmonics.height;

    if (pos.x >= width || pos.y >= height)
        return;

    assert(light.width == 1);

    const T l1 = light(0, 0);
    const T l2 = light(0, 1);
    const T l3 = light(0, 2);
    const T l4 = light(0, 3);

    // shading is harmonics (WxHx4) * Light (4x1xC)
    if (mask(pos.x, pos.y))
        output(pos.x, pos.y) = d_applyLight(harmonics(pos.x, pos.y), l1, l2, l3, l4);
    else
        output(pos.x, pos.y) = make<T>(0.f);
        
}

template <typename T>
__global__ void g_InitLight(DevPtr<T> light)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = light.width;
    const int height = light.height;

    // light always is 4x1xC
    assert(width == 1);

    if (pos.x >= width || pos.y >= height)
        return;
    
    // frontal lighting [0, 0, -1, 0]
    if (pos.y == 2)
        light(pos.x, pos.y) = make<T>(-1);
    else
        light(pos.x, pos.y) = make<T>(0);
}

template <typename T>
__global__ void g_SetLight(DevPtr<T> light, 
                           const DevPtr<float> in)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = light.width;
    const int height = light.height;

    // light always is 4x1xC
    assert(width == 1);

    if (pos.x >= width || pos.y >= height)
        return;

    light(pos.x, pos.y) = make<T>(in(pos.x, pos.y));
}

template <typename T>
void cu_CalculateAMatrix(DevPtr<float> output,
                         const DevPtr<T> albedo,
                         const DevPtr<float4> harmonics,
                         const DevPtr<uchar> mask)
{
    assert(albedo.width == harmonics.width);
    assert(albedo.height == harmonics.height);
    assert(albedo.width == mask.width);
    assert(albedo.height == mask.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(albedo.width, albedo.height, block);

    g_CalculateA <<< grid, block >>>(output, albedo, harmonics, mask);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}

template <typename T>
void cu_CalculateBMatrix(DevPtr<float> output, 
                         const DevPtr<T> image,
                         const DevPtr<uchar> mask)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(image.width, image.height, block);

    g_CalculateB <<< grid, block >>>(output, image, mask);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}

void cu_CalculateATA(DevPtr<float> output,
                     const DevPtr<float> a, 
                     cublasHandle_t cublasHandle)
{
    float* dA = a.data;
    float* dATA = output.data;

    // As Cublas expects column major, we calculate matA*matAT, so "flip" dimensions
    int m = a.width; // number of rows of matrix op(A) and C.
    int n = a.width; // number of columns of matrix op(B) and C.
    int k = a.height; // number of columns of op(A) and rows of op(B).

    // Leading dimension
    int lda = std::max(1, m); // lda>=max(1,m) if transa == CUBLAS_OP_N.
    int ldb = std::max(1, n); // ldb>=max(1,n) if transb == CUBLAS_OP_T.
    int ldc = std::max(1, m); // ldc>=max(1,m).

    auto alpha = new float(1.f);
    auto beta = new float(0.f);

    // Calculates into matATA_
    cublasSafeCall(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                               m, n, k, alpha, dA, lda, dA, ldb, beta, dATA, ldc));
    
    // cudaSafeCall(cudaDeviceSynchronize());
}

void cu_CalculateATb(DevPtr<float> output,
                     const DevPtr<float> a,
                     const DevPtr<float> b,
                     cublasHandle_t cublasHandle)
{
    float* dA = a.data;
    float* db = b.data;
    float* dATb = output.data;

    // As Cublas expects column major, we calculate matA*matbT, so "flip" dimensions
    int m = a.width; // number of rows of matrix op(A) and C.
    int n = b.width; // number of columns of matrix op(B) and C.
    int k = a.height; // number of columns of op(A) and rows of op(B).

    // Leading dimension
    int lda = std::max(1, m); // lda>=max(1,m) if transa == CUBLAS_OP_N.
    int ldb = std::max(1, n); // ldb>=max(1,n) if transb == CUBLAS_OP_T.
    int ldc = std::max(1, m); // ldc>=max(1,m).

    auto alpha = new float(1.f);
    auto beta = new float(0.f);

    // Calculates into matATA_
    cublasSafeCall(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                               m, n, k, alpha, dA, lda, db, ldb, beta, dATb, ldc));
    
    // cudaSafeCall(cudaDeviceSynchronize());
}

template <typename T>
void cu_CalculateShading(DevPtr<T> output, 
                         const DevPtr<float4> harmonics,
                         const DevPtr<T> light,
                         const DevPtr<uchar> mask)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(harmonics.width, harmonics.height, block);

    g_CalculateShading <<< grid, block >>>(output, harmonics, light, mask);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}

template <typename T>
void cu_InitializeLighting(DevPtr<T> light)
{
    dim3 block = block2D(4);
    dim3 grid = grid2D(light.width, light.height, block);

    g_InitLight <<< grid, block >>> (light);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}

template <typename T>
void cu_SetLighting(DevPtr<T> light,
                    const DevPtr<float> input)
{
    dim3 block = block2D(4);
    dim3 grid = grid2D(light.width, light.height, block);

    g_SetLight <<< grid, block >>> (light, input);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}

/**
 * Explicit template instantiations
 */

template
void cu_CalculateAMatrix(DevPtr<float> output, 
                         const DevPtr<float3> albedo,
                         const DevPtr<float4> light,
                         const DevPtr<uchar> mask);

template
void cu_CalculateBMatrix(DevPtr<float> output, 
                         const DevPtr<float3> image,
                         const DevPtr<uchar> mask);

template 
void cu_CalculateShading(DevPtr<float3> output,
                         const DevPtr<float4> harmonics,
                         const DevPtr<float3> light,
                         const DevPtr<uchar> mask);
template 
void cu_InitializeLighting(DevPtr<float3> light);

template
void cu_SetLighting(DevPtr<float3> light,
                    const DevPtr<float> input);