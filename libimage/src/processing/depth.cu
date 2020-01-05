#include "core/cuda_kernel.h"
#include "depth_cu.h"

#include <curand_kernel.h>

using namespace core;

#define PI 3.14159265359f

#define BLOCKSIZE 32

__global__ void g_Depth2Normals(DevPtr<float3> out, const DevPtr<float> in,
    const float fx, const float fy, const float cx, const float cy)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = in.width;
    const int height = in.height;

    if (pos.x >= width || pos.y >= height)
        return;

    const float d = in(pos.x, pos.y);

    // Gradient with forward differences
    const float2 grad = d_fwdGradient(in, pos.x, pos.y);
    out(pos.x, pos.y)
        = d_normals(d, grad.x, grad.y, pos.x, pos.y, fx, fy, cx, cy);
}

__global__ void g_Depth2Harmonics(DevPtr<float4> out, const DevPtr<float> in,
    const float fx, const float fy, const float cx, const float cy)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = in.width;
    const int height = in.height;

    if (pos.x >= width || pos.y >= height)
        return;

    const float d = in(pos.x, pos.y);

    // Gradient with forward differences
    const float2 grad = d_fwdGradient(in, pos.x, pos.y);
    const float3 norm
        = d_normals(d, grad.x, grad.y, pos.x, pos.y, fx, fy, cx, cy);
    out(pos.x, pos.y) = make_float4(norm.x, norm.y, norm.z, 1.f);
}

__global__ void g_depth2Theta(DevPtr<float3> theta, const DevPtr<float> depth)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = depth.width;
    const int height = depth.height;

    if (pos.x >= width || pos.y >= height)
        return;

    const float d = depth(pos.x, pos.y);
    const float2 grad = d_fwdGradient(depth, pos.x, pos.y);

    theta(pos.x, pos.y) = make_float3(d, grad.x, grad.y);
}

__global__ void g_Normals2Shading(DevPtr<float3> shading,
    const DevPtr<float3> normals, const DevPtr<float3> light)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = normals.width;
    const int height = normals.height;

    if (pos.x >= width || pos.y >= height)
        return;

    const float3 n = normals(pos.x, pos.y);

    // Light is 4x1
    assert(light.height == 4);

    const float3 l1 = light(0, 0);
    const float3 l2 = light(0, 1);
    const float3 l3 = light(0, 2);
    const float3 l4 = light(0, 3);

    // n.x * l1 + n.y * l2 + n.z * l3 + l4
    shading(pos.x, pos.y) = n.x * l1 + n.y * l2 + n.z * l3 + l4;
}

__global__ void g_BwdGradient(
    DevPtr<float> dx, DevPtr<float> dy, const DevPtr<float> depth)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = depth.width;
    const int height = depth.height;

    if (pos.x >= width || pos.y >= height)
        return;

    const float2 grad = d_bwdGradient(depth, pos.x, pos.y);
    dx(pos.x, pos.y) = grad.x;
    dy(pos.x, pos.y) = grad.y;
}

__global__ void g_FwdGradient(
    DevPtr<float> dx, DevPtr<float> dy, const DevPtr<float> depth)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = depth.width;
    const int height = depth.height;

    if (pos.x >= width || pos.y >= height)
        return;

    const float2 grad = d_fwdGradient(depth, pos.x, pos.y);
    dx(pos.x, pos.y) = grad.x;
    dy(pos.x, pos.y) = grad.y;
}

__global__ void g_SymGradient(
    DevPtr<float> dx, DevPtr<float> dy, const DevPtr<float> depth)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = depth.width;
    const int height = depth.height;

    if (pos.x >= width || pos.y >= height)
        return;

    const float2 grad = d_symGradient(depth, pos.x, pos.y);
    dx(pos.x, pos.y) = grad.x;
    dy(pos.x, pos.y) = grad.y;
}

__global__ void g_NormalDistribution(DevPtr<float> output,
    const DevPtr<float> depth, const float mean, const float var)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = depth.width;
    const int height = depth.height;

    if (pos.x >= width || pos.y >= height)
        return;

    const float d = depth(pos.x, pos.y);

    const float m = mean + d;
    const float v = var * d;

    curandState_t state;
    curand_init(pos.x * pos.y, 0, 0, &state);

    // if d == 0: v = 0
    if (d > 0)
        output(pos.x, pos.y)
            = cuimage::max(0.f, (curand_normal(&state) * v) + m);
    else
        output(pos.x, pos.y) = 0;
}


void cu_DepthToNormals(DevPtr<float3> normals, const DevPtr<float> depth,
    const float fx, const float fy, const float cx, const float cy)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(depth.width, depth.height, block);

    g_Depth2Normals<<<grid, block>>>(normals, depth, fx, fy, cx, cy);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}

void cu_DepthToHarmonics(DevPtr<float4> harmonics, const DevPtr<float> depth,
    const float fx, const float fy, const float cx, const float cy)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(depth.width, depth.height, block);

    g_Depth2Harmonics<<<grid, block>>>(harmonics, depth, fx, fy, cx, cy);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}

void cu_DepthToTheta(DevPtr<float3> theta, const DevPtr<float> depth)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(depth.width, depth.height, block);

    g_depth2Theta<<<grid, block>>>(theta, depth);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}

void cu_NormalsToShading(DevPtr<float3> shading, const DevPtr<float3> normals,
    const DevPtr<float3> light)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(normals.width, normals.height, block);

    g_Normals2Shading<<<grid, block>>>(shading, normals, light);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}

void cu_Gradient(DevPtr<float> dx, DevPtr<float> dy, const DevPtr<float> depth,
    const std::string type)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(depth.width, depth.height, block);

    if (type == "forward")
        g_FwdGradient<<<grid, block>>>(dx, dy, depth);
    else if (type == "backward")
        g_BwdGradient<<<grid, block>>>(dx, dy, depth);
    else if (type == "symmetric")
        g_SymGradient<<<grid, block>>>(dx, dy, depth);
    else
        throw std::runtime_error("Invalid type for gradient: " + type);

    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}

void cu_NormalDistribution(DevPtr<float> output, const DevPtr<float> depth,
    const float mean, const float var)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(depth.width, depth.height, block);

    g_NormalDistribution<<<grid, block>>>(output, depth, mean, var);

    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}
