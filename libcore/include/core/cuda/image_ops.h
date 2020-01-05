#ifndef CORE_CUDA_IMAGE_OPS_H
#define CORE_CUDA_IMAGE_OPS_H

#include "arithmetic.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuimage/cuda/devptr.h>

namespace core
{

using cuimage::DevPtr;

inline __device__ bool d_isvalid(const float& d)
{
    return (!isnan(d) && d > 0.f);
}

inline __device__ bool d_isvalid(const float3& d)
{
    return (d_isvalid(d.x) && d_isvalid(d.y) && d_isvalid(d.z));
}

inline __device__ float d_diff(const float& d, const float& dn)
{
    if (d_isvalid(d) && d_isvalid(dn))
        return d - dn;
    else
        return 0;
}

inline __device__ float2 d_fwdGradient(
    const DevPtr<float>& depth, int posX, int posY)
{
    const float d = depth(posX, posY);

    const int width = depth.width;
    const int height = depth.height;

    float2 grad = make_float2(0, 0);

    if (posX < width - 1)
        grad.x = d_diff(depth(posX + 1, posY), d);

    if (posY < height - 1)
        grad.y = d_diff(depth(posX, posY + 1), d);

    return grad;
}

inline __device__ float2 d_bwdGradient(
    const DevPtr<float>& depth, int posX, int posY)
{
    const float d = depth(posX, posY);

    float2 grad = make_float2(0, 0);

    if (posX > 1)
        grad.x = d_diff(d, depth(posX - 1, posY));

    if (posY > 1)
        grad.y = d_diff(d, depth(posX, posY - 1));

    return grad;
}

inline __device__ float2 d_symGradient(
    const DevPtr<float>& depth, int posX, int posY)
{
    const float d = depth(posX, posY);

    const int width = depth.width;
    const int height = depth.height;

    float2 grad = make_float2(0, 0);

    // 0: Forward, width: Backward
    if (posX == 0)
        grad.x = d_diff(depth(posX + 1, posY), d);
    else if (posX == width - 1)
        grad.x = d_diff(d, depth(posX - 1, posY));
    else
        grad.x = .5f * d_diff(depth(posX + 1, posY), depth(posX - 1, posY));

    // 0: Forward, height: Backward
    if (posY == 0)
        grad.y = d_diff(depth(posX, posY + 1), d);
    else if (posY == height - 1)
        grad.y = d_diff(d, depth(posX, posY - 1));
    else
        grad.y = .5f * d_diff(depth(posX, posY + 1), depth(posX, posY - 1));

    return grad;
}

inline __device__ float3 d_normals(const float& d, const float& dx,
    const float& dy, const int px, const int py, const float& fx,
    const float& fy, const float& cx, const float& cy)
{
    float3 n
        = make_float3(fx * dx, fy * dy, -d - (px - cx) * dx - (py - cy) * dy);
    float dz = max(1e-12, norm(n));
    return n / make<float3>(dz);
}

inline __device__ float3 d_transform(
    const float3 p, const DevPtr<float>& R, const DevPtr<float>& T)
{
    float3 p_out = make<float3>(0.f);
    p_out.x = R(0, 0) * p.x + R(0, 1) * p.y + R(0, 2) * p.z + T(0, 0);
    p_out.y = R(1, 0) * p.x + R(1, 1) * p.y + R(1, 2) * p.z + T(0, 1);
    p_out.z = R(2, 0) * p.x + R(2, 1) * p.y + R(2, 2) * p.z + T(0, 2);
    return p_out;
}

inline __device__ float3 d_transform(const float3 p, const DevPtr<float>& T)
{
    float3 p_out = make<float3>(0.f);
    p_out.x = T(0, 0) * p.x + T(0, 1) * p.y + T(0, 2) * p.z + T(0, 3);
    p_out.y = T(1, 0) * p.x + T(1, 1) * p.y + T(1, 2) * p.z + T(1, 3);
    p_out.z = T(2, 0) * p.x + T(2, 1) * p.y + T(2, 2) * p.z + T(2, 3);
    return p_out;
}

inline __device__ float3 d_backproject(const float2 p, const float d,
    const float& fx, const float& fy, const float& cx, const float& cy)
{
    float x0 = (p.x - cx) / fx;
    float y0 = (p.y - cy) / fy;
    return make_float3(x0 * d, y0 * d, d);
}

inline __device__ float3 d_backproject(
    const float2 p, const float d, const DevPtr<float>& intrinsics)
{
    float x0 = (p.x - intrinsics(0, 2)) / intrinsics(0, 0);
    float y0 = (p.y - intrinsics(1, 2)) / intrinsics(1, 1);
    return make_float3(x0 * d, y0 * d, d);
}

inline __device__ float2 d_project(const float3 p, const float& fx,
    const float& fy, const float& cx, const float& cy)
{
    float2 p_out = make<float2>(0.f);
    p_out.x = (fx * p.x) / p.z + cx;
    p_out.y = (fy * p.y) / p.z + cy;
    return p_out;
}

inline __device__ float2 d_project(
    const float3 p, const DevPtr<float>& intrinsics)
{
    float2 p_out = make<float2>(0.f);
    p_out.x = (intrinsics(0, 0) * p.x) / p.z + intrinsics(0, 2);
    p_out.y = (intrinsics(1, 1) * p.y) / p.z + intrinsics(1, 2);
    return p_out;
}

template <typename T>
inline __device__ T d_interpolate(const DevPtr<T>& i, const float2& p)
{
    int x0 = static_cast<int>(p.x);
    int y0 = static_cast<int>(p.y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float x1_weight = p.x - static_cast<float>(x0);
    float y1_weight = p.y - static_cast<float>(y0);
    float x0_weight = 1.0f - x1_weight;
    float y0_weight = 1.0f - y1_weight;

    if (x0 < 0 || x0 >= i.width)
        x0_weight = 0.0f;
    if (x1 < 0 || x1 >= i.width)
        x1_weight = 0.0f;
    if (y0 < 0 || y0 >= i.height)
        y0_weight = 0.0f;
    if (y1 < 0 || y1 >= i.height)
        y1_weight = 0.0f;

    float w00 = x0_weight * y0_weight;
    float w10 = x1_weight * y0_weight;
    float w01 = x0_weight * y1_weight;
    float w11 = x1_weight * y1_weight;

    float sumWeights = w00 + w10 + w01 + w11;
    T sum = make<T>(0.0f);
    if (w00 > 0.0f)
        sum += i(x0, y0) * w00;
    if (w01 > 0.0f)
        sum += i(x0, y1) * w01;
    if (w10 > 0.0f)
        sum += i(x1, y0) * w10;
    if (w11 > 0.0f)
        sum += i(x1, y1) * w11;

    if (sumWeights > 0.0f)
        return sum / sumWeights;

    return make<T>(std::nanf(""));
}

template <typename T>
inline __device__ void d_splat_naive(DevPtr<T>& i, const float2& p, const T& value)
{
    int x0 = static_cast<int>(p.x);
    int y0 = static_cast<int>(p.y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    i(x0, y0) = value;
    i(x0, y1) = value;
    i(x1, y0) = value;
    i(x1, y1) = value;
}

} // core

#endif // CORE_CUDA_IMAGE_OPS_H
