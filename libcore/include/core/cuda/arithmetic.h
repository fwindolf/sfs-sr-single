#ifndef CORE_CUDA_ARITHM_H
#define CORE_CUDA_ARITHM_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuimage/cuda/arithmetic.h>

namespace core
{

/**
 * Expose some cuimage features into core namespace
 */
using cuimage::uchar;

using cuimage::make;

using cuimage::operator<;
using cuimage::operator<=;
using cuimage::operator==;
using cuimage::operator>=;
using cuimage::operator>;

using cuimage::operator+;
using cuimage::operator+=;
using cuimage::operator-;
using cuimage::operator-=;
using cuimage::operator*;
using cuimage::operator*=;
using cuimage::operator/;
using cuimage::operator/=;

using cuimage::abs;
using cuimage::max;
using cuimage::min;
using cuimage::norm;
using cuimage::normalize;
using cuimage::sum;

using cuimage::isinf;
using cuimage::isnan;
using cuimage::isvalid;
using cuimage::iszero;

/**
 * Additional functionality
 */

inline __host__ __device__ float3 cross(const float3& l, const float3& r)
{
    return make_float3(
        l.y * r.z - r.y * l.z, l.z * r.x - r.z * l.x, l.x * r.y - r.x * l.y);
}

inline __host__ __device__ float dot(const float3& l, const float3& r)
{
    return l.x * r.x + l.y * r.y + l.z * r.z;
}

template <typename T> inline __host__ __device__ T operator-(const T& val)
{
    return make<T>(0.f) - val;
}

/**
 * Overloads for min/max atomic operations for floats
 * https://stackoverflow.com/a/51549250/4658360
 */

inline __device__ float atomicMinF(float* addr, float value)
{
    float old;
    old = (value >= 0)
        ? __int_as_float(atomicMin((int*)addr, __float_as_int(value)))
        : __uint_as_float(
              atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

inline __device__ float atomicMaxF(float* addr, float value)
{
    float old;
    old = (value >= 0)
        ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
        : __uint_as_float(
              atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

} // core

#endif // CORE_CUDA_ARITHM_H
