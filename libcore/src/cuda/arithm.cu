#include "core/cuda/arithm.h"

#include <cmath>


inline __host__ __device__ bool hasZero(const float3 v)
{
    return (v.x == 0.f || v.y == 0.f || v.z == 0.f);
}

inline __device__ bool hasNaN(const float3 v)
{
    return (isnan(v.x) || isnan(v.y) || isnan(v.z));
}

inline __device__ bool isNaN(const float3 v)
{
    return (isnan(v.x) && isnan(v.y) && isnan(v.z));
}

inline __device__ bool hasInf(const float3 v)
{
    return (isinf(v.x) || isinf(v.y) || isinf(v.z));
}

inline __device__ bool isInf(const float3 v)
{
    return (isinf(v.x) && isinf(v.y) && isinf(v.z));
}


inline __host__ __device__ bool hasZero(const float v)
{
    return (v == 0.f);
}

inline __device__ bool hasNaN(const float v)
{
    return isnan(v);
}

inline __device__ bool isNaN(const float v)
{
    return isnan(v);
}

inline __device__ bool hasInf(const float v)
{
    return isinf(v);
}

inline __device__ bool isInf(const float v)
{
    return isinf(v);
}