/**
 * @file   smem.h
 * @brief  Utilities for working with shared memory
 * @author Florian Windolf
 */
#ifndef CORE_CUDA_SMEM_H
#define CORE_CUDA_SMEM_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuimage/cuda.h>

using cuimage::DevPtr;
using cuimage::uchar;

// Read robustly from data
template <typename T>
inline __device__ T read_clamped(const DevPtr<T>& data, int x, int y)
{
    return data(
        max(min(x, data.width - 1), 0), max(min(y, data.height - 1), 0));
}

// Read from data but return nan if not in image
template <typename T>
inline __device__ T read_in(const DevPtr<T>& data, int x, int y)
{
    if (x < 0 || x > data.width - 1)
        return nanf("");
    else if (y < 0 || y > data.height - 1)
        return nanf("");
    else
        return data(x, y);
}

// Fill smem horizontally with data and "replicate" borders
inline __device__ void fill_x(DevPtr<float> smem, const DevPtr<float>& data,
    const DevPtr<uchar>& mask, const dim3 idx, const dim3 pos,
    const int blocksize)
{
    assert(smem.width == 3 * blocksize);

    // Offsets for data
    int off_x[] = {-blocksize, 0, blocksize};

    // Offsets for smem
    int off_sx[] = {0, blocksize, 2 * blocksize};

    // Iterate left|middle|right
    for (int i = 0; i < 3; i++)
    {
        // For data access
        int px = pos.x + off_x[i];
        int py = pos.y;

        // For smem access
        int sx = idx.x + off_sx[i];
        int sy = idx.y;

        if (read_clamped(mask, px, py))
            smem(sx, sy) = read_in(data, px, py);
        else
            smem(sx, sy) = nanf("");
    }
    __syncthreads();
}

// Fill smem with data and "replicate" borders
inline __device__ void fill_y(DevPtr<float> smem, const DevPtr<float>& data,
    const DevPtr<uchar>& mask, const dim3 idx, const dim3 pos,
    const int blocksize)
{
    assert(smem.height == 3 * blocksize);

    // Offsets for data
    int off_y[] = {-blocksize, 0, blocksize};

    // Offsets for smem
    int off_sy[] = {0, blocksize, 2 * blocksize};

    // Iterate left|middle|right
    for (int i = 0; i < 3; i++)
    {
        // For data access
        int px = pos.x;
        int py = pos.y + off_y[i];

        // For smem access
        int sx = idx.x;
        int sy = idx.y + off_sy[i];

        if (read_clamped(mask, px, py))
            smem(sx, sy) = read_in(data, px, py);
        else
            smem(sx, sy) = nanf("");
    }
    __syncthreads();
}

inline __device__ void fill(DevPtr<float> smem, const DevPtr<float>& data,
    const DevPtr<uchar>& mask, const dim3 idx, const dim3 pos,
    const int blocksize)
{
    assert(smem.width == 3 * blocksize);
    assert(smem.height == 3 * blocksize);

    // Offsets for data
    int off_x[] = {-blocksize, 0, blocksize};

    // Offsets for smem
    int off_sx[] = {0, blocksize, 2 * blocksize};

    // Iterate left|middle|right
    for (int i = 0; i < 3; i++)
    {
        // For data access
        dim3 pos_ = dim3(pos.x + off_x[i], pos.y, pos.z);

        // For smem access
        dim3 idx_ = dim3(idx.x + off_sx[i], idx.y, idx.z);

        fill_y(smem, data, mask, idx_, pos_, blocksize);
    }
}


#endif // CORE_CUDA_SMEM_H