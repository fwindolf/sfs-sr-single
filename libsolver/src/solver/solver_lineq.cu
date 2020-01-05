#include "solver_lineq_cu.h"

#include "core/cuda_kernel.h"

#include "stdio.h"

#define BLOCK_SIZE 32

__global__ void g_CopyTransposed(float *out, const DevPtr<float> in)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = in.width;
    const int height = in.height;

    // in is MxN, out is MxN 
    if (pos.x >= width || pos.y >= height)
        return;

    const int idx = pos.y + pos.x * height;
    out[idx] = in(pos.x, pos.y);
}

__global__ void g_CopyTransposed(DevPtr<float> out, const float *in)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = out.width;
    const int height = out.height;

    // in is MxN, out is MxN 
    if (pos.x >= width || pos.y >= height)
        return;

    const int idx = pos.y + pos.x * height;
    out(pos.x, pos.y) = in[idx];
}

__global__ void g_CopyUpperSubmatrix(float* out, const float* in, const int M, const int N, const int subM)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if (pos.x >= subM || pos.y >= N)
        return;

    out[pos.y * subM + pos.x] = in[pos.y * M + pos.x];
}

void cu_CopyUpperSubmatrix(float* dst, const float* src, const int M, const int N, const int subM)
{
    dim3 block = block2D(BLOCK_SIZE);
    dim3 grid = grid2D(subM, N, block);
    g_CopyUpperSubmatrix <<< grid, block >>>(dst, src, M, N, subM);
    cudaCheckLastCall();
}

void cu_CopyTransposed(float* dst, const DevPtr<float> src)
{
    dim3 block = block2D(BLOCK_SIZE);
    dim3 grid = grid2D(src.width, src.height, block);
    
    g_CopyTransposed <<< grid, block >>> (dst, src);
    cudaCheckLastCall();
}

void cu_CopyTransposed(DevPtr<float> dst, const float *src)
{
    dim3 block = block2D(BLOCK_SIZE);
    dim3 grid = grid2D(dst.height, dst.width, block);
    g_CopyTransposed <<< grid, block >>> (dst, src);
    cudaCheckLastCall();
}