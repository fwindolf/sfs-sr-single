#include "solver/optimizer/inpaint_cu.h"

#include "core/cuda_kernel.h"

using namespace core;

__global__ void g_PatchHoles(DevPtr<float> depth, const int radius)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = depth.width;
    const int height = depth.height;

    if (pos.x >= width || pos.y >= height)
        return;

    const float v = depth(pos.x, pos.y);
    if(isvalid(v) && v > 0)
        return;

    // Get mean of valid surrounding pixels
    float sum = 0.f;
    int num = 0;
    for (int yr = -radius; yr <= radius; yr++)
    {
    for (int xr = -radius; xr <= radius; xr++)
    {
        const int px = pos.x + xr;
        const int py = pos.y + yr;

        if (px < 0 || px >= width || py < 0 || py >= height)
            continue;

        const float n = depth(px, py);
        if (isvalid(n) && n > 0)
        {
            sum += n;
            num++;
        }
    }
    }

    // No valid neighbors
    if (num == 0)
        depth(pos.x, pos.y) = nanf("");

    depth(pos.x, pos.y) = sum / num;
}


void cu_PatchHoles(DevPtr<float> depth)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(depth.width, depth.height, block);

    g_PatchHoles <<< grid, block >>> (depth, 3);
}