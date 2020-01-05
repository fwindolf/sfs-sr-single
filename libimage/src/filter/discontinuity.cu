#include "image/filter/filter_cu.h"

#include "core/cuda_kernel.h"

using namespace image;
using namespace cuimage;


template <typename T>
__global__ void g_DiscontinuityFilter(DevPtr<T> output, const DevPtr<T> input, const int r, const float t)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if(pos.x >= output.width || pos.y >= output.height)
        return;

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

            const T val_n = input(px, py);
            if (!isvalid(val_n))
                continue;

            // Check if there is a jump
            if(abs(val - val_n) >= make<T>(t))
            {
                output(pos.x, pos.y) = make<T>(0.f);
                return;
            }
        }
    }

    output(pos.x, pos.y) = val;
}

template <typename T>
void cu_DiscontinuityFilter(DevPtr<T>& output, const DevPtr<T> input, const int radius, const float threshold)
{
    assert(output.width == input.width);
    assert(output.height == input.height);
    
    dim3 block = block2D(32);
    dim3 grid = grid2D(output.width, output.height, block);
    
    g_DiscontinuityFilter <<< grid, block >>> (output, input, radius, threshold);
    
    cudaCheckLastCall();
    cudaDeviceSynchronize();
}
