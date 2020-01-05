#include "color_cu.h"

#include "core/cuda_kernel.h"

#include <cuimage/operations/reduce_cu.h>

using namespace core;

__global__ void g_BoxFilter(DevPtr<float> output, const DevPtr<float> image, int radius, bool x)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if(pos.x >= image.width || pos.y >= image.height)
        return;

    int num = 0;
    float sum = 0;

    if (x)
    {
        for (int r = -radius; r <= radius; r++)
        {
            const int px = pos.x + r;
          
            if (px < 0 || px > image.width)
                continue;
            
            num += 1;
            sum += image(px, pos.y);
        }
    }
    else
    {   
        for (int r = -radius; r <= radius; r++)
        {
            const int py = pos.y + r;
            
            if (py < 0 || py > image.width)
                continue;
            
            num += 1;
            sum += image(pos.x, py);
        }
    }    

    output(pos.x, pos.y) = sum / num;
}

__global__ void g_Variations(DevPtr<float> variation_x, 
                             DevPtr<float> variation_y, 
                             DevPtr<float> gradient_x,
                             DevPtr<float> gradient_y,
                             const DevPtr<float> image,
                             const DevPtr<float> image_blurred_x,
                             const DevPtr<float> image_blurred_y)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    if(pos.x >= image.width || pos.y >= image.height)
        return;

    // Gradients Backward Difference
    float2 D_F = abs(d_bwdGradient(image, pos.x, pos.y));
    float2 D_Bx = abs(d_bwdGradient(image_blurred_x, pos.x, pos.y));
    float2 D_By = abs(d_bwdGradient(image_blurred_y, pos.x, pos.y));
    float2 D_B = make_float2(D_Bx.x, D_By.y);

    // Positive Variations
    float2 V = max(make_float2(0, 0), D_F - D_B);

    gradient_x(pos.x, pos.y) = D_F.x;
    gradient_y(pos.x, pos.y) = D_F.y;
    variation_x(pos.x, pos.y) = V.x;
    variation_y(pos.x, pos.y) = V.y;
}

float cu_Blurriness(const DevPtr<float> image)
{
    // Blur image
    DevPtr<float> Bx(image.width, image.height);
    DevPtr<float> By(image.width, image.height);

    dim3 block = block2D(32);
    dim3 grid = grid2D(image.width, image.height, block);

    g_BoxFilter <<< grid, block >>> (Bx, image, 3, true); // x
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());

    g_BoxFilter <<< grid, block >>> (By, image, 3, false); // y
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());

    DevPtr<float> Gx(image.width, image.height);
    DevPtr<float> Gy(image.width, image.height);

    DevPtr<float> Vx(image.width, image.height);
    DevPtr<float> Vy(image.width, image.height);

    g_Variations <<< grid, block >>> (Vx, Vy, Gx, Gy, image, Bx, By);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());

    float s_Fx = cuimage::cu_Sum(Gx);
    float s_Fy = cuimage::cu_Sum(Gy);
    float s_Vx = cuimage::cu_Sum(Vx);
    float s_Vy = cuimage::cu_Sum(Vy);

    float bx = (s_Fx - s_Vx) / s_Fx;
    float by = (s_Fy - s_Vy) / s_Fy;

    return cuimage::max(bx, by);
}
