#include "image/filter/mumfordshah_cu.h"

#include <cuimage/cuda/arithmetic.h>
#include "core/cuda_kernel.h"

using namespace core;
using namespace cuimage;

namespace image
{

template <typename T>
__device__ void d_calcGradient(const DevPtr<T>& u,
                               const DevPtr<uchar>& mask,
                               T& ux, 
                               T& uy)
{

    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = u.width;
    const int height = u.height;

    if (pos.x < width - 1 && mask(pos.x, pos.y) && mask(pos.x + 1, pos.y))
        ux = u(pos.x + 1, pos.y) - u(pos.x, pos.y);

    if (pos.y < height - 1 && mask(pos.x, pos.y) && mask(pos.x, pos.y + 1))
        uy = u(pos.x, pos.y + 1) - u(pos.x, pos.y);
}


template <typename T>
__device__ T d_calcDivergence(const DevPtr<T>& v1, 
                              const DevPtr<T>& v2,
                              const DevPtr<uchar>& mask)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    T v1x = make<T>(0.f);
    T v2y = make<T>(0.f);

    if (pos.x > 0 && mask(pos.x, pos.y) && mask(pos.x - 1, pos.y))
        v1x = v1(pos.x, pos.y) - v1(pos.x - 1, pos.y);
    
    if (pos.y > 0 && mask(pos.x, pos.y) && mask(pos.x, pos.y - 1))
        v2y = v2(pos.x, pos.y) - v2(pos.x, pos.y - 1);
    
    return -(v1x + v2y);
}

template <typename T>
__device__ T d_square_prox(const T& x0, const T& c, const T& f, const float tau)
{
    const T c_ =  2.f * tau * c;
    return (x0 + c_ * f) / (1.f + c_ * c);
}

template <typename T>
__global__ void g_UpdateDual(DevPtr<T> px,
                             DevPtr<T> py, 
                             const DevPtr<T> u_bar,
                             const DevPtr<uchar> mask,
                             float sigma, float alpha, float lambda)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);
    
    const int width = u_bar.width;
    const int height = u_bar.height;

    if ( pos.x >= width || pos.y >= height)
        return;

    if (!mask(pos.x, pos.y))
        return;

    T u_bar_x = make<T>(0.f);
    T u_bar_y = make<T>(0.f);
    d_calcGradient(u_bar, mask, u_bar_x, u_bar_y);

    px(pos.x, pos.y) += sigma * u_bar_x;
    py(pos.x, pos.y) += sigma * u_bar_y;

    T px_new = px(pos.x, pos.y);
    T py_new = py(pos.x, pos.y);

    float norm_p_squared = sum(px_new * px_new) + sum(py_new * py_new);

    if (alpha == -1.f)
    {        
        if (norm_p_squared > 2.f * lambda * sigma)
        {
            px(pos.x, pos.y) = make<T>(0.f);
            py(pos.x, pos.y) = make<T>(0.f);
        }
    }
    else if (alpha > 0.f)
    {
        if (norm_p_squared > lambda / alpha * sigma * (sigma + 2.f * alpha))
        {
            px(pos.x, pos.y) = make<T>(0.f);
            py(pos.x, pos.y) = make<T>(0.f);
        }
        else
        {
            float scale = 2 * alpha / (sigma + 2 * alpha);
            px(pos.x, pos.y) *= make<T>(scale);
            py(pos.x, pos.y) *= make<T>(scale);
        }
    }
}

template <typename T>
__global__ void g_UpdatePrimal(DevPtr<T> u,
                               DevPtr<T> u_bar,
                               DevPtr<T> u_diff,
                               const DevPtr<T> f, 
                               const DevPtr<T> px,
                               const DevPtr<T> py,
                               const DevPtr<T> scalar_op,
                               const DevPtr<uchar> mask,
                               float tau, float theta)
{
    const dim3 pos = getPos(blockIdx, blockDim, threadIdx);

    const int width = f.width;
    const int height = f.height;

    if(pos.x >= width || pos.y >= height) 
        return;

    if(!mask(pos.x, pos.y)) 
        return;

    const T u_old = u(pos.x, pos.y);
    T divp = d_calcDivergence(px, py, mask);

    const T c = scalar_op(pos.x, pos.y);
    T u_new = d_square_prox<T>(u_old - divp * tau, c, f(pos.x, pos.y), tau);

    u_bar(pos.x, pos.y) = u_new + theta * (u_new - u_old);
    u(pos.x, pos.y) = u_new;

    u_diff(pos.x, pos.y) = abs(u_new - u_old);
}

template <typename T>
void cu_UpdatePrimal(DevPtr<T> u,
                     DevPtr<T> u_bar, 
                     DevPtr<T> u_diff,
                     const DevPtr<T> intensity,                         
                     const DevPtr<T> px,
                     const DevPtr<T> py,
                     const DevPtr<T> scalar_op,
                     const DevPtr<uchar> mask,
                     float tau, float theta)

{
    dim3 block = block2D(32);
    dim3 grid = grid2D(intensity.width, intensity.height, block);

    g_UpdatePrimal <<< grid, block >>>(u, u_bar, u_diff, intensity, px, py, scalar_op, mask, tau, theta);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());
}

template <typename T>
void cu_UpdateDual(DevPtr<T> px,
                   DevPtr<T> py,
                   const DevPtr<T> u_bar, 
                   const DevPtr<uchar> mask,
                   float sigma, float alpha, float lambda)
{
    dim3 block = block2D(32);
    dim3 grid = grid2D(u_bar.width, u_bar.height, block);

    g_UpdateDual <<< grid, block >>>(px, py, u_bar, mask, sigma, alpha, lambda);
    cudaCheckLastCall();
    // cudaSafeCall(cudaDeviceSynchronize());   
}

/**
 * Explicit instantiation
 */

#define DEFINE_PRIMAL_VAR_FUNCTION(type, name) \
    template void name(DevPtr<type>, DevPtr<type>, DevPtr<type>, const DevPtr<type>, const DevPtr<type>, const DevPtr<type>, const DevPtr<type>, const DevPtr<uchar>, float, float);

FOR_EACH_TYPE(DEFINE_PRIMAL_VAR_FUNCTION, cu_UpdatePrimal);

#define DEFINE_DUAL_VAR_FUNCTION(type, name) \
    template void name(DevPtr<type>, DevPtr<type>, const DevPtr<type>, const DevPtr<uchar>, float, float, float);

FOR_EACH_TYPE(DEFINE_DUAL_VAR_FUNCTION, cu_UpdateDual);

} //