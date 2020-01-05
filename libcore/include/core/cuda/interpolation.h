#ifndef CORE_CUDA_INTERPOLATION_H
#define CORE_CUDA_INTERPOLATION_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuimage/cuda/interpolation.h>

namespace core
{
/**
 * Expose some cuimage features into core namespace
 */

using cuimage::d_interpolate_linear;
using cuimage::d_interpolate_linear_masked;
using cuimage::d_interpolate_linear_valid;
using cuimage::d_interpolate_linear_valid_masked;
using cuimage::d_interpolate_linear_nonzero_masked;

using cuimage::d_interpolate_nearest;
using cuimage::d_interpolate_nearest_valid;
using cuimage::d_interpolate_nearest_masked;
using cuimage::d_interpolate_nearest_valid_masked;


} // core

#endif // CORE_CUDA_INTERPOLATION_H