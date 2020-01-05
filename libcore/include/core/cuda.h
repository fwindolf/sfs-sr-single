/**
 * @file   cuda.h
 * @brief  Includes for cuda related functionality
 * @author Florian Windolf
 */
#ifndef CORE_CUDA_H
#define CORE_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuimage/cuda/type.h>
using cuimage::uchar;

using cuimage::is_vector_type;
using cuimage::is_float_type;
using cuimage::is_uchar_type;
using cuimage::is_int_type;

using cuimage::has_0_channels;
using cuimage::has_1_channels;
using cuimage::has_2_channels;
using cuimage::has_3_channels;
using cuimage::has_4_channels;

#include <cuimage/cuda/devptr.h>
using cuimage::DevPtr;

#include "core/cuda/utils.h"


#endif // CORE_CUDA_H