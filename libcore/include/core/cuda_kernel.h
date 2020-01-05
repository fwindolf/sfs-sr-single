/**
 * @file   cuda_kernel.h
 * @brief  Includes to be used in non-header (.cu) files
 * @author Florian Windolf
 */
#ifndef CORE_CUDA_KERNEL_H
#define CORE_CUDA_KERNEL_H

#include <cuimage/cuda/devptr.h>

#include "core/cuda/arithmetic.h"
#include "core/cuda/image_ops.h"
#include "core/cuda/interpolation.h"
#include "core/cuda/utils.h"
#include "core/cuda/smem.h"

#endif //  CORE_CUDA_KERNEL_H