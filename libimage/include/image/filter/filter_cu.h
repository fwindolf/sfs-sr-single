/**
 * @file   filter_cu.h
 * @brief  Filter implementations in cuda
 * @author Florian Windolf
 */
#ifndef FILTER_FILTER_CU_H
#define FILTER_FILTER_CU_H

#include "core/cuda.h"

namespace image
{

template <typename T>
void cu_BilateralFilter(cuimage::DevPtr<T>& output, const cuimage::DevPtr<T> input, const int radius, const float sigmaS, const float sigmaR);

template <typename T>
void cu_DiscontinuityFilter(cuimage::DevPtr<T>& output, const cuimage::DevPtr<T> input, const int radius, const float threshold);

} // image

#endif // FILTER_FILTER_CU_H