#include "core/cuda.h"

namespace image
{

template <typename T>
void cu_UpdatePrimal(DevPtr<T> u,
                     DevPtr<T> u_bar,
                     DevPtr<T> u_diff,
                     const DevPtr<T> intensity,
                     const DevPtr<T> px,
                     const DevPtr<T> py,
                     const DevPtr<T> scalar_op,
                     const DevPtr<uchar> mask,
                     float tau, float theta);

template <typename T>
void cu_UpdateDual(DevPtr<T> px,
                   DevPtr<T> py,
                   const DevPtr<T> u_bar,
                   const DevPtr<uchar> mask,
                   float sigma, float alpha, float lambda);

} // namespace image