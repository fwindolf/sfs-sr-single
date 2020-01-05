#include "core/cuda.h"

void cu_CopyUpperSubmatrix(float* dst, const float* src, const int M, const int N, const int subM);

void cu_CopyTransposed(float* dst, const DevPtr<float> src);

void cu_CopyTransposed(DevPtr<float> dst, const float* src);
