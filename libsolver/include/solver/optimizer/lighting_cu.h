#include "core/cuda.h"

template <typename T>
void cu_CalculateAMatrix(DevPtr<float> output,
                         const DevPtr<T> albedo,
                         const DevPtr<float4> harmonics,
                         const DevPtr<uchar> mask);

template <typename T>
void cu_CalculateBMatrix(DevPtr<float> output,
                         const DevPtr<T> image,
                         const DevPtr<uchar> mask);

void cu_CalculateATA(DevPtr<float> output, 
                     const DevPtr<float> a, 
                     cublasHandle_t cublasHandle);

void cu_CalculateATb(DevPtr<float> output,
                     const DevPtr<float> a,
                     const DevPtr<float> b,
                     cublasHandle_t cublasHandle);

template <typename T>
void cu_CalculateShading(DevPtr<T> output, 
                         const DevPtr<float4> harmonics,
                         const DevPtr<T> light,
                         const DevPtr<uchar> mask);

template <typename T>
void cu_InitializeLighting(DevPtr<T> light);

template <typename T>
void cu_SetLighting(DevPtr<T> light,
                    const DevPtr<float> input);