/**
 * @file   utils.h
 * @brief  Cuda utility functions
 * @author Florian Windolf
 */
#ifndef CORE_CUDA_UTILS_H
#define CORE_CUDA_UTILS_H

#include <cuimage/cuda.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolver_common.h>

#include <stdio.h>
#include <iostream>

/** 
 * Adapted from: https://stackoverflow.com/a/13041774/4658360
 */
static std::string cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "Success";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "Not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED: return "Allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE: return "Invalid value"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "Architecture mismatch"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "Mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "Execution failed"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "Internal error";
        default: return "Unknown error";
    }
}

static void cublasCall(cublasStatus_t call_status,  const char* file, const int line)
{
    if(call_status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << file << "(" << line << ") : Cublas call returned with error: " << cublasGetErrorString(call_status) 
                  << std::endl;
        exit(call_status);
    }
}

#define cublasSafeCall(call_status) cublasCall(call_status, __FILE__ ,__LINE__)

static std::string cusolverGetErrorString(cusolverStatus_t status)
{
    switch(status)
    {
        case CUSOLVER_STATUS_SUCCESS: return "Success";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "Not initialized";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "Allocation failed";
        case CUSOLVER_STATUS_INVALID_VALUE: return "Invalid value"; 
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "Architecture mismatch";
        case CUSOLVER_STATUS_MAPPING_ERROR: return "Mapping error"; 
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "Execution failed"; 
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "Internal error";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "Matrix type not supported";
        case CUSOLVER_STATUS_NOT_SUPPORTED: return "Not supported"; 
        case CUSOLVER_STATUS_ZERO_PIVOT: return "Zero pivot";
        case CUSOLVER_STATUS_INVALID_LICENSE: return "Invalid license";
        default: return "Unknown error";
    }
}

static void cusolverCall(cusolverStatus_t call_status, const char* file, const int line)
{
    if(call_status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << file << "(" << line << ") : Cusolver call returned with error: " << cusolverGetErrorString(call_status) 
                  << std::endl;
        exit(call_status);
    }
}

#define cusolverSafeCall(call_status) cusolverCall(call_status, __FILE__ ,__LINE__)

static std::string cusparseGetErrorString(cusparseStatus_t error)
{
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS: return "Success";
        case CUSPARSE_STATUS_NOT_INITIALIZED: return "Not Initialized";
        case CUSPARSE_STATUS_ALLOC_FAILED: return "Allocation failed";
        case CUSPARSE_STATUS_INVALID_VALUE: return "Invalid value"; 
        case CUSPARSE_STATUS_ARCH_MISMATCH: return "Architecture mismatch";
        case CUSPARSE_STATUS_MAPPING_ERROR: return "Mapping error"; 
        case CUSPARSE_STATUS_EXECUTION_FAILED: return "Execution failed"; 
        case CUSPARSE_STATUS_INTERNAL_ERROR: return "Internal error";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "Matrix type not supported";
        case CUSPARSE_STATUS_ZERO_PIVOT: return "Zero pivot";
        default: return "Unknown error";
        }
}


static void cusparseCall(cusparseStatus_t call_status, const char* file, const int line)
{
    if(call_status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cerr << file << "(" << line << ") : Cusparse call returned with error: " << cusparseGetErrorString(call_status) 
                  << std::endl;
        exit(call_status);
    }
}

#define cusparseSafeCall(call_status) cusparseCall(call_status, __FILE__ ,__LINE__)

#endif // CORE_CUDA_UTILS_H