/**
 * @file   environment.h
 * @brief  Wrapper for a cuda environment
 * @author Robert Maier
 * @author Florian Windolf
 */
#ifndef CORE_CUDA_ENVIRONMENT_H
#define CORE_CUDA_ENVIRONMENT_H

#include <cuda_runtime.h>

namespace core
{

/**
 * @class  CudaEnvironment
 * @brief  Environment wrapper in C++11 thread-safe manner
 * @author Florian Windolf
 */
class CudaEnvironment
{
public:
    static CudaEnvironment& get();

    bool isValid() const;

    /**
     * Intializes deviceProps if it does not match the number of devices
     */
    void printInfo() const;

    int deviceCount() const;

protected:
    CudaEnvironment();
    ~CudaEnvironment() {}; // Infinite lifetime of data

private:
    CudaEnvironment(CudaEnvironment const&) = delete;
    CudaEnvironment& operator=(CudaEnvironment const&);
    
    CudaEnvironment(CudaEnvironment&&) = delete;
    CudaEnvironment& operator=(CudaEnvironment&&) = delete;

    int deviceCount_;
    cudaDeviceProp deviceProp_;
};
    
} // core

#endif // CORE_CUDA_ENVIRONMENT_H