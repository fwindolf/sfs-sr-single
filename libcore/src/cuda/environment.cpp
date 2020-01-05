#include "core/cuda/environment.h"
#include "core/cuda/utils.h"

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>


using namespace core;

CudaEnvironment::CudaEnvironment()
  : deviceCount_(0)
{
    cudaSafeCall(cudaGetDeviceCount(&deviceCount_));
    cudaSafeCall(cudaSetDevice(deviceCount_ - 1));
    cudaSafeCall(cudaGetDeviceProperties(&deviceProp_, deviceCount_ -  1));
}

CudaEnvironment& CudaEnvironment::get()
{
    // C++11 blocks until statics are initialized in a threaded context
    static CudaEnvironment instance;    
    return instance;
}

bool CudaEnvironment::isValid() const
{
    return true;
}


// C++11 does not support generic lambdas, so have to define a function and convert everything to string first...
void formatCout(std::string pname, std::string prop, std::string unit="") 
{
    std::cout << "| " << std::left << std::setw(30) << pname
              << "| " << std::right << std::setw(20) << prop
              << "| " << std::right << std::setw(4) << unit
              << "|" << std::endl;
    
};


void CudaEnvironment::printInfo() const
{
    std::cout << "Device Informations:" << std::endl;
    std::cout << "|-------------------------------|---------------------|-----|" << std::endl;
    formatCout("Name", deviceProp_.name);
    formatCout("Compute Capability", std::to_string(deviceProp_.major) + "." + std::to_string(deviceProp_.minor));
    formatCout("Memory Clock Rate", std::to_string(deviceProp_.memoryClockRate / 1e6), "GHz");
    formatCout("Memory Bus Width", std::to_string(deviceProp_.memoryBusWidth), "bits");
    formatCout("Peak Memory Bandwidth", std::to_string(2.f * deviceProp_.memoryClockRate * (deviceProp_.memoryBusWidth / 8) / 1e6), "GB/s");
    formatCout("Total global memory", std::to_string(deviceProp_.totalGlobalMem / (1024 * 1024)), "MB");
    formatCout("Total shared memory per block", std::to_string(deviceProp_.sharedMemPerBlock / 1024), "KB");
    formatCout("Total registers per block", std::to_string(deviceProp_.regsPerBlock));
    formatCout("Warp size", std::to_string(deviceProp_.warpSize));
    formatCout("Maximum memory pitch", std::to_string(deviceProp_.memPitch/ (1024 * 1024)), "MB");
    formatCout("Maximum threads per block", std::to_string(deviceProp_.maxThreadsPerBlock));
    formatCout("Clock rate", std::to_string(deviceProp_.clockRate / 1e6), "GHz");
    formatCout("Total constant memory", std::to_string(deviceProp_.totalConstMem / 1024), "KB");
    formatCout("Number of multiprocessors", std::to_string(deviceProp_.multiProcessorCount));
    std::cout << "|-------------------------------|---------------------|-----|" << std::endl;
    std::cout << std::flush;
}

int CudaEnvironment::deviceCount() const
{
    return deviceCount_;
}