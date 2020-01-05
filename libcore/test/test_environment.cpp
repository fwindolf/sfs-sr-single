#include <gtest/gtest.h>

#include "core/cuda/environment.h"

using namespace core;
using namespace testing;

TEST(CudaEnv, initializes_on_init_with_correct_device_count)
{
    auto& env = CudaEnvironment::get();

    int devCnt;
    cudaGetDeviceCount(&devCnt);
    EXPECT_EQ(devCnt, env.deviceCount());    
}

TEST(CudaEnv, is_singleton)
{
    auto& env1 = CudaEnvironment::get();
    auto& env2 = CudaEnvironment::get();

    EXPECT_EQ(&env1, &env2);
}

TEST(CudaEnv, prints_device_info)
{
    auto& env = CudaEnvironment::get();
    if(env.deviceCount() > 0)
    {
        internal::CaptureStdout();
        env.printInfo();

        EXPECT_FALSE(internal::GetCapturedStdout().empty());
    }    
}