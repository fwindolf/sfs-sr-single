#include <gtest/gtest.h>

#if TEST_LIGHT

#include "image/data.h"

#include "solver/optimizer/lighting.h"
#include "solver/optimizer/lighting_cu.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace solver;

class LightingTest : public testing::Test
{
public:
    LightingTest()
     : w_(640), h_(480),
      d(std::string(SOURCE_DIR) + "/data/", w_, h_, 1),
      mask(d.mask),
      image(d.image),
      albedo(d.albedo),
      shading(d.shading),
      lighting(d.light)
    {
        image::DepthProcessing p(d.depth_star);
        p.harmonics(spherical_harmonics, d.K_hr);
    }
    const int w_, h_;
    image::DataSet d;

    const Image<float3>& lighting;
    const Image<float3>& image;
    const Image<float3>& shading;
    const Image<float3>& albedo;
    const Image<uchar>& mask;

    Image<float4> spherical_harmonics;
};

TEST(LightCuTest, calculates_ATA)
{
    int W = 4;
    int H = 100;

    cv::Mat a = cv::Mat(H, W, CV_32FC1);
    for (int i = 0; i < W * H; i++)
        a.at<float>(i) = i + 1;         
    
    cv::Mat ata_comp = a.t() * a;

    Image<float> mA(W, H);
    mA.upload((float*)a.data, W, H);

    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    Image<float> mATA(W, W);

    cu_CalculateATA(mATA, mA, handle);

    float* d_ata = mATA.download();
    cv::Mat ata(W, W, CV_32FC1, d_ata);
    
    for (int i = 0; i < W * W; i++)
        EXPECT_FLOAT_EQ(ata.at<float>(i), ata_comp.at<float>(i));

    cublasSafeCall(cublasDestroy(handle));
}

TEST(LightCuTest, calculates_ATb)
{
    int W = 3;
    int P = 1;
    int H = 10;

    cv::Mat a(H, W, CV_32FC1);
    for (int i = 0; i < W * H; i++)
        a.at<float>(i) = i + 1;         

    cv::Mat b(H, P, CV_32FC1);
    for (int i = 0; i < P * H; i++)
        b.at<float>(i) = i + 20;         

    cv::Mat atb_comp = a.t() * b;

    Image<float> mA(W, H);
    mA.upload((float*)a.data, W, H);

    Image<float> mB(P, H);
    mB.upload((float*)b.data, P, H);

    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    Image<float> mATb(W, P);
    cu_CalculateATb(mATb, mA, mB, handle);

    float* result = mATb.download();
    cv::Mat atb(P, W, CV_32FC1, result);
    
    for (int i = 0; i < W * P; i++)
        EXPECT_FLOAT_EQ(atb.at<float>(i), atb_comp.at<float>(i));

    cublasSafeCall(cublasDestroy(handle));
}
TEST_F(LightingTest, finds_solution_toy_example)
{
    const int W = 2, H = 2, C = 3, K = 4;

    float4 h_spherical_harmonics[] = {
        make_float4(0.9961, 0.9619, 0.8687, 1),
        make_float4(0.4427, 0.7749, 0.3998, 1),
        make_float4(0.0782, 0.0046, 0.0844, 1),
        make_float4(0.1067, 0.8173, 0.2599, 1)
    };

    float3 h_albedo[] = {
        make_float3(0.8001, 0.2638, 0.5797),
        make_float3(0.9106, 0.1361, 0.1450),
        make_float3(0.4314, 0.1455, 0.5499),
        make_float3(0.1818, 0.8693, 0.8530)
    };

    float3 h_image[] = {
        make_float3(0.9595, 0.3164, 0.6952),
        make_float3(0.6080, 0.0908, 0.0968),
        make_float3(0.0792, 0.0267, 0.1009),
        make_float3(0.0868, 0.4150, 0.4072)
    };

    float3 h_light[] = {
        make_float3(0.3, 0.3, 0.3),
        make_float3(0.2, 0.2, 0.2),
        make_float3(0.7, 0.7, 0.7),
        make_float3(0.1, 0.1, 0.1)
    };

    uchar h_mask[] = {
        1, 1, 1, 1
    };
   
    Image<float3> image(W, H);
    image.upload(h_image, W, H);

    Image<float3> albedo(W, H);
    albedo.upload(h_albedo, W, H);

    Image<float3> light(1, K);
    light.upload(h_light, 1, K);

    Image<float4> sharm(W, H);
    sharm.upload(h_spherical_harmonics, W, H);

    Image<uchar> mask(W, H);
    mask.upload(h_mask, W, H);

    LightingOptimizer optimizer(W, H, 1);
    optimizer.init(image, mask, sharm);
    optimizer.step(mask, sharm, albedo); 

    for (int i = 0; i < 4; i++)
    {
        auto l_in = h_light[0].x;
        auto l_out = optimizer.lighting_opt.at(0, i);
        EXPECT_NEAR(l_in[0], l_out.x, 1e-3);
        EXPECT_NEAR(l_in[1], l_out.y, 1e-3);
        EXPECT_NEAR(l_in[2], l_out.z, 1e-3);
    }
}

TEST_F(LightingTest, finds_solution_color)
{
    LightingOptimizer optimizer(w_, h_, 1);
    optimizer.init(image, mask, spherical_harmonics);
    optimizer.step(mask, spherical_harmonics, albedo); 

    for (int i = 0; i < 4; i++)
    {
        auto l_in = lighting.at(i, 0);
        auto l_out = optimizer.lighting_opt.at(0, i);
        EXPECT_NEAR(l_in.x, l_out.x, 1e-3);
        EXPECT_NEAR(l_in.y, l_out.y, 1e-3);
        EXPECT_NEAR(l_in.z, l_out.z, 1e-3);
    }

    // Expect the calculated shading to be close to the original shading inside mask
    Image<float3> diff = shading - optimizer.shading_opt;
    diff.mask(mask);
    EXPECT_NEAR(0.f, diff.norm2(), 1.f);
}

#endif
