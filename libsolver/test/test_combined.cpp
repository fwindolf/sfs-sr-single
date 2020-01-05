#include <gtest/gtest.h>
#include <chrono>

#if TEST_COMBINED

#include "image/processing.h"
#include "image/data.h"
#include "image/evaluation.h"

#include "solver/optimizer.h"

#include "../include/solver/optimizer/lighting_cu.h"

using namespace solver;
using namespace testing;

class CombinedTest : public ::testing::Test
{
public:
    CombinedTest()
     : w_(640), h_(480),
       d(std::string(SOURCE_DIR) + "/data/", w_, h_, 1),
       mask(d.mask),
       image(d.image),
       albedo(d.albedo),
       shading(d.shading),
       depth_star(d.depth_star),
       lighting(d.light),
       dual(image.width(), image.height()),
       thetaOpt(w_, h_, gamma, nu, 3, 2),
       depthOpt(w_, h_, mu, 3, 2),
       albedoOpt(w_, h_, alpha, lambda, maxIter),
       lightOpt(w_, h_, 1),
       lagrangeOpt(w_, h_, tau, kappa, rel_res, el_diff)
    {
        image::DepthProcessing p(depth_star);
        p.harmonics(spherical_harmonics, d.K_hr);
        spherical_harmonics.mask(mask);    

        p.addNoise(depth_corrupt, 0, 0.001);
        depth_corrupt.mask(mask);

        p.addNoise(depth_lr, 0, 0.001);
        depth_lr.mask(mask);

        p.theta(theta);
        theta.mask(mask);
        
        image::DepthProcessing pc(depth_corrupt);
        pc.bilateral(depth_smooth, mask, 0.005 * depth_star.width(), 30, 50);
        depth_smooth.mask(mask);

        pc.theta(theta_corrupt);
        theta_corrupt.mask(mask);

        image::DepthProcessing ps(depth_smooth);
        ps.theta(theta_smooth);
        theta_smooth.mask(mask);

        intrinsics = d.K_hr;
    }
    const int w_, h_;
    image::DataSet d;

    const Image<float3>& lighting;
    const Image<float3>& albedo;
    const Image<float3>& image;
    const Image<float>& depth_star;
    const Image<float3>& shading;
    const Image<uchar>& mask;
    
    Image<float3> theta, theta_corrupt, theta_smooth;
    Image<float> depth_corrupt, depth_smooth, depth_lr;
    Image<float4> spherical_harmonics;
    Image<float3> dual;

    float kappa  = 1e-4f;
    float tau    = 4.f;
    float mu     = 1e-2f;
    float gamma  = 1.f;
    float nu     = 1e-4f; 

    float alpha  = -1.f;
    float lambda = 0.5f;
    int maxIter  = 200;
    
    float rel_res = -1; // No stopping based on relative residual
    float el_diff = 1e-6;
    
    AuxiliaryOptimizer thetaOpt;
    DepthOptimizer depthOpt;
    AlbedoOptimizer albedoOpt;
    LightingOptimizer lightOpt;
    LagrangeOptimizer lagrangeOpt;

    IntrinsicsPtr intrinsics;
};
#if 0
TEST_F(CombinedTest, theta_update_admm)
{
    thetaOpt.init(depth_corrupt, mask, depth_star, mask, intrinsics);
    lagrangeOpt.init(depth_corrupt);
  
    int it = 0;
    while(true)
    {   
        thetaOpt.step(image, mask, depth_star, lighting, albedo, lagrangeOpt.dual_opt, intrinsics, lagrangeOpt.kappa);
        thetaOpt.theta_opt.mask(mask);
        EXPECT_EQ(0, thetaOpt.theta_opt.size() - thetaOpt.theta_opt.valid());

        bool converged = lagrangeOpt.step(mask, thetaOpt.theta_opt, depth_star);
        if(converged || it > 100)
            break;

        it++;
    }
    ASSERT_GT(it, 1);
    EXPECT_LT((thetaOpt.theta_opt - theta).norm2(), 1e-4);
}

TEST_F(CombinedTest, depth_update_admm)
{
    depthOpt.init(depth_corrupt, mask, depth_lr, mask);
    lagrangeOpt.init(depthOpt.depth_opt);

    int it = 0;
    while(true)
    {   
        // Should converge to perfect theta with increasing kappa
        depthOpt.step(depth_lr, mask, theta, lagrangeOpt.dual_opt, lagrangeOpt.kappa);
        
        bool converged = lagrangeOpt.step(mask, theta, depthOpt.depth_opt);
        if(converged || it > 100)
            break;

        it++;
    }

    ASSERT_GT(it, 1);
    EXPECT_LT((depthOpt.depth_opt - depth_star).norm2(), 1e-3);
}

TEST_F(CombinedTest, theta_depth_update_admm_with_perfect_lr_depth)
{
    nu = 0;

    depthOpt.init(depth_star, mask, depth_star, mask);
    AuxiliaryOptimizer thetaOptNoSmoothing(w_, h_, gamma, 0, 3, 2);
    thetaOptNoSmoothing.init(depth_smooth, mask, depth_smooth, mask, intrinsics);
    lagrangeOpt.init(depthOpt.depth_opt);
   
    int it = 0;
    while(true)
    {   
        thetaOptNoSmoothing.step(image, mask, depthOpt.depth_opt, lighting, albedo, lagrangeOpt.dual_opt, intrinsics, lagrangeOpt.kappa);
        depthOpt.step(depth_star, mask, thetaOptNoSmoothing.theta_opt, lagrangeOpt.dual_opt, lagrangeOpt.kappa);

        bool converged = lagrangeOpt.step(mask, thetaOptNoSmoothing.theta_opt, depthOpt.depth_opt);
        if(converged || it > 100)
            break;
        
        it++;
    }

    ASSERT_GT(it, 1);
    EXPECT_LT((depthOpt.depth_opt - depth_star).norm2(), (depth_smooth - depth_star).norm2());
}

#endif
TEST_F(CombinedTest, theta_depth_update_admm)
{
    Image<uchar> mask_lr = mask.resized(0.5, mask);
    depth_lr.resize(0.5, mask);
    auto depth_lr_upsampled = depth_lr.resized(2.0, mask_lr);
    
    depthOpt.init(depth_corrupt, mask, depth_lr, mask_lr, mu);
    thetaOpt.init(depthOpt.depth_opt, mask, depth_lr, mask_lr, intrinsics);
    lagrangeOpt.init(depthOpt.depth_opt);
   
    int it = 0;
    while(true)
    {   
        thetaOpt.step(image, mask, depthOpt.depth_opt, lighting, albedo, lagrangeOpt.dual_opt, intrinsics, lagrangeOpt.kappa);
        depthOpt.step(depth_lr_upsampled, mask, thetaOpt.theta_opt, lagrangeOpt.dual_opt, lagrangeOpt.kappa);

        bool converged = lagrangeOpt.step(mask, thetaOpt.theta_opt, depthOpt.depth_opt);
        if(converged || it > 100)
            break;
        
        it++;
    }

    // Error should be lower than that of input
    image::EvaluationBase e(depth_star, depthOpt.depth_opt, mask, intrinsics);
    image::EvaluationBase e_comp(depth_star, depth_corrupt, mask, intrinsics);

    EXPECT_LT(e.meanError(), e_comp.meanError());
    EXPECT_LT(e.rmsError(), e_comp.rmsError());
}


TEST_F(CombinedTest, theta_depth_albedo_update_admm)
{
    Image<uchar> mask_lr = mask.resized(0.5, mask);
    depth_lr.resize(0.5, mask);
    auto depth_lr_upsampled = depth_lr.resized(2.0, mask_lr);

    Image<float3> shading_opt(shading);
    
    depthOpt.init(depth_smooth, mask, depth_lr, mask_lr, mu);
    thetaOpt.init(depthOpt.depth_opt, mask, depth_lr, mask_lr, intrinsics);
    albedoOpt.init(image, mask, shading_opt);
    lagrangeOpt.init(depthOpt.depth_opt);

    int it = 0;
    while(true)
    {   
        albedoOpt.step(shading_opt);
        thetaOpt.step(image, mask, depthOpt.depth_opt, lighting, albedoOpt.albedo_opt, lagrangeOpt.dual_opt, intrinsics, lagrangeOpt.kappa);
        depthOpt.step(depth_lr_upsampled, mask, thetaOpt.theta_opt, lagrangeOpt.dual_opt, lagrangeOpt.kappa);
        

        bool converged = lagrangeOpt.step(mask, thetaOpt.theta_opt, depthOpt.depth_opt);
        if(converged || it > 100)
            break;

        // Update shading (normally during light update)
        cu_CalculateShading<float3>(shading_opt, thetaOpt.spherical_harmonics_opt, lighting, mask);

        it++;
    }

    depthOpt.depth_opt.show<cuimage::DEPTH_TYPE>("Depth");

    // Error should be lower than that of input
    image::EvaluationBase e(depth_star, depthOpt.depth_opt, mask, intrinsics);
    image::EvaluationBase e_comp(depth_star, depth_smooth, mask, intrinsics);
    
    //depthOpt.depth_opt.show<cuimage::DEPTH_TYPE>("Depth");
    EXPECT_LT(e.meanError(), e_comp.meanError());
    EXPECT_LT(e.rmsError(), e_comp.rmsError());
}

TEST_F(CombinedTest, theta_depth_light_albedo_update_admm)
{
    Image<uchar> mask_lr = mask.resized(0.5, mask);
    depth_lr.resize(0.5, mask);
    auto depth_lr_upsampled = depth_lr.resized(2.0, mask_lr);

    depthOpt.init(depth_smooth, mask, depth_lr, mask_lr, mu);
    thetaOpt.init(depthOpt.depth_opt, mask, depth_lr, mask_lr, intrinsics);
    lightOpt.init(image, mask, thetaOpt.spherical_harmonics_opt);
    albedoOpt.init(image, mask, lightOpt.shading_opt);
    lagrangeOpt.init(depthOpt.depth_opt);

    int it = 0;
    while(true)
    {   
        albedoOpt.step(lightOpt.shading_opt);
        lightOpt.step(mask, thetaOpt.spherical_harmonics_opt, albedoOpt.albedo_opt);
        thetaOpt.step(image, mask, depthOpt.depth_opt, lightOpt.lighting_opt, albedoOpt.albedo_opt, lagrangeOpt.dual_opt, intrinsics, lagrangeOpt.kappa);
        depthOpt.step(depth_lr_upsampled, mask, thetaOpt.theta_opt, lagrangeOpt.dual_opt, lagrangeOpt.kappa);        
      
        bool converged = lagrangeOpt.step(mask, thetaOpt.theta_opt, depthOpt.depth_opt);
        if(converged || it > 100)
            break;
        
        it++;
    }

    // Error should be lower than that of input
    image::EvaluationBase e(depth_star, depthOpt.depth_opt, mask, intrinsics);
    image::EvaluationBase e_comp(depth_star, depth_smooth, mask, intrinsics);
    
    //depthOpt.depth_opt.show<cuimage::DEPTH_TYPE>("Depth");
    EXPECT_LT(e.meanError(), e_comp.meanError());
    EXPECT_LT(e.rmsError(), e_comp.rmsError());
}

#endif