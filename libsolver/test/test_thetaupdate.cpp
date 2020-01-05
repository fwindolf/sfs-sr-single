#include <gtest/gtest.h>
#include <chrono>

#if TEST_THETA

#include "solver/optimizer/auxiliary.h"

#include "image/processing.h"

#include "image/data.h"

using namespace solver;

class ThetaUpdateTest : public ::testing::Test
{
public:
    ThetaUpdateTest()
     : w_(640), h_(480), 
       d(std::string(SOURCE_DIR) + "/data/", w_, h_, 2),
       mask(d.mask),
       image(d.image),
       albedo(d.albedo),
       shading(d.shading),
       depth_star(d.depth_star),
       lighting(d.light),
       dual(image.width(), image.height())
    {
        image::DepthProcessing p(depth_star);
        p.harmonics(spherical_harmonics, d.K_hr);
        spherical_harmonics.mask(mask);    

        p.addNoise(depth_corrupt, 0, 0.001);
        depth_corrupt.mask(mask);

        p.theta(theta);
        theta.mask(mask);
        
        image::DepthProcessing pc(depth_corrupt);
        pc.blur(depth_smooth, mask, 0.01 * depth_star.width(), 1.f);
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
    Image<float> depth_corrupt, depth_smooth;
    Image<float4> spherical_harmonics;
    Image<float3> dual;
    

    IntrinsicsPtr intrinsics;
};

TEST_F(ThetaUpdateTest, surface_prior_produces_smooth_output)
{
    float gamma = 0.f;
    float kappa = 0.f;
    float nu = 1.f;    

    AuxiliaryOptimizer optimizer(w_, h_, gamma, nu, 20, 7);
    optimizer.init(depth_corrupt, mask, depth_star, mask, intrinsics);
    optimizer.step(image, mask, depth_corrupt, lighting, albedo, dual, intrinsics, kappa);

    // Output should be closer to optimal than theta
    EXPECT_GT((optimizer.theta_opt - theta).norm2(), (theta - theta_corrupt).norm2());
}

TEST_F(ThetaUpdateTest, auxiliary_pulls_corrupted_depth_to_original)
{
    float gamma = 0.0001f;
    float kappa = 1.f;
    float nu = 0.00001f;    

    AuxiliaryOptimizer optimizer(w_, h_, gamma, nu, 2, 7);
    optimizer.init(depth_corrupt, mask, depth_star, mask, intrinsics);
    optimizer.step(image, mask, depth_star, lighting, albedo, dual, intrinsics, kappa);

    // Cannot compare depth component directly, as that is not what this optimizer goes for
    // Instead expect that the spherical harmonics match (which is based on the shading information that 
    //  the auxiliary optimizer optimizes for)
    auto sh_opt = optimizer.spherical_harmonics_opt.asGray<float>();
    auto sh = spherical_harmonics.asGray<float>();
    EXPECT_NEAR(0.f, (sh_opt - sh).norm2(), 1.f);
}


TEST_F(ThetaUpdateTest, shading_sharpens_blurred_depth)
{
    Image<float4> sharm_in;
    image::DepthProcessing p(depth_smooth);
    p.harmonics(sharm_in, intrinsics);

    float gamma = 1.f;
    float kappa = 0.f;
    float nu = 0.f;    

    AuxiliaryOptimizer optimizer(w_, h_, gamma, nu, 10, 5);
    optimizer.init(depth_smooth, mask, depth_star, mask, intrinsics);
    optimizer.step(image, mask, depth_smooth, lighting, albedo, dual, intrinsics, kappa);

    // Output should be different from smooth depth
    Image<float> depth_out = optimizer.theta_opt.get<float>(0);
    EXPECT_GT((depth_out - depth_smooth).norm2(), 0.f);

    // Harmonics should be closer to optimal harmonics now
    Image<float> sharm_gr_in = sharm_in.asGray<float>();
    Image<float> sharm_gr_out = optimizer.spherical_harmonics_opt.asGray<float>();
    Image<float> sharm_gr_star = spherical_harmonics.asGray<float>();
    
    EXPECT_LT((sharm_gr_out - sharm_gr_star).norm2(), (sharm_gr_in - sharm_gr_star).norm2());
}

TEST_F(ThetaUpdateTest, works_in_admm_style_without_surface_regularizer)
{
    float kappa = 1e-4f;
    float tau = 1.5;

    float gamma = 1.f;
    float nu = 0.f;

    // Initial theta with from filtered noisy depth
    AuxiliaryOptimizer optimizer(w_, h_, gamma, nu, 2, 2);
    optimizer.init(depth_smooth, mask, depth_star, mask, intrinsics);

    Image<float3> theta_z;
    Image<float> last_depth(depth_star);

    while(true)
    {   
        // Theta update with (z, \grad z) from perfect z*
        optimizer.step(image, mask, depth_star, lighting, albedo, dual, intrinsics, kappa);

        Image<float3> t_diff = theta - optimizer.theta_opt; // theta_z is theta_star

        // u^(k+½) = u^(k) + thetaZ - theta
        dual += t_diff;

        // kappa^(k+1) = tau * kappa
        kappa *= tau;
        
        // u^(k+1) = u^(k+½) / tau
        dual /= make_float3(tau, tau, tau);
      
        // 0.5 * kappa * || theta-(z,zx,zy)||_2^2 + || u.*(theta-(z,zx,zy)) ||_1
        t_diff.mask(mask);

        auto t_diff_sq = t_diff * t_diff;    
        float norm2 = .5f * kappa * t_diff_sq.sum();

        t_diff *= dual;
        float norm1 = t_diff.norm1();

        float diffEL = norm1 + norm2;  

        if (fabsf(diffEL) < 1e-6)
            break;
    }   

    // Converged to optimal theta
    EXPECT_LT((optimizer.theta_opt - theta).norm2(), 1e-3);
}


TEST_F(ThetaUpdateTest, works_in_admm_style_with_surface_regularizer)
{
    float kappa = 1e-4f;
    float tau = 1.5;

    float gamma = 1.f;
    float nu = 2.f;

    // Initial theta with from filtered noisy depth
    AuxiliaryOptimizer optimizer(w_, h_, gamma, nu, 2, 2);
    optimizer.init(depth_smooth, mask, depth_star, mask, intrinsics);

    Image<float3> theta_z;
    Image<float> last_depth(depth_star);

    while(true)
    {   
        // Theta update with (z, \grad z) from perfect z*
        optimizer.step(image, mask, depth_star, lighting, albedo, dual, intrinsics, kappa);

        Image<float3> t_diff = theta - optimizer.theta_opt; // theta_z is theta_star

        // u^(k+½) = u^(k) + thetaZ - theta
        dual += t_diff;

        // kappa^(k+1) = tau * kappa
        kappa *= tau;
        
        // u^(k+1) = u^(k+½) / tau
        dual /= make_float3(tau, tau, tau);
      
        // 0.5 * kappa * || theta-(z,zx,zy)||_2^2 + || u.*(theta-(z,zx,zy)) ||_1
        t_diff.mask(mask);

        auto t_diff_sq = t_diff * t_diff;    
        float norm2 = .5f * kappa * t_diff_sq.sum();

        t_diff *= dual;
        float norm1 = t_diff.norm1();

        float diffEL = norm1 + norm2;  

        if (fabsf(diffEL) < 1e-6)
            break;
    }   

    // Converged to optimal theta
    EXPECT_LT((optimizer.theta_opt - theta).norm2(), 1e-3);
}

#endif
