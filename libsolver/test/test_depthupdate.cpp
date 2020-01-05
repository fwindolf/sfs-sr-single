#include <gtest/gtest.h>

#if TEST_DEPTH

#include "solver/optimizer/depth.h"
#include "image/processing.h"
#include "image/data.h"

using namespace solver;

class DepthUpdateTest : public ::testing::Test
{
public:
    DepthUpdateTest()
     : w_(640), h_(480),        
       d(std::string(SOURCE_DIR) + "/data/", w_, h_, 2),
       mask(d.mask),
       mask_lr(d.mask),
       image(d.image),
       depth_star(d.depth_star),
       depth_lr(d.depth_star),
       dual(image.width(), image.height())
    {
        image::DepthProcessing p(depth_star);
        p.addNoise(depth_corrupt, 0, 0.001);
        p.theta(theta);
        
        // Make depth_lr lose information
        mask_lr.resize(w_/2, h_/2, mask, cuimage::NEAREST);
        depth_lr.resize(w_/2, h_/2, mask, cuimage::NEAREST);
        depth_lr_upsampled = depth_lr.resized(w_, h_, mask_lr, cuimage::LINEAR_NONZERO);
    }    

    const int w_, h_;
    image::DataSet d;

    const Image<float3>& image;
    const Image<float>& depth_star;
    const Image<uchar>& mask;

    Image<float3> theta;
    Image<float> depth_lr,  depth_lr_upsampled, depth_corrupt;
    Image<float3> dual;
    Image<uchar> mask_lr;
};

TEST_F(DepthUpdateTest, auxiliary_energy_pulls_depth_towards_theta)
{
    float mu = 0.f;
    float kappa = 1.f;

    // Auxiliary update: min_Theta [Theta - ThetaZ + U]^2
    DepthOptimizer optimizer(w_, h_, mu, 1, 1);
    optimizer.init(depth_corrupt, mask, depth_lr, mask_lr, 1.f);

    optimizer.step(depth_lr_upsampled, mask, theta, dual, kappa);

    // Should be closer to depth from which theta was created then to corrupted (input) depth
    EXPECT_LT((optimizer.depth_opt - depth_star).norm2(), (optimizer.depth_opt - depth_corrupt).norm2());
}

TEST_F(DepthUpdateTest, fitting_energy_pulls_corrupted_to_upsampled_lr)
{
    float mu = 1.f;
    float kappa = 0.f;

    // Input perfect depth instead of low resolution
    DepthOptimizer optimizer(w_, h_, mu, 1, 1);
    optimizer.init(depth_corrupt, mask, depth_star, mask, 1.f);

    optimizer.step(depth_star, mask, theta, dual, kappa);

    // Should be closer to depth from which theta was created then to corrupted (input) depth
    EXPECT_LT((optimizer.depth_opt - depth_star).norm2(), (optimizer.depth_opt - depth_corrupt).norm2());
}


TEST_F(DepthUpdateTest, produces_perfect_output_on_perfect_targets)
{    
    float mu = 1e-2;
    float kappa = 1e-5;

    DepthOptimizer optimizer(w_, h_, mu, 10, 4);
    optimizer.init(depth_corrupt, mask, depth_star, mask, 1.f);

    optimizer.step(depth_star, mask, theta, dual, kappa);

    EXPECT_FALSE(optimizer.depth_opt.empty());
    
    // Should be the same as depth
    EXPECT_LT((optimizer.depth_opt - depth_star).norm2(), 1e-3f);
}


TEST_F(DepthUpdateTest, no_change_on_perfect_input)
{    
    float mu = 0.f;
    float kappa = 1.f;

    DepthOptimizer optimizer(w_, h_, mu, 1, 4);
    optimizer.init(depth_star, mask, depth_star, mask, 1.f);

    optimizer.step(depth_star, mask, theta, dual, kappa);

    EXPECT_FALSE(optimizer.depth_opt.empty());
    
    // Should be the same as depth
    EXPECT_LT((optimizer.depth_opt - depth_star).norm2(), 1e-3f);
}

TEST_F(DepthUpdateTest, depth_update_works_in_admm_scheme_with_optimal_theta)
{
    float kappa = 1e-4f;
    float tau = 1.5;

    float mu = 1e-2f;

    DepthOptimizer optimizer(w_, h_, mu, 1, 4);
    optimizer.init(depth_corrupt, mask, depth_lr, mask_lr, 1.f);

    Image<float3> theta_opt;
    while(true)
    {   
        optimizer.step(depth_lr_upsampled, mask, theta, dual, kappa);

        image::DepthProcessing p(optimizer.depth_opt);
        p.theta(theta_opt);

        Image<float3> t_diff = theta_opt - theta; // theta is theta_star

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

    EXPECT_LT((optimizer.depth_opt - depth_star).norm2(), 1e-3);
}

#endif