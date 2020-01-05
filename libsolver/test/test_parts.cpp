#include <gtest/gtest.h>
#include <chrono>

#if TEST_PARTS

#include "solver/solver/solver_opt.h"
#include "image/processing.h"
#include "image/data.h"
#include "image/evaluation.h"

using namespace testing;

class PartialTests : public ::testing::Test
{
public:
    PartialTests()
     : d(std::string(SOURCE_DIR) + "/data/", 320, 240, 1),
       mask(d.mask),
       image(d.image),
       albedo(d.albedo),
       shading(d.shading),
       depth_star(d.depth_star),
       lighting(d.light),
       dual(d.image.width(), d.image.height())
    {
        image::DepthProcessing p(depth_star);
        p.harmonics(spherical_harmonics, d.K_hr);
        spherical_harmonics.mask(mask);    

        p.addNoise(depth_corrupt, 0, 0.001);
        depth_corrupt.mask(mask);

        p.theta(theta);
        theta.mask(mask);

        image::DepthProcessing pc(depth_corrupt);
        pc.bilateral(depth_smooth, mask, 0.05 * depth_star.width(), 50, 50);
        depth_corrupt.mask(mask);

        pc.theta(theta_corrupt);
        theta_corrupt.mask(mask);

        image::DepthProcessing ps(depth_smooth);
        ps.theta(theta_smooth);
        theta_smooth.mask(mask);

        intrinsics = d.K_hr;
    }

    image::DataSet d;
    const std::string optFilePath = std::string(SOURCE_DIR) + "/libsolver/test/opt/";

    IntrinsicsPtr intrinsics;

    Image<float3> theta, theta_corrupt, theta_smooth;
    Image<float> depth_corrupt, depth_smooth;
    Image<float4> spherical_harmonics;
    Image<float3> dual;

    const Image<float3>& lighting;
    const Image<float3>& albedo;
    const Image<float3>& image;
    const Image<float>& depth_star;
    const Image<float3>& shading;
    const Image<uchar>& mask;
};

class MinimalSolver : public solver::SolverBase
{
public:
    MinimalSolver(const std::string file, int w, int h, int nI, int lI, 
                  const std::string type = "gaussNewtonGPU")
    {
        solverParameters_.add("nIterations", nI);
        solverParameters_.add("lIterations", lI); 

        std::vector<int> dimensions = { w, h };

        solver_ = std::make_shared<solver::OptSolver>(dimensions, file, type, false, 0, 0);
    }

    ~MinimalSolver()
    {
    }

    template <typename T>
    void addParam(const std::string s, const Image<T>& img, const int i)
    {
        problemParameters_.add(s, img, i);
    }

    void addParam(const std::string s, const float f, const int i)
    {
        problemParameters_.add(s, f, i);
    }

    void clearParams()
    {
        problemParameters_.clear();
    }

    virtual void initialize() override {}
    virtual void finalize() override {}

    virtual void preSolve() override {}
    virtual void postSolve() override {}     

    virtual void preNonlinearSolve(int iteration) override {}
    virtual void postNonlinearSolve(int iteration) override {}
};

TEST_F(PartialTests, correct_theta_from_depth)
{
    Image<float3> theta_calc(theta.width(), theta.height());
    MinimalSolver solver(optFilePath + "gradient_theta.t", mask.width(), mask.height(), 1, 5);
    solver.addParam("Theta", theta_calc, 0);
    solver.addParam("Mask", mask, 1);
    solver.addParam("Depth", depth_star, 2);

    auto cost = solver.run();
    theta_calc.mask(mask);

    // Compare with theta calculated from depth
    EXPECT_NEAR(0.0, cost, 1e-9);    
    EXPECT_NEAR(0.0, (theta_calc - theta).norm2(), 0.1);  
}

TEST_F(PartialTests, correct_normals_from_theta)
{
    image::DepthProcessing p(depth_star);
    Image<float3> normals;
    p.normals(normals, intrinsics);
    normals.mask(mask);

    Image<float3> normals_calc(normals.width(), normals.height());

    MinimalSolver solver(optFilePath + "normals.t", mask.width(), mask.height(), 1, 5);
    solver.addParam("Normals", normals_calc, 0);
    solver.addParam("Mask", mask, 1);
    solver.addParam("Theta", theta, 2);

    solver.addParam("fx", d.fx, 3);
    solver.addParam("fy", d.fy, 4);
    solver.addParam("cx", d.cx, 5);
    solver.addParam("cy", d.cy, 6);

    auto cost = solver.run();

    EXPECT_NEAR(0.0, cost, 1e-9);    
    EXPECT_NEAR(0.0, (normals_calc - normals).norm2(), 0.1);  
}


TEST_F(PartialTests, correct_shading_from_normals_and_light)
{
    image::DepthProcessing p(depth_star);
    Image<float3> normals;
    p.normals(normals, intrinsics);
    normals.mask(mask);

    Image<float3> shading_calc(mask.width(), mask.height());

    MinimalSolver solver(optFilePath + "shading.t", mask.width(), mask.height(), 1, 5);

    solver.addParam("Shading", shading_calc, 0);
    solver.addParam("Mask", mask, 1);
    solver.addParam("Normals", normals, 2);

    solver.addParam("L_0", d.l1, 3);
    solver.addParam("L_1", d.l2, 4);
    solver.addParam("L_2", d.l3, 5);
    solver.addParam("L_3", d.l4, 6);

    auto cost = solver.run();
    shading_calc.mask(mask);

    // Compare with theta calculated from depth
    EXPECT_NEAR(0.0, cost, 1e-9);        
    EXPECT_NEAR(0.0, (shading_calc - shading).norm2(), 2.0);
}


TEST_F(PartialTests, correct_image_from_albedo_and_shading)
{
    Image<float3> image_calc(image.width(), image.height());

    MinimalSolver solver(optFilePath + "image.t", mask.width(), mask.height(), 1, 5);

    solver.addParam("Image", image_calc, 0);
    solver.addParam("Mask", mask, 1);
    solver.addParam("Shading", shading, 2);
    solver.addParam("Albedo", albedo, 3);

    auto cost = solver.run();
    image_calc.mask(mask);

    // Compare with theta calculated from depth
    EXPECT_NEAR(0.0, cost, 1e-9);    
    EXPECT_NEAR(0.0, (image_calc - image).norm2(), 0.1);
}

TEST_F(PartialTests, correct_image_from_theta)
{
    Image<float3> image_calc(image);
    MinimalSolver solver(optFilePath + "theta_to_image.t", mask.width(), mask.height(), 1, 5);

    solver.addParam("Image", image_calc, 0);
    solver.addParam("Mask", mask, 1);
    solver.addParam("Theta", theta, 2);
    solver.addParam("Albedo", albedo, 3);

    solver.addParam("L_0", d.l1, 4);
    solver.addParam("L_1", d.l2, 5);
    solver.addParam("L_2", d.l3, 6);
    solver.addParam("L_3", d.l4, 7);

    solver.addParam("fx", d.fx, 8);
    solver.addParam("fy", d.fy, 9);
    solver.addParam("cx", d.cx, 10);
    solver.addParam("cy", d.cy, 11);

    auto cost = solver.run();

    // Compare with theta calculated from depth
    EXPECT_NEAR(0.0, cost, 1e-9);    
    EXPECT_NEAR(0.0, (image_calc - image).norm2(), 1.0);  
    image_calc.show<cuimage::COLOR_TYPE_RGB_F>("Image");
}

TEST_F(PartialTests, perfect_theta_doesnt_change_by_shape_from_shading)
{
    MinimalSolver solver(optFilePath + "shape_from_shading.t", mask.width(), mask.height(), 1, 10);

    Image<float3> theta_calc(theta);

    solver.addParam("Theta", theta_calc, 0);
    solver.addParam("Mask", mask, 1);
    solver.addParam("Image", image, 2);
    solver.addParam("Albedo", albedo, 3);

    solver.addParam("L_0", d.l1, 4);
    solver.addParam("L_1", d.l2, 5);
    solver.addParam("L_2", d.l3, 6);
    solver.addParam("L_3", d.l4, 7);

    solver.addParam("fx", d.fx, 8);
    solver.addParam("fy", d.fy, 9);
    solver.addParam("cx", d.cx, 10);
    solver.addParam("cy", d.cy, 11);

    auto cost = solver.run();

    EXPECT_NEAR(0.0, cost, 1e-9);
    EXPECT_NEAR(0.0, (theta_calc - theta).norm2(), 0.1);  
}

TEST_F(PartialTests, theta_by_shape_from_shading)
{
    MinimalSolver solver(optFilePath + "shape_from_shading.t", mask.width(), mask.height(), 5, 3);

    Image<float3> theta_calc(theta_corrupt);

    solver.addParam("Theta", theta_calc, 0);
    solver.addParam("Mask", mask, 1);
    solver.addParam("Image", image, 2);
    solver.addParam("Albedo", albedo, 3);

    solver.addParam("L_0", d.l1, 4);
    solver.addParam("L_1", d.l2, 5);
    solver.addParam("L_2", d.l3, 6);
    solver.addParam("L_3", d.l4, 7);

    solver.addParam("fx", d.fx, 8);
    solver.addParam("fy", d.fy, 9);
    solver.addParam("cx", d.cx, 10);
    solver.addParam("cy", d.cy, 11);

    auto cost0 = solver.run();
    auto cost1 = solver.run();

    // Theta should be moving towards optimum
    EXPECT_LT(cost1, cost0);
    EXPECT_LT((theta_calc - theta).norm2(), (theta_corrupt - theta).norm2());
}


TEST_F(PartialTests, surface_prior_produces_expected_result)
{
    Image<float3> theta_calc(theta_corrupt);
    Image<float> weights(theta_calc.width(), theta_calc.height(), 0.f);

    MinimalSolver solver(optFilePath + "minimal_surface_value.t", mask.width(), mask.height(), 1, 10, "gaussNewtonGPU");

    solver.addParam("Weights", weights, 0);
    solver.addParam("Theta", theta_calc, 1);
    solver.addParam("Mask", mask, 2);    

    solver.addParam("fx", d.fx, 3);
    solver.addParam("fy", d.fy, 4);
    solver.addParam("cx", d.cx, 5);
    solver.addParam("cy", d.cy, 6);

    auto cost = solver.run();

    // Calculate weights on CPU
    float w[theta_calc.size()];
    for (int y = 0; y < theta_calc.height(); y++)
    {
    for (int x = 0; x < theta_calc.width(); x++)
    {
        float3 v = theta_calc.at(x, y);

        float3 n;
        n.x = v.y * d.fx;
        n.y = v.z * d.fy;
        n.z = -v.x - v.y * (x - d.cx) - v.z * (y - d.cy);

        float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
        float out = fabsf(v.x * len) / (d.fx * d.fy);

        if (mask.at(x, y) > 0)
            w[y * theta_calc.width() + x] = out;
        else
            w[y * theta_calc.width() + x] = 0.f;
    }   
    }

    Image<float> weights_comp;
    weights_comp.upload(w, theta_calc.width(), theta_calc.height());

    EXPECT_NEAR(0.0, (weights_comp - weights).norm2(), 0.1); 
}


TEST_F(PartialTests, minimal_surface_smoothes_theta)
{
    Image<float3> theta_calc(theta_corrupt);

    MinimalSolver solver(optFilePath + "minimal_surface.t", mask.width(), mask.height(), 4, 2, "gaussNewtonGPU");

    solver.addParam("Theta", theta_calc, 0);
    solver.addParam("Mask", mask, 1);    

    solver.addParam("fx", d.fx, 2);
    solver.addParam("fy", d.fy, 3);
    solver.addParam("cx", d.cx, 4);
    solver.addParam("cy", d.cy, 5);

    auto cost0 = solver.run();
    auto cost1 = solver.run();
    
    // Theta should be moving towards optimum
    EXPECT_GT(cost0, cost1);
}

TEST_F(PartialTests, shape_from_shading_from_theta)
{
    Image<float3> theta_calc(theta_smooth);

    MinimalSolver solver(optFilePath + "../../src/opt/updateTheta.t", mask.width(), mask.height(), 20, 7, "gaussNewtonGPU");

    float nu = 1e-6f;
    float kappa = 1e-2f;
    float gamma = 1.f;

    solver.addParam("Theta", theta_calc, 0);
    solver.addParam("Mask", mask, 1);    
    solver.addParam("Albedo", albedo, 2);
    solver.addParam("Depth", depth_smooth, 3);
    solver.addParam("Image", image, 4);
    solver.addParam("Dual", dual, 5);

    solver.addParam("L_0", d.l1, 6);
    solver.addParam("L_1", d.l2, 7);
    solver.addParam("L_2", d.l3, 8);
    solver.addParam("L_3", d.l4, 9);

    solver.addParam("fx", d.fx, 10);
    solver.addParam("fy", d.fy, 11);
    solver.addParam("cx", d.cx, 12);
    solver.addParam("cy", d.cy, 13);
    
    solver.addParam("nu", nu, 14);
    solver.addParam("kappa", kappa, 15);
    solver.addParam("gamma", gamma, 16);

    float cost = solver.run();

    // Expect that result got better (= closer to t_star)
    EXPECT_LT((theta_calc - theta).norm2(),  (theta_smooth - theta).norm2());
}

TEST_F(PartialTests, theta_in_admm_scheme_without_surface_reg)
{
    float kappa = 1e-4f;
    float tau = 1.5;

    float gamma = 1.f;
    float nu = 0.f;

    Image<float3> theta_calc(theta_smooth);
    Image<float> depth(depth_star);

    MinimalSolver solver(optFilePath + "../../src/opt/updateTheta.t", mask.width(), mask.height(), 1, 3, "gaussNewtonGPU");

    int i = 0;
    while(true)
    {
        solver.clearParams();
        solver.addParam("Theta", theta_calc, 0);
        solver.addParam("Mask", mask, 1);    
        solver.addParam("Albedo", albedo, 2);
        solver.addParam("Depth", depth, 3);
        solver.addParam("Image", image, 4);
        solver.addParam("Dual", dual, 5);

        solver.addParam("L_0", d.l1, 6);
        solver.addParam("L_1", d.l2, 7);
        solver.addParam("L_2", d.l3, 8);
        solver.addParam("L_3", d.l4, 9);

        solver.addParam("fx", d.fx, 10);
        solver.addParam("fy", d.fy, 11);
        solver.addParam("cx", d.cx, 12);
        solver.addParam("cy", d.cy, 13);
        
        solver.addParam("nu", nu, 14);
        solver.addParam("kappa", kappa, 15);
        solver.addParam("gamma", gamma, 16);

        // Theta update with (z, \grad z) from perfect z*
        float cost = solver.run();
        
        Image<float3> t_diff = theta - theta_calc; // theta_z is theta_star

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
        
        i++;  
    }
   
    // Expect that result is fully converged towards optimal depth
    Image<float> depth_calc = theta_calc.get<float>(0);
    EXPECT_NEAR(0.0, (depth - depth_calc).norm2(), 1e-3);
}

TEST_F(PartialTests, theta_in_admm_scheme_with_surface_reg)
{
    float kappa = 1e-4f;
    float tau = 5.f;

    float gamma = 1.f;
    float nu = 100.f;
   
    Image<float3> theta_calc(theta_corrupt);
    Image<float> depth(depth_star);

    MinimalSolver solver(optFilePath + "../../src/opt/updateTheta.t", mask.width(), mask.height(), 6, 3, "gaussNewtonGPU");

    int i = 0;
    while(true)
    {
        solver.clearParams();
        solver.addParam("Theta", theta_calc, 0);
        solver.addParam("Mask", mask, 1);    
        solver.addParam("Albedo", albedo, 2);
        solver.addParam("Depth", depth, 3);
        solver.addParam("Image", image, 4);
        solver.addParam("Dual", dual, 5);

        solver.addParam("L_0", d.l1, 6);
        solver.addParam("L_1", d.l2, 7);
        solver.addParam("L_2", d.l3, 8);
        solver.addParam("L_3", d.l4, 9);

        solver.addParam("fx", d.fx, 10);
        solver.addParam("fy", d.fy, 11);
        solver.addParam("cx", d.cx, 12);
        solver.addParam("cy", d.cy, 13);
        
        solver.addParam("nu", nu, 14);
        solver.addParam("kappa", kappa, 15);
        solver.addParam("gamma", gamma, 16);

        // Theta update with (z, \grad z) from perfect z*
        float cost = solver.run();
        
        Image<float3> t_diff = theta - theta_calc; // theta_z is theta_star

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
        
        i++;  
    }
   
    // Expect that result is fully converged towards optimal depth
    Image<float> depth_calc = theta_calc.get<float>(0);
    EXPECT_NEAR(0.0, (depth - depth_calc).norm2(), 1e-3);
}

#endif