#include "optimizer.h"

#include "image/evaluation.h"
#include "image/processing.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace solver;
using namespace core;
using namespace image;

SfSOptimizer::SfSOptimizer(const Image<float3>& image,
    const Image<uchar>& mask, const Image<float>& depth_lr,
    const Image<float>& depth, const IntrinsicsPtr intrinsics,
    const SfsParameters params, const int thetaItOuter, const int thetaItInner,
    const int depthItOuter, const int depthItInner, const bool filterDepth)
    : width_(image.width())
    , height_(image.height())
    , image_(image)
    , mask_orig_(mask)
    , depth_lr_(depth_lr)
    , depth_orig_(depth)
    , intrinsics_(intrinsics)
    , albedoOpt_(width_, height_, params.alpha, params.lambda, params.maxIter)
    , lightOpt_(width_, height_, 1)
    , thetaOpt_(
          width_, height_, params.gamma, params.nu, thetaItOuter, thetaItInner)
    , depthOpt_(width_, height_, params.mu, depthItOuter, depthItInner)
    , lagrangeOpt_(width_, height_, params.tau, params.kappa, params.tolerance,
          params.toleranceEL)
    , filter_depth_(filterDepth)
{
    assert(!image_.empty());
    assert(!mask_orig_.empty());
    assert(!depth_lr_.empty());
}

SfSOptimizer::~SfSOptimizer() {}

bool SfSOptimizer::init()
{
    // Generate mask on different resolutions
    std::cout << "SFS Optimizer "
              << "image(" << image_.width() << "x" << image_.height() << ") "
              << "mask(" << mask_orig_.width() << "x" << mask_orig_.height()
              << ") "
              << "depth(" << depth_orig_.width() << "x" << depth_orig_.height()
              << ") "
              << "depth_lr (" << depth_lr_.width() << "x" << depth_lr_.height()
              << ") " << std::endl;

    // Clean data
    mask_lr_.copyFrom(mask_orig_);
    mask_lr_.resize(depth_lr_.width(), depth_lr_.height(), mask_orig_);

    mask_.copyFrom(mask_orig_);
    mask_.resize(image_.width(), image_.height(), mask_orig_);

    // Remove NaNs in low resolution depth
    InpaintFilter f_lr(mask_lr_);
    f_lr.removeNans(depth_lr_);
    depth_lr_.mask(mask_lr_);

    depth_lr_upsampled_ = depth_lr_.resized(
        image_.width(), image_.height(), mask_lr_, cuimage::LINEAR_NONZERO);

    // For high resolution depth use upsampled original or provided depth
    if (depth_orig_.empty())
    {
        std::cout << "Using upsampled lr depth for superresolution"
                  << std::endl;
        depth_.copyFrom(depth_lr_);
        depth_.resize(image_.width(), image_.height(), mask_lr_,
            ResizeMode::LINEAR_NONZERO);
    }
    else
    {
        std::cout << "Using provided depth for superresolution" << std::endl;
        assert(depth_orig_.width() == image_.width());
        assert(depth_orig_.height() == image_.height());
        depth_.copyFrom(depth_orig_);
    }

    // Smooth inital depth
    if (filter_depth_)
    {
        DepthProcessing p(depth_);
        p.bilateral(depth_, mask_, 0.01 * depth_.width(), 10.f, 10.f);
    }

    InpaintFilter f(mask_);
    f.removeNans(depth_);
    depth_.mask(mask_);

    f.removeNans(depth_lr_upsampled_);
    depth_lr_upsampled_.mask(mask_);

    // depth_lr_.show<cuimage::DEPTH_TYPE>("Depth LR");
    // depth_.show<cuimage::DEPTH_TYPE>("Depth Init");

    assert(depth_.nan() == 0);
    assert(depth_lr_.nan() == 0);

    depthOpt_.init(depth_, mask_, depth_lr_, mask_lr_, 8.8314f);
    thetaOpt_.init(
        depthOpt_.depth_opt, mask_, depth_lr_, mask_lr_, intrinsics_, 8.8045f);
    lightOpt_.init(image_, mask_, thetaOpt_.spherical_harmonics_opt);
    albedoOpt_.init(image_, mask_, lightOpt_.shading_opt);
    lagrangeOpt_.init(depthOpt_.depth_opt);

    return true;
}

void SfSOptimizer::useAlbedo(const Image<float3>& albedo_star)
{
    assert(albedo_star.width() == width_);
    assert(albedo_star.height() == height_);

    optimize_albedo_ = false;

    albedoOpt_.albedo_opt.copyFrom(albedo_star);
}

void SfSOptimizer::useLight(const Image<float3>& light_star)
{
    // optimize_light_ = false;

    lightOpt_.lighting_opt.copyFrom(light_star);
}

void print_it(const int iteration, const float kappa, const int w)
{
    std::cout << "| " << std::setw(w) << iteration << " | " << std::setw(12)
              << kappa;
}

void print_time(
    const std::chrono::time_point<std::chrono::high_resolution_clock>& t_from,
    const std::chrono::time_point<std::chrono::high_resolution_clock>& t_to,
    const int w)
{
    using namespace std::chrono;
    auto t_ms = duration_cast<milliseconds>(t_to - t_from).count();
    std::cout << " | " << std::setw(w) << t_ms << " ms";
}

bool SfSOptimizer::run(bool verbose)
{
    if (verbose)
    {
        std::cout << "Starting optimization" << std::endl
                  << "| " << std::setw(3) << "Iter "
                  << "| " << std::setw(13) << "Kappa "
                  << "| " << std::setw(9) << "Albedo "
                  << "| " << std::setw(7) << "Light "
                  << "| " << std::setw(9) << "Theta "
                  << "| " << std::setw(8) << "Depth "
                  << "| " << std::setw(7) << "Lagrange "
                  << "| " << std::endl;
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    bool converged = false;
    int it = 0;

    // depthOpt_.depth_opt.createWindow<cuimage::DEPTH_TYPE>("Depth Opt");
    // depthOpt_.depth_opt.show();
    // depthOpt_.depth_opt.show<cuimage::DEPTH_TYPE>("Depth Opt");

    while (!converged)
    {
        auto t_it_start = std::chrono::high_resolution_clock::now();
        if (verbose)
            print_it(it, lagrangeOpt_.kappa, 4);

        if (optimize_albedo_)
            albedoOpt_.step(lightOpt_.shading_opt);

        auto t_albedo = std::chrono::high_resolution_clock::now();
        if (verbose)
            print_time(t_it_start, t_albedo, 5);

        if (optimize_light_)
            lightOpt_.step(mask_, thetaOpt_.spherical_harmonics_opt,
                albedoOpt_.albedo_opt);
        else
            lightOpt_.updateShading(mask_, thetaOpt_.spherical_harmonics_opt);

        auto t_light = std::chrono::high_resolution_clock::now();
        if (verbose)
            print_time(t_albedo, t_light, 3);

        thetaOpt_.step(image_, mask_, depthOpt_.depth_opt,
            lightOpt_.lighting_opt, albedoOpt_.albedo_opt,
            lagrangeOpt_.dual_opt, intrinsics_, lagrangeOpt_.kappa);
        auto t_theta = std::chrono::high_resolution_clock::now();
        if (verbose)
            print_time(t_light, t_theta, 5);

        depthOpt_.step(depth_lr_upsampled_, mask_, thetaOpt_.theta_opt,
            lagrangeOpt_.dual_opt, lagrangeOpt_.kappa);
        auto t_depth = std::chrono::high_resolution_clock::now();
        if (verbose)
            print_time(t_theta, t_depth, 4);

        converged = lagrangeOpt_.step(
            mask_, thetaOpt_.theta_opt, depthOpt_.depth_opt);
        auto t_it_end = std::chrono::high_resolution_clock::now();
        if (verbose)
            print_time(t_depth, t_it_end, 5);

        it++;
        std::cout << " | " << std::endl;

        // depthOpt_.depth_opt.show();
    }

    depthOpt_.depth_opt.mask(mask_);

    auto t_end = std::chrono::high_resolution_clock::now();
    auto s_final = std::chrono::duration_cast<std::chrono::milliseconds>(
                       t_end - t_start)
                       .count()
        / 1000.f;
    if (verbose)
    {
        std::cout << "|======================================================="
                     "==================|"
                  << std::endl;
        std::cout << "| " << std::setw(4) << it << " Iterations in " << s_final
                  << " s" << std::endl;
    }

    // depthOpt_.depth_opt.closeWindow(true);
    return true;
}

void SfSOptimizer::evaluate(const Image<float>& depth_star,
    const std::string filePath, const std::string run) const
{
    EvaluationBase eval(depth_star, depthOpt_.depth_opt, mask_, intrinsics_);
    float meanErr = eval.meanError();
    float medianErr = eval.medianError();
    float rmsErr = eval.rmsError();

    if (!filePath.empty())
    {
        std::string fileName = filePath + "/" + run + "/results.txt";
        std::fstream f_out;
        f_out.open(fileName, std::ios::out | std::ios::app);
        f_out << meanErr << ";" << medianErr << ";" << rmsErr << std::endl;
    }
    else
    {
        std::cout << "|======================================================="
                     "==================|"
                  << std::endl;
        std::cout << "| Mean Error = " << std::setw(8) << std::setprecision(6)
                  << meanErr << " | Median Error = " << std::setw(8)
                  << std::setprecision(6) << medianErr
                  << " | RMS Error = " << std::setw(8) << std::setprecision(6)
                  << rmsErr << " | " << std::endl;
    }
}

void SfSOptimizer::save(
    const std::string filePath, const std::string run) const
{

    std::string fileNameBase = filePath + "/" + run + "/";
    std::cout << "Saving to " << fileNameBase << "/*.png" << std::endl;

    auto depth = depthOpt_.depth_opt;
    depth.save(fileNameBase + "depth_refined.exr");

    DepthProcessing p(depth);
    Image<float3> normals;
    p.normals(normals, intrinsics_);
    normals = normals + make_float3(1.f, 1.f, 1.f);
    normals = normals / make_float3(2.f, 2.f, 2.f);
    normals.save(fileNameBase + "normals.png");

    albedoOpt_.albedo_opt.save(fileNameBase + "albedo.png");
    image_.save(fileNameBase + "image.png");
    lightOpt_.shading_opt.save(fileNameBase + "shading.png");

    float3* light = lightOpt_.lighting_opt.download();
    std::fstream fLight;
    fLight.open(fileNameBase + "/light.txt", std::ios::out);
    for (int h = 0; h < lightOpt_.lighting_opt.height(); h++)
        fLight << light[h].x << std::endl;

    fLight.close();

    depth_lr_.save(fileNameBase + "depth.exr");
}

void SfSOptimizer::visualize()
{
    depthOpt_.depth_opt.show<cuimage::DEPTH_TYPE>("Depth");
    depth_lr_.show<cuimage::DEPTH_TYPE>("Depth LR");
    albedoOpt_.albedo_opt.show<cuimage::COLOR_TYPE_RGB_F>("Albedo");
    depthOpt_.depth_opt.show<cuimage::DEPTH_TYPE>("Depth");
}
