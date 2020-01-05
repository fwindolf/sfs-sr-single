/**
 * @file   optimizer.h
 * @brief  Optimizer for Shape from shading with super resolution
 * @author Florian Windolf
 */
#ifndef SFS_OPTIMIZER_H
#define SFS_OPTIMIZER_H

#include "core/image.h"
#include "parameters.h"
#include "solver/optimizer.h"

/**
 * @class SfsOptimizer
 * @brief Runs all optimizers in ADMM scheme
 */
class SfSOptimizer
{
public:
    SfSOptimizer(const Image<float3>& image, const Image<uchar>& mask,
        const Image<float>& depth_lr, const Image<float>& depth,
        const IntrinsicsPtr intrinsics,
        const SfsParameters params = SfsParameters(),
        const int thetaItOuter = 1, const int thetaItInner = 5,
        const int depthItOuter = 1, const int depthItInner = 3, 
        const bool filterDepth = true);

    ~SfSOptimizer();

    bool init();

    void useAlbedo(const Image<float3>& albedo_star);

    void useLight(const Image<float3>& light_star);

    bool run(bool verbose = true);

    void evaluate(const Image<float>& depth_star,
        const std::string filePath = "output/",
        const std::string run = "") const;

    void save(const std::string filePath = "output/",
        const std::string run = "") const;

    void visualize();

private:
    const int width_, height_;

    const SfsParameters params_;

    const Image<float3>& image_;
    const Image<float>& depth_orig_;
    const Image<uchar>& mask_orig_;

    Image<float> depth_, depth_lr_, depth_lr_upsampled_;
    Image<uchar> mask_lr_, mask_;

    bool optimize_albedo_ = true;
    bool optimize_light_ = true;
    bool filter_depth_ = true;

    IntrinsicsPtr intrinsics_;

    solver::AlbedoOptimizer albedoOpt_;
    solver::LightingOptimizer lightOpt_;
    solver::AuxiliaryOptimizer thetaOpt_;
    solver::DepthOptimizer depthOpt_;
    solver::LagrangeOptimizer lagrangeOpt_;
};

#endif // SFS_OPTIMIZER_H