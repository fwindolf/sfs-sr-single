#include "solver/optimizer/auxiliary.h"

#include "image/processing.h"
#include "solver/optimizer/theta_cu.h"

using namespace solver;

AuxiliarySolver::AuxiliarySolver(const int w, const int h, const float gamma,
    const float nu, const int nIterations, const int lIterations)
    : w_(w)
    , h_(h)
    , gamma_(gamma)
    , nu_(nu)
{
    std::cout << "Thetaoptimizer (" << w << "x" << h
              << ") with gamma=" << gamma << " nu=" << nu
              << " it=" << nIterations << "/" << lIterations << std::endl;

    solverParameters_.add("nIterations", nIterations);
    solverParameters_.add("lIterations", lIterations);

    std::vector<int> dimensions = {w_, h_};

    const std::string file
        = std::string(SOURCE_DIR) + "/libsolver/src/opt/updateTheta.t";

    solver_ = std::make_shared<solver::OptSolver>(
        dimensions, file, "gaussNewtonGPU", false, 0, 0);
}

void AuxiliarySolver::update(Image<float3>& theta, const Image<float3>& image,
    const Image<uchar>& mask, const Image<float>& depth,
    const Image<float3>& lighting, const Image<float3>& albedo,
    const Image<float3>& dual, const IntrinsicsPtr intrinsics,
    const float kappa)
{
    assert(!theta.empty());
    assert(!depth.empty());
    assert(!lighting.empty() && lighting.height() >= 4);
    assert(!albedo.empty());
    assert(!dual.empty());

    // Download lighting
    float3* l = lighting.download(4);
    lighting_[0] = l[0].x;
    lighting_[1] = l[1].x;
    lighting_[2] = l[2].x;
    lighting_[3] = l[3].x;

    // Extract intrinsics
    fx_ = intrinsics->fx();
    fy_ = intrinsics->fy();
    cx_ = intrinsics->cx();
    cy_ = intrinsics->cy();

    assert(theta.width() == w_ && theta.height() == h_);
    assert(albedo.width() == w_ && albedo.height() == h_);
    assert(depth.width() == w_ && depth.height() == h_);
    assert(image.width() == w_ && image.height() == h_);
    assert(dual.width() == w_ && dual.height() == h_);

    problemParameters_.clear();
    problemParameters_.add("Theta", theta, 0);
    problemParameters_.add("Mask", mask, 1);
    problemParameters_.add("Albedo", albedo, 2);
    problemParameters_.add("Depth", depth, 3);
    problemParameters_.add("Image", image, 4);
    problemParameters_.add("Dual", dual, 5);

    problemParameters_.add("L_1", lighting_[0], 6);
    problemParameters_.add("L_2", lighting_[1], 7);
    problemParameters_.add("L_3", lighting_[2], 8);
    problemParameters_.add("L_4", lighting_[3], 9);

    problemParameters_.add("fx", fx_, 10);
    problemParameters_.add("fy", fy_, 11);
    problemParameters_.add("cx", cx_, 12);
    problemParameters_.add("cy", cy_, 13);

    problemParameters_.add("nu", nu_, 14);
    problemParameters_.add("kappa", kappa, 15);
    problemParameters_.add("gamma", gamma_, 16);
}

void AuxiliarySolver::normalizeNu(
    const int nnz, const float mean, const float factor)
{
    nu_ = nu_ / ((float)nnz * fabsf(powf(mean / factor, 2))) * (float)nnz;
}

void AuxiliarySolver::initialize() {}
void AuxiliarySolver::finalize() {}

void AuxiliarySolver::preSolve() {}
void AuxiliarySolver::postSolve() {}

void AuxiliarySolver::preNonlinearSolve(int iteration) {}
void AuxiliarySolver::postNonlinearSolve(int iteration) {}

AuxiliaryOptimizer::AuxiliaryOptimizer(const int w, const int h,
    const float gamma, const float nu, const int nIterations,
    const int lIterations)
    : w_(w)
    , h_(h)
    , solver_(w, h, gamma, nu, nIterations, lIterations)
{
}

AuxiliaryOptimizer::~AuxiliaryOptimizer() {}

void AuxiliaryOptimizer::init(const Image<float>& depth,
    const Image<uchar>& mask, const Image<float>& depth_lr,
    const Image<uchar>& mask_lr, const IntrinsicsPtr intrinsics,
    const float normFactor)
{
    assert(depth.width() == mask.width());
    assert(depth.height() == mask.height());

    assert(depth_lr.width() == mask_lr.width());
    assert(depth_lr.height() == mask_lr.height());

    float nnz = mask_lr.nonzero();
    float mean_lr = depth_lr.mean();

    solver_.normalizeNu(nnz, mean_lr, normFactor);

    // Initialize theta and spherical harmonics from depth
    image::DepthProcessing p(depth);
    p.theta(theta_opt);

    /*
    auto d = theta_opt.get<float>(0);
    auto dx = theta_opt.get<float>(1);
    auto dy = theta_opt.get<float>(2);

    d.show<cuimage::DEPTH_TYPE>("Theta Depth");
    dx.show<cuimage::COLOR_TYPE_GREY_F>("Theta DX");
    dy.show<cuimage::COLOR_TYPE_GREY_F>("Theta DY");
    */
    theta_opt.mask(mask);

    p.harmonics(spherical_harmonics_opt, intrinsics, 1);
    spherical_harmonics_opt.mask(mask);
}

void AuxiliaryOptimizer::step(const Image<float3>& image,
    const Image<uchar>& mask, const Image<float>& depth,
    const Image<float3>& lighting, const Image<float3>& albedo,
    const Image<float3>& dual, const IntrinsicsPtr intrinsics,
    const float kappa)
{
    assert(!depth.empty());
    assert(!lighting.empty());
    assert(!albedo.empty());
    assert(!dual.empty());

    solver_.update(theta_opt, image, mask, depth, lighting, albedo, dual,
        intrinsics, kappa);
    auto cost = solver_.run();

    // Rebuild spherical harmonics
    cu_ThetaToHarmonics(spherical_harmonics_opt, theta_opt, intrinsics->fx(),
        intrinsics->fy(), intrinsics->cx(), intrinsics->cy());
}