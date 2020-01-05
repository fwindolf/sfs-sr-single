#include "solver/optimizer/depth.h"

#include "image/processing.h"
#include "solver/optimizer/inpaint.h"

// Do not change
#define DEPTH_MODE 0

using namespace solver;
using namespace image;

DepthSolver::DepthSolver(const int w, const int h, const float mu,
    const int nIterations, const int lIterations)
    : w_(w)
    , h_(h)
    , mu_(mu)
{
    std::cout << "Depthoptimizer (" << w << "x" << h << ") with mu=" << mu
              << " it=" << nIterations << "/" << lIterations << std::endl;
              
    solverParameters_.add("nIterations", nIterations);
    solverParameters_.add("lIterations", lIterations);

    std::vector<int> dimensions = {w_, h_};

    const std::string file
        = std::string(SOURCE_DIR) + "/libsolver/src/opt/updateDepth.t";

    solver_ = std::make_shared<solver::OptSolver>(
        dimensions, file, "gaussNewtonGPU", false, 0, 0);
}

void DepthSolver::update(Image<float>& depth, const Image<float>& depth_lr,
    const Image<uchar>& mask, const Image<float3>& theta,
    const Image<float3>& dual, const float kappa)
{
    assert(!depth.empty());
    assert(!theta.empty());
    assert(!dual.empty());

    assert(depth_lr.width() == depth.width());
    assert(depth_lr.height() == depth.height());

    problemParameters_.clear();
    problemParameters_.add("Depth", depth, 0);
    problemParameters_.add("Mask", mask, 1);
    problemParameters_.add("DepthLr", depth_lr, 2);
    problemParameters_.add("Theta", theta, 3);
    problemParameters_.add("Dual", dual, 4);

    problemParameters_.add("mu", mu_, 5);
    problemParameters_.add("kappa", kappa, 6);
}

void DepthSolver::normalizeMu(
    const int nnz, const int nnz_lr, const float mean, const float factor)
{
    mu_ = mu_ / ((float)nnz_lr * powf(mean / factor, 2)) * (float)nnz;
}

void DepthSolver::initialize() {}

void DepthSolver::finalize() {}

void DepthSolver::preSolve() {}
void DepthSolver::postSolve() {}

void DepthSolver::preNonlinearSolve(int iteration) {}
void DepthSolver::postNonlinearSolve(int iteration) {}

DepthOptimizer::DepthOptimizer(const int w, const int h, const float mu,
    const int nIterations, const int lIterations)
    : w_(w)
    , h_(h)
    , solver_(w, h, mu, nIterations, lIterations)
{
}

DepthOptimizer::~DepthOptimizer() {}

void DepthOptimizer::init(const Image<float>& depth, const Image<uchar>& mask,
    const Image<float>& depth_lr, const Image<uchar>& mask_lr,
    const float normFactor)
{
    assert(!mask.empty());
    assert(!mask_lr.empty());

    assert(depth.width() == mask.width());
    assert(depth.height() == mask.height());

    assert(depth_lr.width() == mask_lr.width());
    assert(depth_lr.height() == mask_lr.height());

    float nnz = mask.nonzero();
    float nnz_lr = mask_lr.nonzero();

    // Calculate mean of original depth inside mask
    float mean = depth_lr.mean();

    solver_.normalizeMu(nnz, nnz_lr, mean, normFactor);

    // Initialize depth_opt from depth_lr by bilaterally filtering
    DepthProcessing p(depth);
#if DEPTH_MODE == 1
    p.bilateral(depth_opt, mask, 0.02 * depth.width(), 3.f, 100.f);
#elif DEPTH_MODE == 2
    p.blur(depth_opt, mask, 0.02 * depth.width(), 2.f);
#else
    depth_opt.copyFrom(depth);
#endif

    depth_opt.mask(mask);

    // Remove NaNs if there are any
    InpaintFilter f(mask);
    auto noNaNs = f.removeNans(depth_opt);
    if (!noNaNs)
        throw std::runtime_error("Could not remove all NaNs!");
}

void DepthOptimizer::step(const Image<float>& depth_lr_upsampled,
    const Image<uchar>& mask, const Image<float3>& theta,
    const Image<float3>& dual, const float kappa)
{
    assert(!depth_lr_upsampled.empty());
    assert(!theta.empty());
    assert(!dual.empty());

    solver_.update(depth_opt, depth_lr_upsampled, mask, theta, dual, kappa);
    solver_.run();
}
