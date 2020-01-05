#include "solver/optimizer/lagrange.h"

#include "image/processing.h"

using namespace solver;

LagrangeOptimizer::LagrangeOptimizer(const int w, const int h, const float tau,
    const float kappa, const float tolerance, const float toleranceEL)
    : w_(w)
    , h_(h)
    , tau_(tau)
    , kappa_(kappa)
    , kappa(kappa)
    , tolerance_(tolerance)
    , toleranceEL_(toleranceEL)
{
}

LagrangeOptimizer::~LagrangeOptimizer() {}

void LagrangeOptimizer::init(const Image<float>& depth)
{
    depth_last.copyFrom(depth);
    depthNorm_ = depth.norm2();

    // Initialize
    dual_opt.realloc(w_, h_);
    dual_opt.setTo(0.f);

    kappa = kappa_;
}

bool LagrangeOptimizer::step(const Image<uchar>& mask,
    const Image<float3>& theta, const Image<float>& depth)
{
    assert(!theta.empty());
    assert(!depth.empty());

    // Calculate theta from depth
    image::DepthProcessing p(depth);
    p.theta(theta_z);

    Image<float3> t_diff = theta_z - theta;

    // u^(k+½) = u^(k) + thetaZ - theta
    dual_opt += t_diff;

    // kappa^(k+1) = tau * kappa
    kappa *= tau_;

    // u^(k+1) = u^(k+½) / tau
    dual_opt /= make_float3(tau_, tau_, tau_);

    // Check convergence
    // || z - z_last ||_2^2 / norm(z_init);
    depth_last -= depth;
    depth_last.mask(mask);

    const float residual = depth_last.norm2() / depthNorm_;

    // 0.5 * kappa * || theta-(z,zx,zy)||_2^2 + || u.*(theta-(z,zx,zy)) ||_1
    t_diff.mask(mask);

    auto t_diff_sq = t_diff * t_diff;
    float norm2 = .5f * kappa * t_diff_sq.sum();

    t_diff *= dual_opt;
    float norm1 = t_diff.norm1();

    float diffEL = norm1 + norm2;

    // std::cout << std::endl << "EL: " << diffEL << " RES: " << residual <<
    // std::endl;

    if (fabsf(diffEL) < toleranceEL_)
        return true;

    if (tolerance_ > 0 && residual < tolerance_)
        return true;

    if (isinff(kappa))
        return true;

    depth_last.copyFrom(depth);
    return false;
}