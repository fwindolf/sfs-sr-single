/**
 * @file   lagrange.h
 * @brief  Lagrange update
 * @author Florian Windolf
 */
#ifndef SOLVER_OPTIMIZERIZER_LAGRANGE_H
#define SOLVER_OPTIMIZERIZER_LAGRANGE_H

#include "core/image.h"

namespace solver
{

/**
 * @class LagrangeOptimizer
 * @brief Performs lagrange update u = u + theta - (z, gradz)
 */
class LagrangeOptimizer
{
public:
    LagrangeOptimizer(const int w, const int h,
                      const float tau, const float kappa, 
                      const float tolerance, const float toleranceEL);
    
    ~LagrangeOptimizer();

    void init(const Image<float>& depth);

    bool step(const Image<uchar>& mask, const Image<float3>& theta,
              const Image<float>& depth);

    Image<float3> dual_opt;
    Image<float3> theta_z;
    Image<float> depth_last;

    float kappa;
    
private:
    const int w_, h_;
    const float tau_;
    const float kappa_;
    
    float depthNorm_;

    float tolerance_;
    float toleranceEL_;
};

} // solver

#endif // SOLVER_OPTIMIZERIZER_LAGRANGE_H