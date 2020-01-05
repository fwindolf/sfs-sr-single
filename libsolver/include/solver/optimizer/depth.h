/**
 * @file   depth.h
 * @brief  Depth step
 * @author Florian Windolf
 */
#ifndef SOLVER_OPTIMIZER_DEPTH_H
#define SOLVER_OPTIMIZER_DEPTH_H

#include "core/image.h"
#include "core/camera.h"
#include "solver/solver.h"

namespace solver
{

class DepthSolver : public SolverBase
{
public:
    DepthSolver(const int w, const int h, const float mu,
                const int nIterations, const int lIterations);

    void update(Image<float>& depth, const Image<float>& depth_lr,
                const Image<uchar>& mask, const Image<float3>& theta,
                const Image<float3>& dual, const float kappa);

    void normalizeMu(const int nnz, const int nnz_lr, 
                     const float mean, const float factor);

    virtual void initialize() override;
    virtual void finalize() override;

    virtual void preSolve() override;
    virtual void postSolve() override;     

    virtual void preNonlinearSolve(int iteration) override;
    virtual void postNonlinearSolve(int iteration) override;

private:
    const int w_, h_;
    float mu_;
};

/**
 * @class DepthOptimizer
 * @brief Performs depth update argmin[z] | K z - z0 |² + kappa/2 | theta - [z, grad z] + u |²
 */
class DepthOptimizer
{
public:
    DepthOptimizer(const int w, const int h, const float mu,
                   const int nIterations = 1, const int lIterations = 1);

    ~DepthOptimizer();

    void init(const Image<float>& depth, const Image<uchar>& mask,
              const Image<float>& depth_lr, const Image<uchar>& mask_lr,
              const float normFactor = 1.f);

    void step(const Image<float>& depth_lr, const Image<uchar>& mask, 
              const Image<float3>& theta, const Image<float3>& dual,
              const float kappa);

    Image<float> depth_opt;    
private:
    const int w_, h_;
    DepthSolver solver_;
};

} // solver

#endif // SOLVER_OPTIMIZER_DEPTH_H