/**
 * @file   albedo.h
 * @brief  Auxiliary step
 * @author Florian Windolf
 */
#ifndef SOLVER_OPTIMIZER_AUXILIARY_H
#define SOLVER_OPTIMIZER_AUXILIARY_H

#include "core/image.h"
#include "core/camera.h"
#include "solver/solver.h"

namespace solver
{

class AuxiliarySolver : public solver::SolverBase
{
public:
    AuxiliarySolver(const int w, const int h, 
                    const float gamma, const float nu, 
                    const int nIterations, const int lIterations);

    // Update with the latest data
    void update(Image<float3>& theta,  
                const Image<float3>& image, const Image<uchar>& mask,
                const Image<float>& depth, const Image<float3>& lighting, 
                const Image<float3>& albedo, const Image<float3>& dual,
                const IntrinsicsPtr intrinsics,
                const float kappa);

    // Normalize nu for this input depth
    void normalizeNu(const int nnz, const float mean_depth_lr, 
                     const float factor);

    virtual void initialize() override;
    virtual void finalize() override;

    virtual void preSolve() override;
    virtual void postSolve() override;     

    virtual void preNonlinearSolve(int iteration) override;
    virtual void postNonlinearSolve(int iteration) override;

private:
    const int w_, h_;

    float gamma_, nu_;
    float lighting_[4];
    float fx_, fy_, cx_, cy_;
};


/**
 * @class AuxiliaryOptimizer
 * @brief Performs theta update argmin[theta] | (l * m_theta) * rho  - I |² + nu * | dA_theta | + kappa/2 | theta - [z, grad z] + u |²
 */
class AuxiliaryOptimizer
{
public:
    AuxiliaryOptimizer(const int w, const int h, 
                       const float gamma, const float nu,
                       const int nIterations = 1, const int lIterations = 3);
    
    ~AuxiliaryOptimizer();

    void init(const Image<float>& depth, const Image<uchar>& mask, 
              const Image<float>& depth_lr, const Image<uchar>& mask_lr,
              const IntrinsicsPtr intrinsics,
              const float normFactor = 1.f);

    void step(const Image<float3>& image, const Image<uchar>& mask,
              const Image<float>& depth, const Image<float3>& lighting,
              const Image<float3>& albedo, const Image<float3>& dual,
              const IntrinsicsPtr intrinsics, 
              const float kappa);

    Image<float3> theta_opt;
    Image<float4> spherical_harmonics_opt;

private:
    const int w_, h_;

    AuxiliarySolver solver_;
};

} // solver

#endif // SOLVER_OPTIMIZER_AUXILIARY_H