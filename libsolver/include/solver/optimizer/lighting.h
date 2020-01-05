/**
 * @file   lighting.h
 * @brief  Lighting update step
 * @author Florian Windolf
 */
#ifndef SOLVER_OPTIMIZER_LIGHTING_H
#define SOLVER_OPTIMIZER_LIGHTING_H

#include "core/image.h"
#include "solver/solver.h"

namespace solver
{

/**
 * @class LightingOptimizer
 * @brief Performs lighting update argmin[l] | (l * m_theta) * rho  - I |²
 */
class LightingOptimizer
{
public:
    LightingOptimizer(const int w, const int h, const int order = 1);
    LightingOptimizer(LightingOptimizer&& other);
    
    ~LightingOptimizer();

    void init(const Image<float3>& image, const Image<uchar>& mask,
              const Image<float4>& spherical_harmonics);

    void initFrom(const Image<float3>& image, const Image<uchar>& mask, 
                  const Image<float3>& lighting, const Image<float4>& spherical_harmonics);

    void step(const Image<uchar>& mask, const Image<float4>& spherical_harmonics, 
              const Image<float3>& albedo);

    void updateShading(const Image<uchar>& mask, const Image<float4>& spherical_harmonics);

    Image<float3> shading_opt;
    Image<float3> lighting_opt;

private:
    const int w_, h_;

    Image<float> matrixA_, matrixb_; // For Ax = b LGS
    Image<float> matrixATA_, matrixATb_; 

    std::unique_ptr<LinearEquationSolver> lineq;

    const int p_;

    cublasHandle_t cublasHandle_;
};

} // solver

#endif // SOLVER_OPTIMIZER_LIGHTING_H