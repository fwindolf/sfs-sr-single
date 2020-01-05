/**
 * @file   albedo.h
 * @brief  Albedo step
 * @author Florian Windolf
 */
#ifndef SOLVER_OPTIMIZER_ALBEDO_H
#define SOLVER_OPTIMIZER_ALBEDO_H

#include "core/image.h"
#include "image/filter/mumfordshah.h"

namespace solver
{

/**
 * @class AlbedoOptimizer
 * @brief Performs albedo update argmin[rho] | (l * m_theta) * rho  - I |² + lambda * | grad rho | 
 */
class AlbedoOptimizer
{
public:
    AlbedoOptimizer(const int w, const int h, float msAlpha, float msLambda, int msMaxIter);
    AlbedoOptimizer(AlbedoOptimizer&& other);
    
    ~AlbedoOptimizer();

    void init(const Image<float3>& image, const Image<uchar>& mask, 
              const Image<float3>& shading);

    void initFrom(const Image<float3>& image, const Image<uchar>& mask, 
                  const Image<float3>& albedo);

    void step(const Image<float3>& shading);

    Image<float3> albedo_opt;
private:    
    const int w_, h_;

    const float msAlpha_, msLambda_;
    const int msMaxIter_;
    std::unique_ptr<image::MumfordShahFilter<float3>> msFilter_;    
};

} // solver

#endif // SOLVER_OPTIMIZER_ALBEDO_H