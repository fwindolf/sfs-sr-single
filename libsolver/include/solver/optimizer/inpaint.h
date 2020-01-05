/**
 * @file   inpaint.h
 * @brief  Inpainting of undefined/zero values in images
 * @author Florian Windolf
 */
#ifndef SOLVER_OPTIMIZERIZER_INPAINT_H
#define SOLVER_OPTIMIZERIZER_INPAINT_H

#include "core/image.h"
#include "solver/solver.h"

namespace solver
{

/**
 * @class InpaintNaNFilter
 * @brief Inpaint holes in images
 */
class InpaintFilter : public SolverBase
{
public:
    InpaintFilter(const Image<uchar>& mask);
    ~InpaintFilter();

    bool removeNans(Image<float>& depth);   
private:

    void initialize() override {}   
    void finalize() override {}

    void preSolve() override {}
    void postSolve() override {}
    
    void preNonlinearSolve(int iteration) override {}
    void postNonlinearSolve(int iteration) override {}
    int numMaskedNans_ = -1;

    const Image<uchar>& mask_;    
    Image<float> target_;
};

} // solver

#endif // SOLVER_OPTIMIZERIZER_INPAINT_H