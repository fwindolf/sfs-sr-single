/**
 * @file   solver_lineq.h
 * @brief  Solve linear equation systems of form Ax = b 
 * @author Florian Windolf
 * 
 * Adapted from: https://stackoverflow.com/a/28484749/4658360
 */
#ifndef SOLVER_SOLVER_SOLVER_LINEQ_H
#define SOLVER_SOLVER_SOLVER_LINEQ_H

#include "core/image.h"

#include <cusolverDn.h>
#include <cublas_v2.h>

namespace solver
{

/**
 * @class LinearEquationSolver
 * @brief Solves linear equations using cuSolverDn Dense with QR factorization
 */
class LinearEquationSolver
{
public:
    /**
     * @brief Initialize the LineqSolver with data from A [MxN] and b [MxK]
     */
    LinearEquationSolver(const Image<float>& A, const Image<float>& b);

    /**
     * @brief Initialize the LineqSolver with dimensions M, N, K
     * A [MxN] * x [NxK] = b [MxK]
     */
    LinearEquationSolver(const int M, const int N, const int K);

    ~LinearEquationSolver();

    /**
     * @brief Solve the equation for x
     * x [NxK]
     */
    void solve(Image<float>& x);

    /**
     * @brief Solve the equation A * x = b for x
     * A [MxN] * x [NxK] = b [MxK]
     */
    void solve(Image<float>& x, const Image<float> &A, const Image<float> &b);

private:
    void fillA(const Image<float>& A);

    void fillb(const Image<float>& b);

    void solve_for(Image<float>& x);

    cusolverDnHandle_t solverHandle_;
    cublasHandle_t cublasHandle_;

    float *d_A, *d_b, *d_tau, *d_work, *d_R, *d_x;
    int *d_devInfo;
    int devInfo;

    const int M, N, K;
};

} // namespace solver

#endif // SOLVER_SOLVER_SOLVER_LINEQ_H