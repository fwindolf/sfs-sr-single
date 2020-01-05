#include "solver/solver/solver_lineq.h"
#include "solver_lineq_cu.h"

#include <Eigen/Dense>

using namespace solver;

void swap(float **r, float **s)
{
    float *swap = *r;
    *r = *s;
    *s = swap;
}

LinearEquationSolver::LinearEquationSolver(const Image<float>& A, const Image<float>& b)
    : M(A.height()),
      N(A.width()),
      K(b.width()),
      d_A(nullptr),
      d_b(nullptr)
{
    // b fits A in height (A:MxN, x:NxK, b:MxK)
    assert(b.height() == M);

    // System is (over)determined
    assert(M >= N);

    // Initialize cusolver, cublas
    cusolverSafeCall(cusolverDnCreate(&solverHandle_));
    cublasSafeCall(cublasCreate(&cublasHandle_));

    cudaSafeCall(cudaMalloc(&d_tau, M * sizeof(float)));
    cudaSafeCall(cudaMalloc(&d_devInfo, sizeof(int)));

    cudaSafeCall(cudaMalloc(&d_R, N * N * sizeof(float)));
    cudaSafeCall(cudaMalloc(&d_x, N * K * sizeof(float)));

    fillA(A);
    fillb(b);
}

LinearEquationSolver::LinearEquationSolver(const int dimM, const int dimN, const int dimK)
    : M(dimM),
      N(dimN),
      K(dimK),
      d_A(nullptr),
      d_b(nullptr)
{
    assert(M >= 1);
    assert(M >= N);

    cudaSafeCall(cudaMalloc(&d_tau, M * sizeof(float)));
    cudaSafeCall(cudaMalloc(&d_devInfo, sizeof(int)));

    cudaSafeCall(cudaMalloc(&d_R, N * N * sizeof(float)));
    cudaSafeCall(cudaMalloc(&d_x, N * K * sizeof(float)));

    // Initialize cusolver, cublas
    cusolverSafeCall(cusolverDnCreate(&solverHandle_));
    cublasSafeCall(cublasCreate(&cublasHandle_));
}

LinearEquationSolver::~LinearEquationSolver()
{
    cudaSafeCall(cudaFree(d_tau));
    cudaSafeCall(cudaFree(d_devInfo));

    cudaSafeCall(cudaFree(d_R));
    cudaSafeCall(cudaFree(d_x));

    cublasSafeCall(cublasDestroy(cublasHandle_));
    cusolverSafeCall(cusolverDnDestroy(solverHandle_));
}

/**
 * @brief Solve the equation system and return x
 */
void LinearEquationSolver::solve(Image<float>& x)
{
    if (x.empty())
        x.realloc(K, M);

    assert(d_A);
    assert(d_b);

    solve_for(x);
}

void LinearEquationSolver::fillA(const Image<float>& A)
{
    if (!d_A)
        cudaSafeCall(cudaMalloc(&d_A, M * N * sizeof(float)));

    cu_CopyTransposed(d_A, A);
    assert(d_A);
}

void LinearEquationSolver::fillb(const Image<float>& b)
{
    if (!d_b)
        cudaSafeCall(cudaMalloc(&d_b, M * K * sizeof(float)));

    cu_CopyTransposed(d_b, b);
    assert(d_b);
}

void LinearEquationSolver::solve(Image<float> &x, const Image<float>& A, const Image<float>& b)
{
    assert(A.height() == M);
    assert(A.width() == N);

    assert(b.height() == M);
    assert(b.width() == K);

    fillA(A);
    fillb(b);

    // Reset work buffers
    cudaSafeCall(cudaMemset(d_tau, 0, M * sizeof(float)));
    cudaSafeCall(cudaMemset(d_devInfo, 0, sizeof(int)));

    cudaSafeCall(cudaMemset(d_R, 0, N * N * sizeof(float)));
    cudaSafeCall(cudaMemset(d_x, 0, N * K * sizeof(float)));

    solve_for(x);
}

void LinearEquationSolver::solve_for(Image<float>& x)
{
    if (x.empty())
        x.realloc(K, M);

    assert(d_A);
    assert(d_b);

    // GEQRF: QR factorization
    // Calculate memory requirement
    int bufSize;
    cusolverSafeCall(cusolverDnSgeqrf_bufferSize(solverHandle_, M, N, d_A, M, &bufSize));
    cudaSafeCall(cudaMalloc(&d_work, bufSize * sizeof(float)));

    // Computes R, Q in A (R in upper, household vectors of Q in lower triangle of A)
    cusolverSafeCall(cusolverDnSgeqrf(solverHandle_, M, N, d_A, M, d_tau, d_work, bufSize, d_devInfo));
    cudaSafeCall(cudaMemcpy(&devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo != 0)
    {
        std::cerr << "Could not execute GEQRF properly, quit solving..." << std::endl;
        return;
    }

    cudaSafeCall(cudaMemset(d_work, 0, bufSize * sizeof(float)));
    // ORMQR: Computes Q^T * C and stores it in C
    cusolverSafeCall(cusolverDnSormqr(solverHandle_, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, K, N,
                                      d_A, M, d_tau, d_b, M, d_work, bufSize, d_devInfo));
    // cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaMemcpy(&devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo != 0)
    {
        std::cerr << "Could not execute ORMQR properly, quit solving..." << std::endl;
        return;
    }

    //  A contains R in upper triangle, copy explicitly to d_R
    cu_CopyUpperSubmatrix(d_R, d_A, M, N, N);
    cu_CopyUpperSubmatrix(d_x, d_b, M, K, N);

    // TRSM: Solve upper triangular LGS, x = R \ Q^T * B
    const float one = 1.f;
    cublasSafeCall(cublasStrsm(cublasHandle_, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, 
                               CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                               N, K, &one, d_R, N, d_x, N));
    // cudaSafeCall(cudaDeviceSynchronize());

    cu_CopyTransposed(x, d_x);

    cudaSafeCall(cudaFree(d_work)); // No reallocation...
}
