#include "solver/solver.h"

using namespace solver;

SolverBase::SolverBase(std::shared_ptr<Solver> solver)
    : solver_(solver) {}

double SolverBase::run()
{
    initialize();
    solve();
    finalize();

    return solver_->finalCost();
}

void SolverBase::solve()
{
    preSolve();
    std::vector<solver::SolverIteration> iterations;

    for (int i = 0; i < parameters_.iterations; i++)
    {
        preNonlinearSolve(i);
        auto cost = solver_->solve(solverParameters_, problemParameters_, iterations_);
        postNonlinearSolve(i);

        if (parameters_.iterations > 1 && parameters_.earlyStop && endEarly_)
        {
            endEarly_ = false;
            break;
        }
    }
    postSolve();
}   