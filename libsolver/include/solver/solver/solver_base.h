/**
 * @file   solver.h
 * @brief  Base classes for solvers, adapted from Opt 
 * @author Florian Windolf
 */
#ifndef SOLVER_SOLVER_BASE_H
#define SOLVER_SOLVER_BASE_H

#include "core/parameter.h"

#include "solver/solver/solver_iteration.h"
#include "solver/solver/solver_parameters.h"

#include <memory>

namespace solver
{

class Solver 
{
public:
    Solver() {}
    virtual double solve(const core::NamedParameters& solverParameters, const core::NamedParameters& problemParameters, std::vector<SolverIteration>& iterations) = 0;

    virtual double finalCost() const = 0;

protected:
    double finalCost_ = nan("");
};


class SolverBase
{
public:
    SolverBase(){}
    SolverBase(std::shared_ptr<Solver> solver);
    
    virtual double run();

protected:
    virtual void initialize() = 0;
    virtual void finalize() = 0;
    
    virtual void preSolve() = 0;
    virtual void postSolve() = 0;

    virtual void preNonlinearSolve(int iteration) = 0;
    virtual void postNonlinearSolve(int iteration) = 0;

    virtual void solve();    

    std::shared_ptr<Solver> solver_;
    SolverParameters parameters_;

    bool endEarly_ = false;

    core::NamedParameters solverParameters_;
    core::NamedParameters problemParameters_;
    std::vector<SolverIteration> iterations_;
};

} // solver

#endif // SOLVER_SOLVER_BASE_H