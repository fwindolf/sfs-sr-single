/**
 * @file   solver_opt.h
 * @brief  Wrapper class for Opt solver
 * @author Florian Windolf
 */
#ifndef SOLVER_SOLVER_OPT_H
#define SOLVER_SOLVER_OPT_H

#include "solver/solver/solver_base.h"

class Opt_State;
class Opt_Problem;
class Opt_Plan;

namespace solver
{

/**
 * @class OptSolver
 * @brief Wrapper for Opt solver according to Solver interface
 */
class OptSolver : public Solver 
{
public:
    OptSolver(const std::vector<int>& dimensions, const std::string& terraFile, const std::string& method,
              bool doublePrecision = false, int verbosityLeveL = 1, int collectTimnig = 1);

    ~OptSolver();

    virtual double solve(const core::NamedParameters& solverParameters, const core::NamedParameters& problemParameters, std::vector<SolverIteration>& iterations) override;
    
    virtual double finalCost() const override;

protected:
    Opt_State* state_;
    Opt_Problem* problem_;
    Opt_Plan* plan_;

    bool doublePrecision_ = false;
};

} // solver

#endif // SOLVER_SOLVER_OPT_H