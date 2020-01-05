/**
 * @file   solver_iteration.h
 * @brief  Iteration struct for solvers, adapted from Opt
 * @author Florian Windolf
 */

#ifndef SOLVER_SOLVER_ITERATION_H
#define SOLVER_SOLVER_ITERATION_H

namespace solver
{
    
class SolverIteration
{
public:
    SolverIteration() {}
    SolverIteration(double cost, double timeMs)
        : cost(cost), timeMs(timeMs){}

    double cost;
    double timeMs;
};

} // solver

#endif // SOLVER_SOLVER_ITERATION_H