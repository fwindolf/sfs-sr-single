#ifndef SOLVER_SOLVER_PARAMETERS_H
#define SOLVER_SOLVER_PARAMETERS_H

namespace solver
{

struct SolverParameters
{
    bool earlyStop = false;
    bool doublePrecision = false;

    unsigned int iterations = 1;
    unsigned int linearIterations = 10;
    unsigned int nonlinearIterations = 200;
};

} // solver

#endif // SOLVER_SOLVER_PARAMETERS_H