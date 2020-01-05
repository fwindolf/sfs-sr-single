#include "solver/solver/solver_opt.h"

extern "C" {
    #include "Opt.h"
}

using namespace solver;

OptSolver::OptSolver(const std::vector<int>& dimensions, const std::string& terraFile, const std::string& method,
                     bool doublePrecision, int verbosityLevel, int collectTiming)
    : state_(nullptr), 
      problem_(nullptr), 
      plan_(nullptr), 
      doublePrecision_(doublePrecision)
{
    // Initialize Opt
    Opt_InitializationParameters initParams;
    memset(&initParams, 0, sizeof(Opt_InitializationParameters));
    initParams.verbosityLevel = verbosityLevel;
    initParams.collectPerKernelTimingInfo = collectTiming;
    initParams.doublePrecision = (int)doublePrecision_;

    std::string solver_method = method;
    if (method != "gaussNewtonGPU" && method != "LMGPU")
    {
        std::cout << "Solver method was set to invalid method " << method << ", setting to gaussNewtonGPU" << std::endl;
        solver_method = "gaussNewtonGPU";
    }

    state_ = Opt_NewState(initParams);
    if (!state_)
    {
        std::cerr << "Solver state could not be created, maybe the initParams were invalid?" << std::endl;
        throw std::invalid_argument("Unable to create Opt State");
    }

    problem_ = Opt_ProblemDefine(state_, terraFile.c_str(), solver_method.c_str());
    if(!problem_)
    {
        std::cerr << "Solver problem could not be defined, maybe the filepath " << terraFile << " was wrong?" << std::endl;
        throw std::invalid_argument("Unable to define Opt Problem");
    }

    plan_ = Opt_ProblemPlan(state_, problem_, (unsigned int*)dimensions.data());
    if (!plan_)
        throw std::invalid_argument("Unable to create Opt Problem Plan");
}

OptSolver::~OptSolver()
{
    if (plan_) 
        Opt_PlanFree(state_, plan_);

    if (problem_)
        Opt_ProblemDelete(state_, problem_);
}

double OptSolver::solve(const core::NamedParameters& solverParameters, const core::NamedParameters& problemParameters, std::vector<SolverIteration>& iterations) 
{
    finalCost_ = nan("");
    
    core::NamedParameters finalProblemParameters = problemParameters;

    // Convert problem parameters to double precision if set
    if (doublePrecision_)
    {
        // TODO: Convert
        throw std::runtime_error("Double precision currently not supported!");
    }

    // Set the solver parameters
    for(std::string& name : solverParameters.names())
    {
        auto param = solverParameters.at(name);
        Opt_SetSolverParameter(state_, plan_, name.c_str(), param.data());
    }

    // Run the solver routine
    Opt_ProblemInit(state_, plan_, problemParameters.data().data());
    while(Opt_ProblemStep(state_, plan_, problemParameters.data().data()) != 0) 
    {
        // inspect and update problem state as desired.
    }


    finalCost_ = Opt_ProblemCurrentCost(state_, plan_);

    // Convert results back from double precision if set
    if (doublePrecision_)
    {
        // TODO: Convert
        throw std::runtime_error("Double precision currently not supported!");
    }
    
    return finalCost_;
}

double OptSolver::finalCost() const
{
    return finalCost_;
}