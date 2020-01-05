/**
 * @file   solver_ceres.h
 * @brief  Wrapper for Ceres solver
 * @author Florian Windolf
 */
#ifndef SOLVER_SOLVER_CERES_H
#define SOLVER_SOLVER_CERES_H

#include "solver/solver.h"

#if USE_CERES
#include "ceres/ceres.h"
using ceres::DynamicAutoDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
#endif

namespace solver
{

/**
 * @class CeresSolver
 * @brief Wrapper for Ceres solver according to Solver interface
 */
class CeresSolver : public Solver 
{
public:
    CeresSolver(const std::vector<int>& dimensions)
#if USE_CERES
     : options_(new ceres::Solver::Options()),
       problem_(new ceres::Problem())
    {
        options_->num_threads = 8;
        options_->num_linear_solver_threads = 8;
        options_->linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
        options_->max_num_iterations = 10000;
        options_->function_tolerance = 1e-3;
        options_->gradient_tolerance = 1e-4 * options_->function_tolerance;
    }
#endif
    {
    }

    ~CeresSolver()
    {        
    }

    virtual double solve(const core::NamedParameters& solverParameters, const core::NamedParameters& problemParameters, std::vector<SolverIteration>& iterations) override
    {
#if USE_CERES
        Solver::Summary summary;
        Solve(*options, problem, &summary);
        
        problem_->Evaluate(ceres::Problem::EvaluateOptions(), &finalCost_, nullptr, nullptr, nullptr);

        std::cout << summary.FullReport() << std::endl;
        return finalCost_;
#endif
    }

    virtual double finalCost() const
    {
        return finalCost_;
    }


#if USE_CERES
    std::shared_ptr<ceres::Problem> getProblem() const
    {
        return problem_;
    }
#endif

protected:
    bool doublePrecision_ = false;

#if USE_CERES
    std::unique_ptr<ceres::Solver::Options> options_;
    std::shared_ptr<ceres::Problem> problem_;
#endif
};

} // solver

#endif // SOLVER_SOLVER_CERES_H