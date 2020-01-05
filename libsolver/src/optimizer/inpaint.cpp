#include "solver/optimizer/inpaint.h"
#include "solver/optimizer/inpaint_cu.h"

using namespace core;
using namespace solver;

InpaintFilter::InpaintFilter(const Image<uchar>& mask)
 : mask_(mask)
{
    assert(!mask_.empty());
}

InpaintFilter::~InpaintFilter()
{
}

bool InpaintFilter::removeNans(Image<float>& depth)
{
    assert(!depth.empty());
    
    assert(depth.width() == mask_.width());
    assert(depth.height() == mask_.height());
   
    // Get valid pixels inside mask
    depth.replace(0.f, std::nanf(""));
    depth.mask(mask_);
    int numNan = depth.size() - depth.valid(); // doesnt count 0s (outside mask)

    if (numNan == 0)
        return true;
    
    target_.copyFrom(depth);
    target_.replace(0.f, std::nanf(""));

    int tries = 0;
    while(numNan > 0)
    {
        if (tries < 100)
            target_.replace(0.f, std::nanf(""));
        else
            break;

        target_.mask(mask_);
        cu_PatchHoles(target_);
       
        numNan = target_.size() - target_.valid();
        // std::cout << "num NaN : " << numNan << std::endl;
        tries++;
    }

    if (numNan == 0)
    {
        depth.copyFrom(target_);
        return true;
    }
                  
    std::vector<int> dimensions = { (int)depth.width(), (int)depth.height() };
    if (!solver_)
    {
        std::string file = std::string(SOURCE_DIR) + "/libsolver/src/opt/inpaintNan.t";     
        solver_ = std::make_shared<solver::OptSolver>(dimensions, file, "gaussNewtonGPU", false, 0, 0);
    }
    
    if (numNan > 0)
        target_.replaceNan(target_.mean());
    
    problemParameters_.add("X", target_, 0);
    problemParameters_.add("A", depth, 1);      
    problemParameters_.add("M", mask_, 2);   

    // Run this solver
    solverParameters_.clear();
    solverParameters_.add("nIterations", 5);
    solverParameters_.add("lIterations", 4);
    run();

    depth.copyFrom(target_);
    numNan = depth.size() - depth.valid();
    depth.mask(mask_);

    if (numNan > 0)
        std::cout << "Could not remove NaNs from image!" << std::endl;
    
    return (numNan == 0);
}