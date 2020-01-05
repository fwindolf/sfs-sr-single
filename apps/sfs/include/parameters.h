/**
 * @file   parameters.h
 * @brief  Parameters for SFS optimizer
 * @author Florian Windolf
 */
#ifndef SFS_PARAMETERS_H
#define SFS_PARAMETERS_H


/**
 * @class SfsParameters
 * @brief Default parameters for SfS optimization
 */
struct SfsParameters
{
    // Albedo Update
    float alpha         = -1.f;
    float lambda        = .5f;
    float maxIter       = 200.f;

    // Theta update
    float gamma         = 1.0f;
    float nu            = 1e-2f;

    // Depth Update
    float mu            = 1e-3f;
    
    // ADMM parameters
    float kappa         = 1e-4f;
    float tau           = 2.0f;
    float tolerance     = 1e-6f; // 0: No relative error considered
    float toleranceEL   = 1e-6f;
};

#endif // SFS_PARAMETERS_H