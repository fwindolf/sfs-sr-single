/**
 * @file   evaluation.h
 * @brief  Different methods to easily evaluate the difference between images, depth maps, ...
 * @author Florian Windolf
 */
#ifndef IMAGE_EVALUATION_BASE_H
#define IMAGE_EVALUATION_BASE_H

#include "core/image.h"
#include "core/camera.h"

namespace image
{

/**
 * @class EvaluationBase
 * @brief Standard operations for evaluations
 */
class EvaluationBase
{
public:
    EvaluationBase(const Image<float>& depth, const Image<float>& depth_opt, 
                   const Image<uchar>& mask, const IntrinsicsPtr intrinsics);
    ~EvaluationBase(){};

    /**
     * Calculate the mean pixelwise error
     */
    float meanError();

    /**
     * Calculate the median pixelwise error
     */
    float medianError();

    /**
     * Calculate the RMSE between z*, z
     */
    float rmsError();

private:
    /**
     * Calculate the pixelwise error between the normals of z*, z
     * atan2(sqrt(sum(cross(N,Nn).^2,2)),dot(N,Nn,2))*180/pi;
     */
    void calculatePixelwiseError();

    IntrinsicsPtr intrinsics_;

    const Image<float>& depth_, depth_opt_;
    const Image<uchar>& mask_;

    Image<float> pixelwiseError_;
};



} // image

#endif // IMAGE_EVALUATION_BASE_H