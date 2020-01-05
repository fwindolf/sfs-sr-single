/**
 * @file   normals.h
 * @brief  Normals Processing Interface
 * @author Florian Windolf
 */
#ifndef IMAGE_PROCESSING_NORMALS_H
#define IMAGE_PROCESSING_NORMALS_H

#include "core/camera.h"
#include "image/processing/base.h"

namespace image
{
/**
 * @class NormalsProcessing
 * @brief Base class for manipulation of normals images
 */
class NormalsProcessing :  public ProcessingBase<float3>
{
public:
    NormalsProcessing(const Image<float3>& normals);
    
    ~NormalsProcessing(){};

    virtual void harmonics(Image<float4>& output, const int order = 1) const;
};

} // image

#endif // IMAGE_PROCESSING_DEPTH_H