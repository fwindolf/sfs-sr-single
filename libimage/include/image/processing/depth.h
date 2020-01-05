/**
 * @file   depth.h
 * @brief  Depth Processing Interface
 * @author Florian Windolf
 */
#ifndef IMAGE_PROCESSING_DEPTH_H
#define IMAGE_PROCESSING_DEPTH_H

#include "core/camera.h"
#include "image/processing/base.h"

namespace image
{
/**
 * @class DepthProcessing
 * @brief Base class for all depth image manipulation
 */
class DepthProcessing : public ProcessingBase<float>
{
public:
    DepthProcessing(const Image<float>& depth);
    
    ~DepthProcessing(){};

    virtual void cleanNaNs(Image<float>& output, const float val = std::nanf(""));

    virtual void normals(Image<float3>& output, const IntrinsicsPtr camera) const;

    //virtual void normals(Image<float3>& output, const float threshold = 0) const;

    virtual void harmonics(Image<float4>& output, const IntrinsicsPtr camera, const int order = 1) const;

    virtual void shading(Image<float3>& output, const Image<float3>& light, const IntrinsicsPtr depthIntrinsics, const int order = 1) const;

    virtual void addNoise(Image<float>& output, const float mean = 0, const float var = 1) const;

    virtual void blur(Image<float>& output, const Image<uchar>& mask, const int radius, const float sigma) const;

    virtual void bilateral(Image<float>& output, const Image<uchar>& mask, const int radius, const float sigma_space, const float sigma_color) const;

    virtual void theta(Image<float3>& output) const;
    
    virtual int hasNaN() const;
};

} // image

#endif // IMAGE_PROCESSING_DEPTH_H