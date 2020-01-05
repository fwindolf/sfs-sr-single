#include "image/processing/depth.h"

#include "depth_cu.h"

using namespace core;
using namespace image;

DepthProcessing::DepthProcessing(const Image<float>& depth)
 : ProcessingBase<float>(depth)
{
}

void DepthProcessing::cleanNaNs(Image<float>& output, const float val)
{
    if (output.empty())
        output.realloc(image_.width(), image_.height());

    output.copyFrom(image_);
    output.replace(0.f, val);
    output.replaceNan(val);
}

void DepthProcessing::normals(Image<float3>& output, const IntrinsicsPtr depthIntrinsics) const
{
    if (output.empty())
        output.realloc(image_.width(), image_.height());

    //Image<float> img(image_);
    //img.show<cuimage::DEPTH_TYPE>("Depth");
    cu_DepthToNormals(output, image_, 
        depthIntrinsics->fx(), depthIntrinsics->fy(),
        depthIntrinsics->cx(), depthIntrinsics->cy());
}

void DepthProcessing::harmonics(Image<float4>& output, const IntrinsicsPtr depthIntrinsics, const int order) const
{
    if (order > 1)
        throw std::runtime_error("Second (+) order spherical harmonics are not implemented!");
    
    if (output.empty())
        output.realloc(image_.width(), image_.height());
    
    cu_DepthToHarmonics(output, image_, 
        depthIntrinsics->fx(), depthIntrinsics->fy(),
        depthIntrinsics->cx(), depthIntrinsics->cy());
}

void DepthProcessing::shading(Image<float3>& output, const Image<float3>& light, const IntrinsicsPtr depthIntrinsics, const int order) const
{
    // Calculate normals
    Image<float3> n;
    normals(n, depthIntrinsics);

    // Get spherical harmonics coefficients from light image
    const int numSpHarm = (order + 1) * (order + 1);
    assert(light.height() >= numSpHarm);
    
    if (output.empty())
        output.realloc(image_.width(), image_.height());

    cu_NormalsToShading(output, n, light);
}


void DepthProcessing::addNoise(Image<float>& output, const float mean, const float var) const
{
    if (output.empty())
        output.realloc(image_.width(), image_.height());

    cu_NormalDistribution(output, image_, mean, var);
}

void DepthProcessing::blur(Image<float>& output, const Image<uchar>& mask, const int radius, const float sigma) const
{
    assert(image_.width() == mask.width());
    assert(image_.height() == mask.height());

    output.copyFrom(image_);    

    if (mask.empty())
    {
        Image<uchar> mask_(image_.width(), image_.height(), 255);
        cu_GaussianSmooth(output, mask_, radius, sigma);
    }
    else
    {
        cu_GaussianSmooth(output, mask, radius, sigma);
    }
}

void DepthProcessing::bilateral(Image<float>& output, const Image<uchar>& mask, const int radius, const float sigma_space, const float sigma_color) const
{
    assert(image_.width() == mask.width());
    assert(image_.height() == mask.height());
    output.copyFrom(image_);
    
    if (mask.empty())
    {
        Image<uchar> mask_(image_.width(), image_.height(), 255);
        cu_BilateralSmooth(output, mask_, radius, sigma_space, sigma_color);
    }
    else
    {
        cu_BilateralSmooth(output, mask, radius, sigma_space, sigma_color);
    }    
}

void DepthProcessing::theta(Image<float3>& output) const
{
    if (output.empty())
        output.realloc(image_.width(), image_.height());
    
    cu_DepthToTheta(output, image_); 
}

int DepthProcessing::hasNaN() const
{
    return (bool)(image_.size() - image_.valid());
}
