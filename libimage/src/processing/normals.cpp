#include "image/processing/normals.h"

#include "normals_cu.h"

using namespace image;

NormalsProcessing::NormalsProcessing(const Image<float3>& normals)
 : ProcessingBase<float3>(normals)
{
}


void NormalsProcessing::harmonics(Image<float4>& output, const int order) const
{
    if (order > 1)
        throw std::runtime_error("Second (+) order spherical harmonics are not implemented!");
    
    // 1. order
    // s(0..2) = normals
    // s(3) == 1
    if(output.empty())
        output.realloc(image_.width(), image_.height());
    
    cu_NormalsToHarmonics(output, image_);
}
