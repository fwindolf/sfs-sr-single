#include "image/evaluation/base.h"

#include "image/processing.h"

#include "eval_cu.h"

using namespace image;

EvaluationBase::EvaluationBase(const Image<float>& depth, const Image<float>& depth_opt, 
                               const Image<uchar>& mask, const IntrinsicsPtr intrinsics)
 : depth_(depth),
   depth_opt_(depth_opt),
   mask_(mask),
   intrinsics_(intrinsics)
{    
    assert(!depth_.empty());
    assert(!depth_opt_.empty());
    assert(!mask_.empty());
}

void EvaluationBase::calculatePixelwiseError()
{
    assert(pixelwiseError_.empty());

    pixelwiseError_.realloc(depth_.width(), depth_.height());

    // Calculate normals
    Image<float3> normals, normals_opt;
    DepthProcessing pD(depth_);
    pD.normals(normals, intrinsics_);
    
    DepthProcessing pD_opt(depth_opt_);
    pD_opt.normals(normals_opt, intrinsics_);

    // Run kernel on normals
    cu_PixelwiseError(pixelwiseError_, normals, normals_opt);
}

float EvaluationBase::meanError()
{
    if(pixelwiseError_.empty())
        calculatePixelwiseError();

    return pixelwiseError_.mean();
}

float EvaluationBase::medianError()
{
    if(pixelwiseError_.empty())
        calculatePixelwiseError();

    return pixelwiseError_.median();
}

float EvaluationBase::rmsError()
{   
    // sqrt(sum((z-zn).^2)./length(z));
    Image<float> sqerr = (depth_ - depth_opt_).square();   
    sqerr.mask(mask_);
    return sqrtf(sqerr.sum() / mask_.nonzero());
}