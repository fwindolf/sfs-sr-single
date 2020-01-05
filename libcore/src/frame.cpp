#include "core/frame.h"

using namespace core;

/**
 * Frame
 */

Frame::Frame(const Image<float3>& color, const Image<float>& gray,
    const Image<float>& depth, const Image<uchar>& mask,
    const double& timeColor, const double& timeDepth)
    : colorRef_(color)
    , grayRef_(gray)
    , depthRef_(depth)
    , maskRef_(mask)
    , timeColor_(timeColor)
    , timeDepth_(timeDepth)
{
}

std::shared_ptr<Frame> Frame::create(const int w, const int h,
    const std::string& directory, const std::string& colorName,
    const std::string& depthName, const double timeColor,
    const double timeDepth, const float scaleColor, const float scaleDepth)
{
    Image<float3> color(directory + colorName);
    color *= make_float3(scaleColor, scaleColor, scaleColor);

    Image<float> gray = color.asGray<float>();

    Image<float> depth(directory + depthName);
    depth *= scaleDepth;

    Image<float> depth_mask = depth;
    depth_mask.threshold(0.1f, 0.f, 1.f); // Everything that has a valid
                                          // value is set to 1, else to 0
    Image<uchar> mask = depth_mask.as<uchar>() * (uchar)255;

    return std::make_shared<core::FrameContainer>(std::move(color),
        std::move(gray), std::move(depth), std::move(mask), timeColor,
        timeDepth);
}

std::shared_ptr<Frame> Frame::create(const int w, const int h,
    const std::string& directory, const std::string& colorName,
    const std::string& depthName, const std::string& maskName,
    const double timeColor, const double timeDepth, const float scaleColor,
    const float scaleDepth)
{
    assert(!colorName.empty());
    assert(!depthName.empty());
    assert(!maskName.empty());

    Image<float3> color(directory + colorName);
    color *= make_float3(scaleColor, scaleColor, scaleColor);

    Image<float> gray = color.asGray<float>();

    Image<float> depth(directory + depthName);
    depth *= scaleDepth;

    Image<uchar> mask(directory + maskName);
    return std::make_shared<core::FrameContainer>(std::move(color),
        std::move(depth), std::move(mask), timeColor, timeDepth);
}

const Image<float3>& Frame::color() const { return colorRef_; }
const Image<float>& Frame::gray() const { return grayRef_; }
const Image<float>& Frame::depth() const { return depthRef_; }
const Image<uchar>& Frame::mask() const { return maskRef_; }
double Frame::timeColor() const { return timeColor_; }
double Frame::timeDepth() const { return timeDepth_; }

/**
 * FrameContainer
 */
FrameContainer::FrameContainer(Image<float3>&& color, Image<float>&& depth,
    const double& timeColor, const double& timeDepth)
    : color_(color)
    , gray_(color.asGray<float>())
    , depth_(depth)
    , Frame(color_, gray_, depth_, mask_, timeColor, timeDepth)
{
    Image<float> depth_mask = depth;
    depth_mask.threshold(0.1f, 0.f, 1.f); // Everything that has a valid
                                          // value is set to 1, else to 0
    mask_ = depth_mask.as<uchar>() * (uchar)255;
}

FrameContainer::FrameContainer(Image<float3>&& color, Image<float>&& depth,
    Image<uchar>&& mask, const double& timeColor, const double& timeDepth)
    : color_(color)
    , gray_(color.asGray<float>())
    , depth_(depth)
    , mask_(mask)
    , Frame(color_, gray_, depth_, mask_, timeColor, timeDepth)
{
}

FrameContainer::FrameContainer(Image<float3>&& color, Image<float>&& gray,
    Image<float>&& depth, Image<uchar>&& mask, const double& timeColor,
    const double& timeDepth)
    : color_(color)
    , gray_(gray)
    , depth_(depth)
    , mask_(mask)
    , Frame(color_, gray_, depth_, mask_, timeColor, timeDepth)
{
}

FrameContainer::FrameContainer(FrameContainer&& other)
    : color_(other.color_)
    , depth_(other.depth_)
    , gray_(other.gray_)
    , mask_(other.mask_)
    , Frame(color_, gray_, depth_, mask_, other.timeColor_, other.timeDepth_)
{
}

void FrameContainer::setDepth(const Image<float>& depth)
{
    depth_.copyFrom(depth);
}

void FrameContainer::setColor(const Image<float3>& color)
{
    color_.copyFrom(color);
}

void FrameContainer::setMask(const Image<uchar>& mask)
{
    mask_.copyFrom(mask);
}