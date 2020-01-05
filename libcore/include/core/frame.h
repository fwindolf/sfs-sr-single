/**
 * @file   frame.h
 * @brief  Container for multiple images
 * @author Florian Windolf
 */
#ifndef CORE_FRAME_H
#define CORE_FRAME_H

#include "core/camera.h"
#include "core/image.h"

#include <string>
#include <vector>

namespace core
{

/**
 * @class Frame
 * @brief Provides access to frames
 */
class Frame
{
public:
    Frame(const Image<float3>& color, const Image<float>& gray,
        const Image<float>& depth, const Image<uchar>& mask,
        const double& timeColor, const double& timeDepth);

    static std::shared_ptr<Frame> create(const int w, const int h,
        const std::string& directory, const std::string& colorName,
        const std::string& depthName, const double timeColor = 0.0,
        const double timeDepth = 0.0, const float scaleColor = 1.f,
        const float scaleDepth = 1.f);

    static std::shared_ptr<Frame> create(const int w, const int h,
        const std::string& directory, const std::string& colorName,
        const std::string& depthName, const std::string& maskName,
        const double timeColor = 0.0, const double timeDepth = 0.0,
        const float scaleColor = 1.f, const float scaleDepth = 1.f);

    /**
     * Access the color image
     */
    const Image<float3>& color() const;

    /**
     * Access the grayscale image
     */
    const Image<float>& gray() const;

    /**
     * Access the depth image
     */
    const Image<float>& depth() const;

    /**
     * Access the mask
     */
    const Image<uchar>& mask() const;

    /**
     * Access the timestamp of color image
     */
    double timeColor() const;

    /**
     * Access the timestamp of depth image
     */
    double timeDepth() const;

protected:
    const Image<float3>& colorRef_;
    const Image<float>& grayRef_;
    const Image<float>& depthRef_;
    const Image<uchar>& maskRef_;

    const double& timeColor_;
    const double& timeDepth_;
};

/**
 * @class FrameContainer
 * @brief Owns the data of a frame
 */
class FrameContainer : public Frame
{
public:
    FrameContainer(Image<float3>&& color, Image<float>&& depth,
        const double& timeColor, const double& timeDepth);

    FrameContainer(Image<float3>&& color, Image<float>&& depth,
        Image<uchar>&& mask, const double& timeColor, const double& timeDepth);

    FrameContainer(Image<float3>&& color, Image<float>&& gray,
        Image<float>&& depth, Image<uchar>&& mask, const double& timeColor,
        const double& timeDepth);

    FrameContainer(FrameContainer&& other);

    /**
     * Update the depth of this frame
     */
    void setDepth(const Image<float>& depth);

    /**
     * Update the color image of this frame
     */
    void setColor(const Image<float3>& color);

    /**
     * Update the mask of this frame
     */
    void setMask(const Image<uchar>& mask);

private:
    Image<float3> color_;
    Image<float> gray_;
    Image<float> depth_;
    Image<uchar> mask_;
};

} // core

using FramePtr = std::shared_ptr<core::Frame>;
using FrameContainerPtr = std::shared_ptr<core::FrameContainer>;

#endif // CORE_FRAME_H
