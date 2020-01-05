/**
 * Pyramid
 * - downsampling type
 * - vectors for different levels -> vector<Frame>
 * - owns only the non-native levels
 * - access to different levels -> FrameViews
 */
/**
 * @file   pyramid.h
 * @brief  Frame pyramids
 * @author Florian Windolf
 */
#ifndef CORE_PYRAMID_H
#define CORE_PYRAMID_H

#include "core/camera.h"
#include "core/frame.h"
#include "core/image.h"

#include <iterator>

namespace core
{

/**
 * @class PyramidBase
 * @brief Pyramid base class
 */
class PyramidBase
{
public:
    PyramidBase(const int w, const int h, const int levels)
        : w_(w)
        , h_(h)
        , levels_(levels)
    {
    }

    int width() const { return w_; }
    int height() const { return h_; }
    int levels() const { return levels_; }

protected:
    const int w_, h_;
    const int levels_;
};

/**
 * @class ImagePyramid
 * @brief Pyramid for images
 */
template <typename T> class ImagePyramid : public PyramidBase
{
public:
    ImagePyramid(const int w, const int h, const int levels,
        const cuimage::ResizeMode mode = cuimage::ResizeMode::LINEAR)
        : PyramidBase(w, h, levels)
        , mode_(mode)
    {
    }

    ImagePyramid(ImagePyramid&& other)
        : PyramidBase(other.w_, other.h_, other.levels_)
        , mode_(other.mode_)
    {
        images_.clear();
        images_.insert(images_.end(),
            std::make_move_iterator(other.images_.begin()),
            std::make_move_iterator(other.images_.end()));
        other.images_.clear();
    }

    void build(const Image<T>& image)
    {
        Image<T> copy = image;
        build(std::move(copy));
    }

    void build(Image<T>&& image)
    {
        images_.clear();
        images_.push_back(image);

        int w = w_;
        int h = h_;
        for (int l = 1; l < levels_; l++)
        {
            w /= 2;
            h /= 2;
            const auto& last = images_.at(l - 1);
            images_.emplace_back(last.resized(w, h, mode_));
        }
    }

    const Image<T>& at(const int level) const
    {
        assert(level >= 0 && level < levels_);
        return images_.at(level);
    }

    void swap(ImagePyramid& other)
    {
        assert(images_.size() == other.images_.size());
        assert(mode_ == other.mode_);

        images_.swap(other.images_);
    }

    cuimage::ResizeMode resizeMode() const { return mode_; }

protected:
    const cuimage::ResizeMode mode_;
    std::vector<Image<T>> images_;
};

/**
 * @class IntrinsicsPyramid
 * @brief Pyramid for intrinsics
 */
class CameraPyramid : public PyramidBase
{
public:
    CameraPyramid(const int w, const int h, const int levels)
        : PyramidBase(w, h, levels)
    {
    }

    CameraPyramid(CameraPyramid&& other)
        : PyramidBase(other.w_, other.h_, other.levels_)
    {
        cameras_.clear();
        cameras_.insert(cameras_.end(),
            std::make_move_iterator(other.cameras_.begin()),
            std::make_move_iterator(other.cameras_.end()));
        other.cameras_.clear();
    }

    void build(const CameraPtr cam)
    {
        cameras_.clear();     

        int w = w_;
        int h = h_;
        cameras_.push_back(Camera::resizeCamera(cam, w, h));
        for (int l = 1; l < levels_; l++)
        {
            w /= 2;
            h /= 2;
            auto last = cameras_.at(l - 1);
            cameras_.push_back(Camera::resizeCamera(last, w, h));
        }
    }

    const CameraPtr at(const int level) const
    {
        assert(level >= 0 && level < levels_);
        return cameras_.at(level);
    }

protected:
    std::vector<CameraPtr> cameras_;
};

/**
 * @class FramePyramid
 * @brief Encapsulates different images
 */

class FramePyramid : public PyramidBase
{
public:
    FramePyramid(const int w, const int h, const int levels)
        : PyramidBase(w, h, levels)
        , color_(w, h, levels, cuimage::ResizeMode::LINEAR)
        , gray_(w, h, levels, cuimage::ResizeMode::LINEAR)
        , depth_(w, h, levels, cuimage::ResizeMode::LINEAR_NONZERO)
        , mask_(w, h, levels, cuimage::ResizeMode::NEAREST)

    {
    }

    FramePyramid(FramePyramid&& other)
        : PyramidBase(other.w_, other.h_, other.levels_)
        , color_(std::move(other.color_))
        , gray_(std::move(other.gray_))
        , depth_(std::move(other.depth_))
        , mask_(std::move(other.mask_))
    {
    }

    void build(const FramePtr frame)
    {
        // Resize to the smalles resolution of frame data
        int hMin = frame->color().height();
        int wMin = frame->color().width();

        if (frame->gray().height() < hMin)
            hMin = frame->gray().height();
        if (frame->depth().height() < hMin)
            hMin = frame->depth().height();
        if (frame->mask().height() < hMin)
            hMin = frame->mask().height();

        if (frame->gray().width() < wMin)
            wMin = frame->gray().width();
        if (frame->depth().width() < wMin)
            wMin = frame->depth().width();
        if (frame->mask().width() < wMin)
            wMin = frame->mask().width();

        // Build color pyramid
        if (frame->color().height() == hMin && frame->color().width() == wMin)
            color_.build(frame->color());
        else
            color_.build(
                frame->color().resized(wMin, hMin, color_.resizeMode()));

        // Build gray pyramid
        if (frame->gray().height() == hMin && frame->gray().width() == wMin)
            gray_.build(frame->gray());
        else
            gray_.build(frame->gray().resized(wMin, hMin, gray_.resizeMode()));

        // Build depth pyramid
        if (frame->depth().height() == hMin && frame->depth().width() == wMin)
            depth_.build(frame->depth());
        else
            depth_.build(
                frame->depth().resized(wMin, hMin, depth_.resizeMode()));

        // Build mask pyramid
        if (frame->mask().height() == hMin && frame->mask().width() == wMin)
            mask_.build(frame->mask());
        else
            mask_.build(frame->mask().resized(wMin, hMin, mask_.resizeMode()));

        timeColor_ = frame->timeColor();
        timeDepth_ = frame->timeDepth();
    }

    const FramePtr at(const int level) const
    {
        assert(level >= 0 && level < levels_);

        return std::make_shared<core::Frame>(color_.at(level), gray_.at(level),
            depth_.at(level), mask_.at(level), timeColor_, timeDepth_);
    }

    void swap(FramePyramid& other)
    {
        color_.swap(other.color_);
        gray_.swap(other.gray_);
        depth_.swap(other.depth_);
        mask_.swap(other.mask_);

        std::swap(timeColor_, other.timeColor_);
        std::swap(timeDepth_, other.timeDepth_);
    }

protected:
    ImagePyramid<float3> color_;
    ImagePyramid<float> gray_;
    ImagePyramid<float> depth_;
    ImagePyramid<uchar> mask_;

    double timeColor_, timeDepth_;
};

} // core

#endif // CORE_PYRAMID_H