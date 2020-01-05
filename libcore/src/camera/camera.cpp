#include "core/camera/camera.h"

using namespace core;

Camera::Camera()
    : colorIntrinsics_(new Intrinsics())
    , depthIntrinsics_(new Intrinsics())
{
}

Camera::Camera(const IntrinsicsPtr intrinsics)
    : colorIntrinsics_(intrinsics)
    , depthIntrinsics_(intrinsics)
{
}

Camera::Camera(const IntrinsicsPtr color, const IntrinsicsPtr depth)
    : colorIntrinsics_(color)
    , depthIntrinsics_(depth)
{
}

CameraPtr Camera::create(const std::string& directory,
    const std::string& colorName, const std::string& depthName)
{
    return std::make_shared<Camera>(
        std::make_shared<Intrinsics>(directory + colorName),
        std::make_shared<Intrinsics>(directory + depthName));
}

CameraPtr Camera::create(const std::string& directory,
    const std::string& fileName)
{
    auto intrinsics = std::make_shared<Intrinsics>(directory + fileName);
    return std::make_shared<Camera>(intrinsics, intrinsics);
}

CameraPtr Camera::create(const int w, const int h, const float fx,
    const float fy, const float cx, const float cy)
{
    return std::make_shared<Camera>(
        std::make_shared<Intrinsics>(w, h, fx, fy, cx, cy),
        std::make_shared<Intrinsics>(w, h, fx, fy, cx, cy));
}

CameraPtr Camera::resizeCamera(
    const CameraPtr camera, const int w, const int h)
{
    assert(camera);

    auto cI = camera->colorIntrinsics();
    auto dI = camera->depthIntrinsics();

    float factorX = w / (float)cI->width();
    float factorY = h / (float)cI->height();
    auto cI_resized = std::make_shared<Intrinsics>(w, h, cI->fx() * factorX,
        cI->fy() * factorY, (cI->cx() + .5f) * factorX - .5f,
        (cI->cy() + .5f) * factorY - .5f);

    factorX = w / (float)dI->width();
    factorY = h / (float)dI->height();
    auto dI_resized = std::make_shared<Intrinsics>(w, h, dI->fx() * factorX,
        dI->fy() * factorY, (dI->cx() + .5f) * factorX - .5f,
        (dI->cy() + .5f) * factorY - .5f);

    return std::make_shared<Camera>(cI_resized, dI_resized);
}

IntrinsicsPtr Camera::colorIntrinsics() const { return colorIntrinsics_; }

IntrinsicsPtr Camera::depthIntrinsics() const { return depthIntrinsics_; }

void Camera::setIntrinsics(IntrinsicsPtr color, IntrinsicsPtr depth)
{
    colorIntrinsics_ = color;
    depthIntrinsics_ = depth;
}

void Camera::print() const
{
    colorIntrinsics_->print();
    depthIntrinsics_->print();
}