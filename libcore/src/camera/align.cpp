#include "core/camera/align.h"

using namespace core;

Alignment::Alignment(const CameraPtr camera)
 : camera_(camera)
{
    if (!camera->colorIntrinsics() || !camera->depthIntrinsics())
        throw std::runtime_error("Cannot create Alignment with incomplete camera information!");
}

IntrinsicsPtr Alignment::colorIntrinsics() const
{
    return camera_->colorIntrinsics();
}

IntrinsicsPtr Alignment::depthIntrinsics() const
{
    return camera_->depthIntrinsics();
}

Eigen::Vector3f Alignment::projectDepth(const Eigen::Vector4f &point) const
{
    const auto depthIntrinsics = camera_->depthIntrinsics();

    const Eigen::Vector3f point3 = { point[0], point[1], point[2] };
    const Eigen::Vector3f point2Homogeneous = depthIntrinsics->matrix() * point3;
    const float zInverse = 1.0f / point2Homogeneous[2];

    return { point2Homogeneous[0] * zInverse, point2Homogeneous[1] * zInverse, 1.f };
}

Eigen::Vector3f Alignment::projectColor(const Eigen::Vector4f &point) const
{
    const auto colorIntrinsics = camera_->colorIntrinsics();

    const Eigen::Vector3f point3 = { point[0], point[1], point[2] };
    const Eigen::Vector3f point2Homogeneous = colorIntrinsics->matrix() * point3;
    const float zInverse = 1.0f / point2Homogeneous[2];

    return { point2Homogeneous[0] * zInverse, point2Homogeneous[1] * zInverse, 1.f };
}

Eigen::Vector4f Alignment::backProjectDepth(const float x , const float y, const float depth) const
{
    const auto depthIntrinsics = camera_->depthIntrinsics();
    const Eigen::Vector2f point2 = { x , y };

    const float x0 = (x - depthIntrinsics->cx() ) / depthIntrinsics->fx();
    const float y0 = (y - depthIntrinsics->cy() ) / depthIntrinsics->fy();
    
    return { x0 * depth, y0 * depth, depth, 1.f };
}

bool Alignment::insideColor(const Eigen::Vector3f& point) const
{
    const auto colorIntrinsics = camera_->colorIntrinsics();

    if (point[0] >= colorIntrinsics->width())
        return false;

    if (point[1] >= colorIntrinsics->height())
        return false;

    return true;
}

bool Alignment::insideDepth(const Eigen::Vector3f& point) const 
{
    const auto depthIntrinsics = camera_->depthIntrinsics();

    if (point[0] >= depthIntrinsics->width())
        return false;

    if (point[1] >= depthIntrinsics->height())
        return false;

    return true;
}