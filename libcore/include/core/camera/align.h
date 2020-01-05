/**
 * @file   align.h
 * @brief  Alignment and projectsions using (pinhole) cameras
 * @author Florian Windolf
 */
#ifndef CORE_CAMERA_ALIGN_H
#define CORE_CAMERA_ALIGN_H

#include "core/camera/camera.h"

namespace core
{

/**
 * @class Alignment
 * @brief Wraps a camera model to provide align/projection functionality
 */
class Alignment
{
public:
    Alignment(const CameraPtr camera);

    ~Alignment(){};

    IntrinsicsPtr colorIntrinsics() const;

    IntrinsicsPtr depthIntrinsics() const;

    /**
     * Project a 3D point into the depth image (using depth intrinsics)
     */
    virtual Eigen::Vector3f projectDepth(const Eigen::Vector4f& point) const = 0;

    /**
     * Project a 3D point into the color image (using color intrinsics)
     */
    virtual Eigen::Vector3f projectColor(const Eigen::Vector4f& point) const = 0;

    /**
     * Backproject a point into 3D using its depth (using depth intrinsics)
     * Returns the result in homogeneous coordinates
     */
    virtual Eigen::Vector4f backProjectDepth(const float x , const float y, const float depth) const = 0;

    /**
     * Check if a point is inside the color image
     */
    virtual bool insideColor(const Eigen::Vector3f& point) const = 0;

    /**
     * Check if a point is inside the depth image
     */
    virtual bool insideDepth(const Eigen::Vector3f& point) const = 0;

private:
    CameraPtr camera_;
};


} // core

#endif // CORE_CAMERA_ALIGN_H