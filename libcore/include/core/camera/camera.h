/**
 * @file   camera.h
 * @brief  Camera model
 * @author Florian Windolf
 */
#ifndef CORE_CAMERA_CAMERA_H
#define CORE_CAMERA_CAMERA_H

#include "core/camera/intrinsics.h"

namespace core
{

/**
 * @class Camera
 * @brief Pinhole camera model
 */
class Camera
{
public:
    Camera();

    Camera(const IntrinsicsPtr intrinsics);

    Camera(const IntrinsicsPtr color, const IntrinsicsPtr depth);

    ~Camera(){};

    /**
     * Create a camera from color and depth intrinsics txt file
     */
    static std::shared_ptr<Camera> create(const std::string& directory,
        const std::string& colorName, const std::string& depthName);

    /**
     * Create a camera with identical depth and color intrisics from txt file
     */
    static std::shared_ptr<Camera> create(
        const std::string& directory, const std::string& fileNamee);

    /**
     * Create a camera from manual values
     */
    static std::shared_ptr<Camera> create(const int w, const int h,
        const float fx, const float fy, const float cx, const float cy);

    /**
     * Resize a both intrinsics of a camera to a different resolution
     */
    static std::shared_ptr<core::Camera> resizeCamera(
        const std::shared_ptr<core::Camera> camera, const int w, const int h);

    /**
     * Set new intrinsics for this camera
     */
    void setIntrinsics(const IntrinsicsPtr color, const IntrinsicsPtr depth);

    IntrinsicsPtr colorIntrinsics() const;

    IntrinsicsPtr depthIntrinsics() const;

    void print() const;

private:
    IntrinsicsPtr colorIntrinsics_;
    IntrinsicsPtr depthIntrinsics_;

    // Eigen::Matrix4f registrationPose_;
};

} // core

typedef std::shared_ptr<core::Camera> CameraPtr;

#endif // CORE_CAMERA_CAMERA_H