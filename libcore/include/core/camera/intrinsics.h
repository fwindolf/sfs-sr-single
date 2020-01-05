/**
 * @file   intrinsics.h
 * @brief  Specifies the different types of intrinsic camera parameters
 * @author Florian Windolf
 */
#ifndef CORE_CAMERA_INTRINSICS_H
#define CORE_CAMERA_INTRINSICS_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include <memory>
#include <iostream>

namespace core
{

/**
 * @class Intrinsics
 * @brief Interface for intrinsic camera parameters
 */
class Intrinsics
{
public:
    Intrinsics();

    Intrinsics(const std::string& fileName);

    Intrinsics(const int width, const int height);

    Intrinsics(const int width, const int height, const Eigen::Matrix3f& K);

    Intrinsics(const int width, const int height, const float fx, const float fy, const float cx, const float cy);

    ~Intrinsics();

    void set(const int width, const int height, const Eigen::Matrix3f& K);

    void setDimensions(const int width, const int height);

    void setMatrix(const Eigen::Matrix3f& K);

    void setFocalLength(const float fx, const float fy);

    void setCenter(const float cx, const float cy);

    Eigen::Matrix3f matrix() const;

    Eigen::Matrix3f inverse() const;

    int width() const;

    int height() const;

    float fx() const;

    float fy() const;
    
    float cx() const;

    float cy() const;

    virtual void print() const;

protected:
    int width_;
    int height_;
    Eigen::Matrix3f matrix_;
};

} // core

typedef std::shared_ptr<core::Intrinsics> IntrinsicsPtr;

#endif // CORE_CAMERA_INTRINSICS_H