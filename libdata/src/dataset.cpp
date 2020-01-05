#include "data/dataset.h"

#include <sophus/se3.hpp>

using namespace core;
using namespace data;

DataSetBase::DataSetBase(const Config& config, const Settings& settings)
    : config_(config)
    , settings_(settings)
    , numImages_(0)
    , currentTime_(0)
    , currentFrame_(nullptr)
    , currentPose_(Eigen::Matrix4f::Identity())
{
    // Read intrinisics
    try
    {
        camera_ = Camera::create(config_.directory, settings_.colorIntrinsics,
            settings_.depthIntrinsics);
    }
    catch (std::exception& e)
    {
        colorIntrinsics_.reset();
        depthIntrinsics_.reset();
        camera_.reset();
        std::cerr << "Could not create camera!" << config_.directory
                  << settings_.colorIntrinsics << " "
                  << settings_.depthIntrinsics << std::endl;
    }
}

double DataSetBase::time() { return currentTime_; }

FramePtr DataSetBase::frame() { return currentFrame_; }

Eigen::Matrix4f DataSetBase::pose() { return currentPose_; }

CameraPtr DataSetBase::camera() { return camera_; }

FramePtr DataSetBase::createFrame(const std::string& colorName,
    const float& colorTime, const std::string& depthName,
    const float& depthTime, const std::string& maskName)
{
    Image<float3> color(config_.directory + colorName);
    Image<float> depth(config_.directory + depthName);
    Image<uchar> mask(config_.directory + maskName);

    color *= settings_.colorScale;
    depth *= settings_.depthScale;

    if (config_.forceResolution)
    {
        color.resize(config_.width, config_.height, cuimage::LINEAR);
        depth.resize(config_.width, config_.height, cuimage::LINEAR_NONZERO);
        mask.resize(config_.width, config_.height, cuimage::NEAREST);
    }

    return std::make_shared<core::FrameContainer>(std::move(color),
        std::move(depth), std::move(mask), colorTime, depthTime);
}

FramePtr DataSetBase::createFrame(const std::string& colorName,
    const float& colorTime, const std::string& depthName,
    const float& depthTime)
{
    Image<float3> color(config_.directory + colorName);
    Image<float> depth(config_.directory + depthName);
    Image<float> depth_mask = depth;
    depth_mask.threshold(0.f, 0.f, 1.f); // Everything that has a valid
                                         // value is set to 1, else to 0
    Image<uchar> mask = depth_mask.as<uchar>() * (uchar)255;

    color *= settings_.colorScale;
    depth *= settings_.depthScale;

    if (config_.forceResolution)
    {
        color.resize(config_.width, config_.height, cuimage::LINEAR);
        depth.resize(config_.width, config_.height, cuimage::LINEAR_NONZERO);
        mask.resize(config_.width, config_.height, cuimage::NEAREST);
    }

    return std::make_shared<core::FrameContainer>(
        std::move(color), std::move(depth), std::move(mask), colorTime, depthTime);
}

Eigen::Matrix4f DataSetBase::_interpolatePose(const double time,
    const std::pair<double, Eigen::Matrix4f>& lower,
    const std::pair<double, Eigen::Matrix4f>& upper)
{
    const double t = (time - upper.first) / (upper.first - lower.first);

    static auto B = (Eigen::Matrix2f() << 1, 0, 0, 1).finished();
    const Eigen::Vector2f Bt = B * Eigen::Vector2f(1, t);

    auto T = lower.second;
    T *= Sophus::SE3f::exp(Bt[1]
        * Sophus::SE3f::log(
              Sophus::SE3f(lower.second.inverse() * upper.second)))
             .matrix();
    return T;
}

Eigen::Matrix4f DataSetBase::_interpolatePose(const double time,
    const std::pair<double, Eigen::Matrix4f>& lower,
    const std::pair<double, Eigen::Matrix4f>& upper,
    const std::pair<double, Eigen::Matrix4f>& upper1,
    const std::pair<double, Eigen::Matrix4f>& upper2)
{
    return lower.second;

    const double t = (time - upper.first) / (upper.first - lower.first);

    // B(t) = (6 0 0 0; 5 3 -3 1; 1 3 3 -2; 0 0 0 1) * (1 t  t² t³)^T
    static auto B = (Eigen::Matrix4f() << 6, 0, 0, 0, 5, 3, -3, 1, 1, 3, 3, -2,
        0, 0, 0, 1)
                        .finished();

    const Eigen::Vector4f Bt = B * Eigen::Vector4f(1, t, t * t, t * t * t);

    // T(t) = T(-1) Product(k=0..2)[(exp(Bk+1(t)log(T⁻¹(k-1)T(k)))]
    auto T = lower.second;

    // For 0 - 2
    T *= Sophus::SE3f::exp(Bt[1]
        * Sophus::SE3f::log(
              Sophus::SE3f(lower.second.inverse() * upper.second)))
             .matrix();
    T *= Sophus::SE3f::exp(Bt[2]
        * Sophus::SE3f::log(
              Sophus::SE3f(upper.second.inverse() * upper1.second)))
             .matrix();
    T *= Sophus::SE3f::exp(Bt[3]
        * Sophus::SE3f::log(
              Sophus::SE3f(upper1.second.inverse() * upper2.second)))
             .matrix();

    return T;
}