/**
 * @file   pose.h
 * @brief  3D transformations
 * @author Florian Windolf
 */
#ifndef CORE_POSE_H
#define CORE_POSE_H

#include "Eigen/Dense"
#include "sophus/se3.hpp"

namespace Eigen
{
typedef Matrix<float, 6, 1> Vector6f;
}

namespace core
{

/**
 * @class Pose
 * @brief 3D transformation (R, T) with associated timestamp
 */
class TimedPose : Eigen::Matrix4f
{
public:
    TimedPose() {}
    ~TimedPose(){};

    double timestamp() const;

    Eigen::Matrix3f rotation() const;

    Eigen::Vector3f translation() const;

    Eigen::Vector6f xi() const;

    static Eigen::Matrix4f interpolate(std::vector<double>& times,
        std::vector<Eigen::Matrix4f>& poses, double time)
    {
        // -1, 0, 1, 2
        assert(times.size() == 4);
        assert(poses.size() == 4);

        // between 0, 1
        assert(time > times.at(1));
        assert(time < times.at(2));

        const double t = (time - times.at(1)) / (times.at(1) - times.at(0));

        // B(t) = (6 0 0 0; 5 3 -3 1; 1 3 3 -2; 0 0 0 1) * (1 t  t² t³)^T
        static auto B = (Eigen::Matrix4f() << 6, 0, 0, 0, 5, 3, -3, 1, 1, 3, 3,
            -2, 0, 0, 0, 1)
                            .finished();

        const Eigen::Vector4f Bt = B * Eigen::Vector4f(1, t, t * t, t * t * t);

        // T(t) = T(-1) Product(k=0..2)[(exp(Bk+1(t)log(T⁻¹(k-1)T(k)))]
        auto T = poses.at(0);
        for (int k = 0; k <= 2; k++)
        {
            Sophus::SE3f T_rel(poses.at(k).inverse() * poses.at(k + 1));
            T *= Sophus::SE3f::exp(Bt[k + 1] * Sophus::SE3f::log(T_rel))
                     .matrix();
        }

        return T;
    }

private:
    double timestamp;
};

} // core

#endif // CORE_POSE_H