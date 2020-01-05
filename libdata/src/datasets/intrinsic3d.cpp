#include "data/datasets/intrinsic3d.h"

using namespace data;

const int precision = 6; // 6 places for index with leading zeros

static DataSet::Settings intrinsic3dSettings(
    // Intrinsics
    "colorIntrinsics.txt", "depthIntrinsics.txt",
    // Color
    "", "frame-", ".color.png", make_float3(1.f, 1.f, 1.f),
    // Depth
    "", "frame-", ".depth.png", 1/1000.f,
    // Mask
    "", "", "", false,
    // Poses
    "", "frame-", ".pose.txt", true);

Intrinsic3dDataSet::Intrinsic3dDataSet(const Config& config)
    : DataSetIndexBased(config, intrinsic3dSettings, precision)
{
}

Eigen::Matrix4f Intrinsic3dDataSet::_readPose()
{
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();

    // Assumes a 4x4 matrix
    for (int i = 0; i < pose.rows(); i++)
    {
        for (int j = 0; j < pose.cols(); j++)
        {
            fPose_ >> pose(i, j);
        }
    }
    return pose;
}