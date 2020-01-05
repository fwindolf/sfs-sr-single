#include "data/datasets/tum.h"

#include <iomanip>

using namespace data;
using namespace core;

static DataSet::Settings tumRGBDSettings(
    // Intrinsics
    "camera_color.txt", "camera_depth.txt",
    // Color
    "rgb/", "color_", ".png", make_float3(1.f, 1.f, 1.f),
    // Depth
    "depth/", "depth_", ".png", 1.f / 5000.f,
    // Mask
    "mask/", "mask_", ".png", false,
    // Poses
    "", "groundtruth", ".txt", false);

TumRGBDDataSet::TumRGBDDataSet(const Config& config)
    : DataSetAssoc(config, tumRGBDSettings, "assoc.txt")
    , w_(config.width)
    , h_(config.height)
{
    assert(camera_->colorIntrinsics());
    assert(camera_->depthIntrinsics());

    camera_->colorIntrinsics()->print();
}

void TumRGBDDataSet::savePoses(const std::string& fileName,
    const std::vector<Eigen::Matrix4f>& poses,
    const std::vector<double>& timestamps)
{
    assert(!fileName.empty());

    std::ofstream outFile;
    outFile.open(fileName.c_str());
    if (!outFile.is_open())
    {
        std::cerr << "Could not write to " << fileName << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(6);
    assert(poses.size() == timestamps.size());

    for (size_t i = 0; i < poses.size(); i++)
    {
        Eigen::Matrix4f pose = poses.at(i);

        // write into evaluation file
        // timestamp
        double timestamp = timestamps.at(i);
        outFile << timestamp << " ";

        // translation

        Eigen::Vector3f T(pose.block<3, 1>(0, 3));
        outFile << T.x() << " " << T.y() << " " << T.z();

        Eigen::Quaternionf Q(pose.block<3, 3>(0, 0));
        outFile << " " << Q.x() << " " << Q.y() << " " << Q.z() << " " << Q.w()
                << std::endl;
    }

    outFile.close();
}

void TumRGBDDataSet::_readCurrentFrame()
{
    // Load and preprocess color image
    Image<float3> color_(config_.directory + fColor_);
    color_ *= settings_.colorScale;
    if (color_.width() != w_ || color_.height() != h_)
        color_.resize(config_.width, config_.height, ResizeMode::LINEAR);

    Image<float> depth_(config_.directory + fDepth_);
    depth_ *= settings_.depthScale;
    if (depth_.width() != w_ || depth_.height() != h_)
        depth_.resize(w_, h_, ResizeMode::LINEAR_VALID);

    if (config_.depthThreshold > 0.f)
        depth_.threshold(config_.depthThreshold,
            0.f); // Set all depths above threshold to invalid

    Image<uchar> mask_(w_, h_, 255.f);

    currentFrame_ = std::make_shared<FrameContainer>(std::move(color_),
        std::move(depth_), std::move(mask_), timeColor_, timeDepth_);
}