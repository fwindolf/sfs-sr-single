#include "data/dataset_assoc.h"

#include <iomanip>

using namespace data;

DataSetAssoc::DataSetAssoc(const Config& config, const Settings& settings,
    const std::string& assocFileName)
    : DataSetBase(config, settings)
    , assocFileName_(assocFileName)
    , poseFileName_(settings.poseBasename + settings.poseType)
    , fAssoc_(config.directory + assocFileName_)
    , fPoses_(config.directory + poseFileName_)
{
    std::cout << "Opening dataset: " << config.directory + assocFileName_
              << std::endl;

    // Do not read data, only open files
    if (!fAssoc_.is_open())
        throw std::runtime_error("Assoc file at " + config.directory
            + assocFileName_ + " could not be opened...");

    assert(!settings.poseFilePerFrame);

    if (!fPoses_.is_open())
        throw std::runtime_error("Pose file at " + config.directory
            + poseFileName_ + " could not be opened...");

    // Read poses into memory
    _readPoses();

    std::cout << "DatasetAssoc initialized!" << std::endl;
}

bool DataSetAssoc::next()
{
    if (numImages_ > config_.maxImages)
        return false;

    assert(fAssoc_.is_open());

    // Get the next pair of color and depth images
    std::string line;
    std::getline(fAssoc_, line);

    // Break on empty lines
    if (line.empty())
        return false;

    // Skip comments
    if (line.find("#") == 0)
        return next();

    // Parse line
    _readCurrentFiles(line);

    // Find the best matching pose
    assert(timeColor_ - timeDepth_ < 0.02);
    currentTime_ = std::min(timeDepth_, timeColor_);

    currentFrame_.reset();
    currentPose_.setIdentity();

    numImages_++;
    return true;
}

FramePtr DataSetAssoc::frame()
{
    // Read the current frame
    if (!currentFrame_)
        _readCurrentFrame();

    return currentFrame_;
}

Eigen::Matrix4f DataSetAssoc::pose()
{
    if (currentPose_.isIdentity())
        _findPose();

    return currentPose_;
}

void DataSetAssoc::_readPoses()
{
    double time;
    Eigen::Vector3f translation;
    Eigen::Quaternionf quaternion;

    std::string line;
    std::istringstream sline;
    while (std::getline(fPoses_, line))
    {
        if (line.empty() || line.compare(0, 1, "#") == 0)
            continue;

        sline.clear();
        sline.str(line);
        sline >> time >> translation.x() >> translation.y() >> translation.z()
            >> quaternion.x() >> quaternion.y() >> quaternion.z()
            >> quaternion.w();

        auto ret = poses_.emplace(time, Eigen::Matrix4f::Identity());
        if (!ret.second)
        {
            std::cerr << "Invalid pose, pose with same timestamp exists: "
                    + std::to_string(time)
                      << std::endl;
            continue;
        }

        auto& pose = ret.first->second; // ret is (it, true/false)
        pose.block<3, 3>(0, 0) = quaternion.toRotationMatrix();
        pose.block<3, 1>(0, 3) = translation;
    }
}

void DataSetAssoc::_readCurrentFiles(const std::string& line)
{
    std::istringstream sline;
    sline.str(line);

    timeColor_ = 0.f;
    timeDepth_ = 0.f;
    timeMask_ = 0.f;
    fColor_ = "";
    fDepth_ = "";
    fMask_ = "";

    sline >> timeColor_ >> fColor_ >> timeDepth_ >> fDepth_ >> timeMask_
        >> fMask_;
}

void DataSetAssoc::_readCurrentFrame()
{
    if (fMask_.empty())
    {
        currentFrame_ = createFrame(fColor_, timeColor_, fDepth_, timeDepth_);
    }
    else
    {
        currentFrame_
            = createFrame(fColor_, timeColor_, fDepth_, timeDepth_, fMask_);
    }
}

void DataSetAssoc::_findPose()
{
    auto it_bound = poses_.lower_bound(currentTime_);

    if (it_bound == poses_.end())
    {
        // No lower found -> return last
        currentPose_ = std::prev(it_bound)->second;
        return;
    }

    if (it_bound == poses_.begin())
    {
        // Is first -> return first
        currentPose_ = it_bound->second;
        return;
    }

    // Take closer pose
    auto it_lower = std::prev(it_bound);
    if (it_bound->first - currentTime_ < currentTime_ - it_lower->first)
        currentPose_ = it_bound->second;
    else
        currentPose_ = it_lower->second;
    return;

#if 0
    // Interpolate
    auto it_lower = std::prev(it_bound);
    auto it_upper1 = std::next(it_bound, 1);
    auto it_upper2 = std::next(it_bound, 2);

    if (it_upper2 == poses_.end())
        currentPose_ = _interpolatePose(timeCurrent_, *it_lower, *it_bound);
    else
        currentPose_ = _interpolatePose(
            timeCurrent_, *it_lower, *it_bound, it_upper1, it_upper2);
#endif
}
