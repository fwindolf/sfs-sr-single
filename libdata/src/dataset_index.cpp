#include "data/dataset_index.h"

#include <iomanip>

using namespace data;

DataSetIndexBased::DataSetIndexBased(
    const Config& config, const Settings& settings, const int precision)
    : DataSetBase(config, settings)
    , index_(0)
    , precision_(precision)
{
    std::cout << "Opening dataset: " << config.directory << std::endl;

    if (!settings_.poseFilePerFrame)
    {
        std::stringstream poseFile;
        poseFile << settings_.poseFolder << settings_.poseBasename
                 << settings.poseType;
        fPose_.open(poseFile.str());
    }
}

bool DataSetIndexBased::next()
{
    int precision = 6;

    if (numImages_ > config_.maxImages)
        return false;

    _readCurrentFrame();
    _readCurrentPose();

    numImages_++;
    index_++;
    return true;
}

FramePtr DataSetIndexBased::frame() { return currentFrame_; }

Eigen::Matrix4f DataSetIndexBased::pose() { return currentPose_; }

void DataSetIndexBased::_readCurrentFrame()
{
    std::stringstream colorFile, depthFile, maskFile;
    colorFile << settings_.colorFolder << settings_.colorBasename
              << std::setw(precision_) << std::setfill('0') << index_
              << settings_.colorType;

    depthFile << settings_.depthFolder << settings_.depthBasename
              << std::setw(precision_) << std::setfill('0') << index_
              << settings_.depthType;

    maskFile << settings_.maskFolder << settings_.maskBasename
             << std::setw(precision_) << std::setfill('0') << index_
             << settings_.maskType;

    if (settings_.hasMask)
    {
        currentFrame_ = createFrame(
            colorFile.str(), index_, depthFile.str(), index_, maskFile.str());
    }
    else
    {
        currentFrame_
            = createFrame(colorFile.str(), index_, depthFile.str(), index_);
    }
}

void DataSetIndexBased::_readCurrentPose()
{
    if (settings_.poseFilePerFrame)
    {
        std::stringstream poseFile;
        poseFile << settings_.poseFolder << settings_.poseBasename
                 << std::setw(precision_) << std::setfill('0') << index_
                 << settings_.poseType;

        fPose_.open(config_.directory + poseFile.str());
        // std::cout << config_.directory << poseFile.str() << std::endl;
        assert(fPose_.is_open());
    }

    // Parse the file (entry)
    currentPose_ = _readPose();
    fPose_.close();
}

Eigen::Matrix4f DataSetIndexBased::_readPose()
{
    if (settings_.poseFilePerFrame)
    {
        // Go to 0 and forward to line index
        fPose_.seekg(0, std::ios::beg);
        std::string line;
        int i = 0;
        while (i < index_)
        {
            fPose_ >> line;
            if (line.find("#") == 0 || line.size() == 0)
                continue;

            i++;
        }
    }

    // Assumes a row with TX TY TZ QX QY QZ QW
    Eigen::Matrix4f pose;
    Eigen::Vector3f translation;
    Eigen::Quaternionf quaternion;

    fPose_ >> translation.x() >> translation.y() >> translation.z()
        >> quaternion.x() >> quaternion.y() >> quaternion.z()
        >> quaternion.w();

    pose.block<3, 3>(0, 0) = quaternion.normalized().toRotationMatrix();
    pose.block<3, 1>(0, 3) = translation;

    return pose;
}