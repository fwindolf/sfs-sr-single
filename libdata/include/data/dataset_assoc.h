/**
 * @file   dataset.h
 * @brief  Base class to load frames from dataset using assoc files
 * @author Florian Windolf
 */
#ifndef DATA_DATASET_ASSOC_H
#define DATA_DATASET_ASSOC_H

#include "dataset.h"

namespace data
{

/**
 * @class DataSetAssoc
 * @brief DataSet base class for datasets using assoc files
 */
class DataSetAssoc : public DataSetBase
{
public:
    DataSetAssoc(const DataSet::Config& config,
        const DataSet::Settings& settings, const std::string& assocFileName);

    virtual bool next() override;

    virtual FramePtr frame() override;

    virtual Eigen::Matrix4f pose() override;

protected:
    /**
     * Fill poses_
     */
    virtual void _readPoses();

    /**
     * Fill timeColor_, ... and fColor_, ...
     */
    virtual void _readCurrentFiles(const std::string& line);

    /**
     * Fill currentFrame_ from fColor_, fDepth_, ...
     */
    virtual void _readCurrentFrame();

    /**
     * Fill currentPose_ from timeCurrent_
     */
    virtual void _findPose();

    std::string assocFileName_;
    std::ifstream fAssoc_;

    std::string poseFileName_;
    std::ifstream fPoses_;

    double timeColor_, timeDepth_, timeMask_;
    std::string fColor_, fDepth_, fMask_;

    std::map<double, Eigen::Matrix4f> poses_;
};

} // data

#endif // DATA_DATASET_ASSOC_H