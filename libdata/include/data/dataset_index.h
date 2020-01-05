/**
 * @file   dataset_index.h
 * @brief  Base class to load frames from dataset using continuous indices
 * @author Florian Windolf
 */
#ifndef DATA_DATASET_INDEX_H
#define DATA_DATASET_INDEX_H

#include "dataset.h"

namespace data
{

/**
 * @class DataSetIndexBased
 * @brief DataSet base class for index based data
 */
class DataSetIndexBased : public DataSetBase
{
public:
    DataSetIndexBased(const DataSet::Config& config,
        const DataSet::Settings& settings, const int precision);

    virtual bool next() override;

    virtual FramePtr frame() override;

    virtual Eigen::Matrix4f pose() override;

protected:
    /**
     * Fill currentFrame_ from files with index
     */
    virtual void _readCurrentFrame();

    /**
     * Fill currentPose_ from file with index
     */
    virtual void _readCurrentPose();

    /**
     * Read a pose file (entry)
     */
    virtual Eigen::Matrix4f _readPose();

    int index_;
    int precision_; // For index formatting with leading zeros

    std::ifstream fPose_;
};

} // data

#endif // DATA_DATASET_INDEX_H