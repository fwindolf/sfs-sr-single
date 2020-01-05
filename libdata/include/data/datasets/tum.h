/**
 * @file   tum.h
 * @brief  Dataset wrapper for TUM-RGBD dataset based on dataset_assoc.h
 * @author Florian Windolf
 */
#ifndef DATA_DATASET_TUM_H
#define DATA_DATASET_TUM_H

#include "data/dataset_assoc.h"

namespace data
{

/**
 * @class TumRGBDDataSet
 * @brief Dataset wrapper for TUM-RGBD dataset
 * https://vision.in.tum.de/data/datasets/rgbd-dataset
 */
class TumRGBDDataSet : public DataSetAssoc
{
public:
    TumRGBDDataSet(const Config& config);

    /**
     * Save vectors of poses and timestamps in the dataset format
     */
    void savePoses(const std::string& fileName,
        const std::vector<Eigen::Matrix4f>& poses,
        const std::vector<double>& timestamps);

private:
    virtual void _readCurrentFrame() override;
    const int w_, h_;
};

} // data

#endif // DATA_DATASET_TUM_H