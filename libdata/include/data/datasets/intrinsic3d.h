/**
 * @file   intrinsic3d.h
 * @brief  Loader for the Intrinsic3d dataset
 * @author Florian Windolf
 */
#ifndef DATA_DATASETS_INTRINSIC3D_H
#define DATA_DATASETS_INTRINSIC3D_H

#include "data/dataset_index.h"

namespace data
{

/**
 * @class Intrinsic3dDataSet
 * @brief DataSet wrapper for Intrinsic3D dataset
 * https://vision.in.tum.de/data/datasets/intrinsic3d
 */
class Intrinsic3dDataSet : public DataSetIndexBased
{
public:
    Intrinsic3dDataSet(const Config& config);

    ~Intrinsic3dDataSet(){};

private:
    virtual Eigen::Matrix4f _readPose() override;
};
    

} // data

#endif // DATA_DATASETS_INTRINSIC3D_H