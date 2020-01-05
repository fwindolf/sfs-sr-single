/**
 * @file   dataset.h
 * @brief  Interface for datasets
 * @author Florian Windolf
 */
#ifndef DATA_DATASET_H
#define DATA_DATASET_H

#include "core/camera.h"
#include "core/frame.h"

namespace data
{

/**
 * @class DataSet
 * @brief Interface for all datasets
 */
class DataSet
{
public:
    /**
     * @struct Config
     * @brief  Settings that are shared across all dataset types
     */
    struct Config
    {
        // Dataset configuration
        std::string directory;
        int width;
        int height;
        int maxImages = 10000;

        bool forceResolution = false;

        float depthThreshold = 2.5f;
    };

    /**
     * @struct Settings
     * @brief  Settings that are different between different datasets
     * Should be overwritten by dataset wrappers, eg when implementing a new
     * type of dataset
     */
    struct Settings
    {
        Settings(std::string colorIntrinsics, std::string depthIntrinsics,
            std::string colorFolder, std::string colorBasename,
            std::string colorType, float3 colorScale, std::string depthFolder,
            std::string depthBasename, std::string depthType, float depthScale,
            std::string maskFolder, std::string maskBasename,
            std::string maskType, bool hasMask, std::string poseFolder,
            std::string poseBasename, std::string poseType,
            bool poseFilePerFrame)
            : colorIntrinsics(colorIntrinsics)
            , depthIntrinsics(depthIntrinsics)
            , colorFolder(colorFolder)
            , colorBasename(colorBasename)
            , colorType(colorType)
            , colorScale(colorScale)
            , depthFolder(depthFolder)
            , depthBasename(depthBasename)
            , depthType(depthType)
            , depthScale(depthScale)
            , maskFolder(maskFolder)
            , maskBasename(maskBasename)
            , maskType(maskType)
            , hasMask(hasMask)
            , poseFolder(poseFolder)
            , poseBasename(poseBasename)
            , poseType(poseType)
            , poseFilePerFrame(poseFilePerFrame)
        {
        }

        // Intrinsics
        std::string colorIntrinsics, depthIntrinsics;

        // Color images
        std::string colorFolder, colorBasename, colorType;
        float3 colorScale;

        // Depth images
        std::string depthFolder, depthBasename, depthType;
        float depthScale;

        // Mask images
        std::string maskFolder, maskBasename, maskType;
        bool hasMask;

        // Poses
        std::string poseFolder, poseBasename, poseType;
        bool poseFilePerFrame;
    };

    virtual ~DataSet(){};

    /**
     * Advance to and load the next frame returns false if no more frames
     * Implementations need to check if maxImages is bigger than numImages and
     * increment numImages accordingly
     */
    virtual bool next() = 0;

    /**
     * Get the current timestamp
     */
    virtual double time() = 0;

    /**
     * Get the current frame
     */
    virtual FramePtr frame() = 0;

    /**
     * Get the current pose
     */
    virtual Eigen::Matrix4f pose() = 0;

    /**
     * Get the camera for this dataset
     */
    virtual CameraPtr camera() = 0;
};

class DataSetBase : public DataSet
{
public:
    DataSetBase(const Config& config, const Settings& settings);

    ~DataSetBase(){};

    virtual double time() override;

    virtual FramePtr frame() override;

    virtual Eigen::Matrix4f pose() override;

    virtual CameraPtr camera() override;

protected:
    /**
     * Generate a frame from the specified file names
     */
    virtual FramePtr createFrame(const std::string& color,
        const float& colorTime, const std::string& depth,
        const float& depthTime, const std::string& mask);

    /**
     * Generate a frame from the file names
     */
    virtual FramePtr createFrame(const std::string& color,
        const float& colorTime, const std::string& depth,
        const float& depthTime);

    /**
     * Interpolate the pose from 2 neighbors
     */
    virtual Eigen::Matrix4f _interpolatePose(const double time,
        const std::pair<double, Eigen::Matrix4f>& lower,
        const std::pair<double, Eigen::Matrix4f>& upper);

    /**
     * Interpolate the pose from 4 neighbors
     */
    virtual Eigen::Matrix4f _interpolatePose(const double time,
        const std::pair<double, Eigen::Matrix4f>& lower,
        const std::pair<double, Eigen::Matrix4f>& upper,
        const std::pair<double, Eigen::Matrix4f>& upper1,
        const std::pair<double, Eigen::Matrix4f>& upper2);

    const DataSet::Config config_;
    const DataSet::Settings settings_;

    int numImages_;

    double currentTime_;
    FramePtr currentFrame_;
    Eigen::Matrix4f currentPose_;

    IntrinsicsPtr colorIntrinsics_;
    IntrinsicsPtr depthIntrinsics_;
    CameraPtr camera_;
};

} // data

#endif // DATA_DATASET_H