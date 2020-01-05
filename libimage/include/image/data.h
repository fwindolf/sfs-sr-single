/**
 * @file   data.h
 * @brief  Test data class
 * @author Florian Windolf
 */
#ifndef IMAGE_DATA_H
#define IMAGE_DATA_H

#include "image/processing.h"

namespace image
{

class DataSet
{
public:
    DataSet(const std::string path, const int w, const int h, const int num = 0, bool usecolor=false);

    float fx, fy, cx, cy;
    float l1, l2, l3, l4;

    IntrinsicsPtr K_lr, K_hr;
    CameraPtr cam_lr, cam_hr;

    Image<float> depth_lr, depth_star;
    Image<uchar> mask_lr, mask;
    Image<float3> albedo, shading, light, image, color;

    bool preferAlbedo_;
};

} // image

#endif // IMAGE_DATA_H