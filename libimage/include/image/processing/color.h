/**
 * @file   color.h
 * @brief  Color Processing Interface
 * @author Florian Windolf
 */
#ifndef IMAGE_PROCESSING_COLOR_H
#define IMAGE_PROCESSING_COLOR_H

#include "image/processing/base.h"

#include <type_traits>

using cuimage::has_0_channels;
using cuimage::has_1_channels;
using cuimage::has_2_channels;
using cuimage::has_3_channels;
using cuimage::has_4_channels;

namespace image
{

/**
 * @class ColorProcessing
 * @brief Base class for all color image manipulation
 */
class ColorProcessing : public ProcessingBase<float3>
{
public:
    ColorProcessing(Image<float3> image);
    
    ~ColorProcessing();

    /** 
     * Compute the blurriness measure from Crete 2008
     * https://hal.archives-ouvertes.fr/hal-00232709/document
     */
    float blurriness() const;
};


} // image

#endif // IMAGE_PROCESSING_COLOR_H