#include "image/processing/color.h"

#include "color_cu.h"

using namespace image;

float ColorProcessing::blurriness() const
{
    Image<float> grey = image_.asGray<float>();
    return cu_Blurriness(grey);
}
