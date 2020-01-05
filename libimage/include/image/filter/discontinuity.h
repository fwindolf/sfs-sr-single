/**
 * @file   discontinuity.h
 * @brief  Discontinuity filter
 * @author Florian Windolf
 */
#ifndef IMAGE_FILTER_DISCONTINUITY_H
#define IMAGE_FILTER_DISCONTINUITY_H

#include "image/filter/filter_cu.h"

#include "core/image.h"

namespace image
{

/**
 * @class DiscontinuityFilter
 * @brief Remove area around jumps in depth images
 */
class DiscontinuityFilter
{
public:
    DiscontinuityFilter(const int radius = 3, const float threshold = 10.f)
     : radius_(radius), threshold_(threshold)
    {   
    }
    
    template <typename T>
    void compute(const ImagePtr<T> in, ImagePtr<T>& out)
    {
        assert(in != out);

        cu_DiscontinuityFilter(out, in, radius_, threshold_);
    }

private:    
    const int radius_;
    const float threshold_;
};

} // image

#endif // IMAGE_FILTER_DISCONTINUITY_H