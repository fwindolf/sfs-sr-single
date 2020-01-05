/**
 * @file   base.h
 * @brief  Image processing base implementation
 * @author Florian Windolf
 */
#ifndef IMAGE_PROCESSING_BASE_H
#define IMAGE_PROCESSING_BASE_H

#include "core/image.h"

namespace image
{

template <typename T>
class ProcessingBase
{
public:
    ProcessingBase(const Image<T>& image) 
     : image_(image)
    {
        assert(!image_.empty());
    }
    
    ~ProcessingBase() {};

protected:
    const Image<T>& image_;
};

} // image

#endif // IMAGE_PROCESSING_BASE_H