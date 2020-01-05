
/**
 * @file   image.h
 * @brief  Namedparameter for image
 * @author Florian Windolf
 */
#ifndef CORE_PARAMETER_IMAGE_H
#define CORE_PARAMETER_IMAGE_H

#include "core/image.h"
#include "core/parameter/base.h"

namespace core
{

/**
 * @class  ImageParameter
 * @brief  NamedParameter that encapsules a shared Image
 */
template <typename T>
class ImageParameter : public ParameterBase
{
public:
    ImageParameter(const Image<T>& image);

    ~ImageParameter();

    void* data() const override;

    ParameterBase* copy() const override;
    
private:
    const Image<T>& image_;
};

/**
 * Implementation
 */

template <typename T>
ImageParameter<T>::ImageParameter(const Image<T>& image)
 : image_(image)
{
    assert(!image_.empty());
}

template <typename T>
ImageParameter<T>::~ImageParameter()
{
}

template <typename T>
void* ImageParameter<T>::data() const
{
    return (void*)image_.data();
}

template <typename T>
ParameterBase* ImageParameter<T>::copy() const
{
    return new ImageParameter<T>(image_);
}


} // core

#endif // CORE_PARAMETER_IMAGE_H
