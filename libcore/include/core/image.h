/**
 * @file   image.h
 * @brief  Image related core functionality
 * @author Florian Windolf
 */
#ifndef CORE_IMAGE_H
#define CORE_IMAGE_H

#include <cuimage/image.h>
#include <cuimage/cuda/type.h>

namespace core
{

} // core

template <typename T>
using Image = cuimage::Image<T>;

using cuimage::uchar;

#endif // CORE_IMAGE_H