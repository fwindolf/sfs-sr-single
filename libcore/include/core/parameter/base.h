/**
 * @file   parameters.h
 * @brief  Named and ordered parameters that provide void* data access
 * @author Florian Windolf
 */
#ifndef CORE_PARAMETER_BASE_H
#define CORE_PARAMETER_BASE_H

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "core/image.h"

namespace core 
{

/**
 * @class  ParameterBase
 * @brief  Access the data of any parameter via common interface
 * @author Florian Windolf
 */
class ParameterBase
{
public:  
    virtual ~ParameterBase(){};

    virtual void* data() const = 0;
    
    virtual ParameterBase* copy() const = 0;
};

} // core

#endif // CORE_UTIL_PARAMETER_H

