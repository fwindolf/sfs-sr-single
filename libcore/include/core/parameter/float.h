/**
 * @file   float.h
 * @brief  Namedparameter for float
 * @author Florian Windolf
 */
#ifndef CORE_PARAMETER_FLOAT_H
#define CORE_PARAMETER_FLOAT_H

#include "core/parameter/base.h"

namespace core
{

/**
 * @class  FloatParameter
 * @brief  NamedParameter that encapsules an float (copy on heap) 
 */
class FloatParameter : public ParameterBase
{
public:
    FloatParameter(float data);

    ~FloatParameter();

    void* data() const override;

    ParameterBase* copy() const override;
    
private:
    float* data_;
};

} // core

#endif // CORE_PARAMETER_FLOAT_H
