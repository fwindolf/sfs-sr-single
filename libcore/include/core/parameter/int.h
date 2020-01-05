/**
 * @file   int.h
 * @brief  Namedparameter for ints
 * @author Florian Windolf
 */
#ifndef CORE_PARAMETER_INT_H
#define CORE_PARAMETER_INT_H

#include "core/parameter/base.h"

namespace core
{

/**
 * @class  IntParameter
 * @brief  NamedParameter that encapsules an int (copy on heap) 
 */
class IntParameter : public ParameterBase
{
public:
    IntParameter(int data);

    ~IntParameter();

    void* data() const override;

    ParameterBase* copy() const override;
    
private:
    int* data_;
};

} // core 

#endif // CORE_PARAMETER_INT_H