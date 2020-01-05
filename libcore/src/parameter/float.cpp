#include "core/parameter/float.h"

using namespace core;

FloatParameter::FloatParameter(float data)
    : data_(new float(data)) {}

FloatParameter::~FloatParameter()
{
    delete data_;
}

void *FloatParameter::data() const
{
    return data_;
}

ParameterBase *FloatParameter::copy() const
{
    return new FloatParameter(*data_);
}