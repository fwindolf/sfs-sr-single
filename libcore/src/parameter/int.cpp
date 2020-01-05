#include "core/parameter/int.h"

using namespace core;

IntParameter::IntParameter(int data)
    : data_(new int(data))
{
}

IntParameter::~IntParameter()
{
    delete data_;
}

void *IntParameter::data() const
{
    return data_;
}

ParameterBase *IntParameter::copy() const
{
    return new IntParameter(*data_);
}