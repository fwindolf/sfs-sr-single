#include "core/parameter/graph.h"

using namespace core;

GraphParameter::GraphParameter(const Graph& g)
 : graph_(g)
{
}

GraphParameter::~GraphParameter()
{
}

void* GraphParameter::data() const
{
    return graph_.data();
}

ParameterBase* GraphParameter::copy() const 
{
    return new GraphParameter(graph_);
}