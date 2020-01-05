/**
 * @file   graph.h
 * @brief  NamedParameter for graph
 * @author Florian Windolf
 */
#ifndef CORE_PARAMETER_GRAPH_H
#define CORE_PARAMETER_GRAPH_H

#include "core/graph.h"
#include "core/parameter/base.h"

namespace core
{

/**
 * @class GraphParameter
 * @brief Provides implementation of graph to Parameter interface
 */
class GraphParameter : public ParameterBase
{
public:
    GraphParameter(const Graph& g);

    ~GraphParameter();

    void* data() const override;

    ParameterBase* copy() const override;
    
private:
    const Graph& graph_;
};

} // core

#endif // CORE_PARAMETER_GRAPH_H
