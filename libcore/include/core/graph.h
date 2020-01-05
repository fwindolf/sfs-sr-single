/**
 * @file   graph.h
 * @brief  Graph class
 * @author Florian Windolf
 */
#ifndef CORE_GRAPH_H
#define CORE_GRAPH_H

#include "core/image.h"
#include <vector>

#include "assert.h"

namespace core
{

/**
 * @class Graph
 * @brief Implements Graph types using the Image class
 */
class Graph
{
public:
    Graph(const int n, const int edges, std::vector<int*> indices);

    void* data() const;

    void print() const;

    const Image<int>& at(const int idx) const;

    int edges() const;

    int count() const;
    
private:
    const int count_, edges_;
    std::vector<Image<int>> indices_;
};

} // core

#endif // CORE_GRAPH_H