#include "core/graph.h"

using namespace core;

Graph::Graph(const int n, const int edges, std::vector<int*> indices)
 : count_(n), edges_(edges)
{
    // assert(indices.size() == n);

    for (int* idata : indices)
        indices_.emplace_back(idata, 1, edges);
}

void* Graph::data() const
{
    std::vector<void*> data;
    
    void* n = (void*)&edges_;
    data.push_back(n);
    for (auto& id : indices_)
        data.push_back((void*)id.data());
    
    return data.data();
}

void Graph::print() const
{
    std::cout << "Graph with " << count_ << " node pairs and " << edges_ << " edges." << std::endl;
    for (auto& node : indices_)
    {
        std::cout << "Node: " << node.width() << "x" << node.height() << " " << static_cast<void*>(node.data()) << std::endl;
        // node.print();
    }
    std::cout << "End of Graph." << std::endl;        
    
}

const Image<int>& Graph::at(const int idx) const
{
    assert(idx < count_);
    return indices_.at(idx);
}

int Graph::edges() const
{
    return edges_;
}

int Graph::count() const
{
    return count_;
}