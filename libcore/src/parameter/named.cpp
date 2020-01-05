#include "core/parameter/named.h"

#include "core/parameter/int.h"
#include "core/parameter/float.h"
#include "core/parameter/image.h"

#include <algorithm>
#include <iostream>

using namespace core;

NamedParameters::NamedParameter::NamedParameter()
    : NamedParameter("", nullptr, 0, false) {}

NamedParameters::NamedParameter::NamedParameter(std::string name, ParameterBase *data, unsigned int position, bool unknown)
    : name_(name),
      data_(data), 
      position_(position),
      unknown_(unknown) {}

NamedParameters::NamedParameter::NamedParameter(NamedParameter&& p)
{
    if (!empty())
    {
        std::cout << "Trying to overwrite parameter " << name_ << " with data " << data_ << std::endl;
        throw std::runtime_error("Non-empty parameters cannot be overwritten!");
    }
    
    name_ = std::move(p.name_);
    data_ = std::move(p.data_);
    position_ = std::move(p.position_);
    unknown_ = std::move(p.unknown_);

    p.data_ = nullptr;
}

NamedParameters::NamedParameter& NamedParameters::NamedParameter::operator=(NamedParameter&& p)
{
    if (!empty())
    {
        std::cout << "Trying to overwrite parameter " << name_ << " with data " << data_ << std::endl;
        throw std::runtime_error("Non-empty parameters cannot be overwritten!");
    }

    name_ = std::move(p.name_);
    data_ = std::move(p.data_);
    position_ = std::move(p.position_);
    unknown_ = std::move(p.unknown_);

    p.data_ = nullptr;
    return *this;
}

NamedParameters::NamedParameter::~NamedParameter()
{
    delete data_;
}

bool NamedParameters::NamedParameter::empty() const
{
    return name_.empty();
}

NamedParameters::NamedParameter NamedParameters::NamedParameter::copy() const
{
    return NamedParameter(name_, data_->copy(), position_, unknown_);
}

NamedParameters::NamedParameters(const NamedParameters& p)
{
    // Clear the existing parameters
    namedParameters_.clear();

    // Deep copy 
    for (auto&& param : p.namedParameters_)
    {
        add(param.copy(), param.position());
    }
}

bool NamedParameters::add(NamedParameters::NamedParameter&& parameter, size_t position)
{
    assert(position >= 0);

    if (position >= namedParameters_.size())
        namedParameters_.resize(position + 1);

    // Cannot overwrite existing entries...
    if (!namedParameters_.at(position).empty())
        return false;

    assert(namedParameters_.size() > position);
    namedParameters_.at(position) = std::move(parameter);
    return true;
}

bool NamedParameters::add(std::string name, int data, int position, bool unknown)
{
    if(position == -1)
        position = namedParameters_.size();
    
    auto p = NamedParameter(name, new IntParameter(data), position, unknown);
    return add(std::move(p), position);
}

bool NamedParameters::add(std::string name, float data, int position, bool unknown)
{
    if(position == -1)
        position = namedParameters_.size();
    
    auto p = NamedParameter(name, new FloatParameter(data), position, unknown);
    return add(std::move(p), position);
}

bool NamedParameters::add(std::string name, const Graph& graph, int position, bool unknown)
{
    if(position == -1)
        position = namedParameters_.size();
    
    // Number of edges in graphs
    auto p_edges = NamedParameter(name, new IntParameter(graph.edges()), position, unknown);
    auto status = add(std::move(p_edges), position);
    if(!status)
        return false;

    // Index data
    for (int i = 0; i < graph.count(); i++)
    {
        auto& img = graph.at(i);
        auto p = NamedParameter(name, new ImageParameter<int>(img), position + i + 1, unknown);
        status = add(std::move(p), position + i + 1);
        if(!status)
            return false;
    }

    return status;
}

NamedParameters::NamedParameter NamedParameters::at(std::string name) const
{
    auto it = std::find_if(namedParameters_.begin(), namedParameters_.end(),
    [&name](const NamedParameter& p) {
        return p.name() == name;
    });
    auto p = it->copy();
    return p;
}

NamedParameters::NamedParameter NamedParameters::at(unsigned int position) const
{
    auto it = std::find_if(namedParameters_.begin(), namedParameters_.end(), 
    [&position](const NamedParameter& p){
        return p.position() == position;
    });

    assert((*it).position() == it - namedParameters_.begin());
    auto p = it->copy();
    return p;
}

std::vector<std::string> NamedParameters::names() const
{
    std::vector<std::string> names;
    for (auto& p : namedParameters_)
    {
        if(!p.empty())
            names.push_back(p.name());
    }
    return names;
}

std::vector<void *> NamedParameters::data() const
{
    std::vector<void*> data;
    for (size_t i = 0; i < namedParameters_.size(); i++)
    {
        auto &p = namedParameters_.at(i);
        if(p.empty())
            continue;

        assert(p.position() == i); // Not at anticipated position!
        data.push_back(p.data());
    }
    return data;
}

void NamedParameters::clear()
{
    namedParameters_.clear();
}

size_t NamedParameters::size() const
{
    return namedParameters_.size();
}