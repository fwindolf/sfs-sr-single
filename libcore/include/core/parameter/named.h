/**
 * @file   named.h
 * @brief  Parameter with names encapsuling concrete parameters
 * @author Florian Windolf
 */
#ifndef CORE_PARAMETER_NAMED_H
#define CORE_PARAMETER_NAMED_H

#include <string>
#include <vector>

#include "core/parameter/base.h"
#include "core/parameter/image.h"
#include "core/parameter/graph.h"

namespace core
{

/**
 * @class  NamedParameters
 * @brief  Stores parameters of different types
 * 
 * Takes care of providing parameters as void* to objects like opt solver.
 * Furthermore takes care of allocation, ordering and naming.
 */
class NamedParameters
{
    /**
     * @class NamedParameter
     * @brief Allows easy access to attributes of the parameters
     */
    struct NamedParameter{
        NamedParameter();

        NamedParameter(std::string name, ParameterBase* data, unsigned int position, bool unknown);

        NamedParameter(const NamedParameter& p) = delete;

        NamedParameter(NamedParameter&& p);

        NamedParameter& operator=(NamedParameter&& p);

        ~NamedParameter();

        bool empty() const;

        std::string name()      const { return name_; }
        void* data()            const { return data_->data(); }
        unsigned int position() const { return position_; }
        bool unknown()          const { return unknown_; }

        NamedParameter copy() const;

    private:
        std::string name_;
        ParameterBase* data_;
        unsigned int position_;
        bool unknown_;
    };

public:
    NamedParameters(){};

    NamedParameters(const NamedParameters& p);

    bool add(std::string name, int data, int position = -1, bool unknown = false);

    bool add(std::string name, float data, int position = -1, bool unknown = false);

    bool add(std::string name, const Graph& graph, int position = -1, bool unknown = false);

    template <typename T>
    bool add(std::string name, const Image<T>& image, int position = -1, bool unknown = false);

    NamedParameter at(std::string name) const;

    NamedParameter at(unsigned int position) const;

    std::vector<std::string> names() const;

    std::vector<void*> data() const;

    void clear();

    size_t size() const;

private:
    bool add(NamedParameter&& parameter, size_t position);

    std::vector<NamedParameter> namedParameters_;
};

template <typename T>
bool NamedParameters::add(std::string name, const Image<T>& image, int position, bool unknown)
{
    if (position == -1)
        position = namedParameters_.size();
        
    auto p = NamedParameter(name, new ImageParameter<T>(image), position, unknown);
    return add(std::move(p), position);
}


} // core

#endif // CORE_PARAMETER_NAMED_H