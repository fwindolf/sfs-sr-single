#include "solver/optimizer/albedo.h"

#include "image/processing.h"

using namespace solver;

AlbedoOptimizer::AlbedoOptimizer(
    const int w, const int h, float msAlpha, float msLambda, int msMaxIter)
    : w_(w)
    , h_(h)
    , msAlpha_(msAlpha)
    , msLambda_(msLambda)
    , msMaxIter_(msMaxIter)
{
}

AlbedoOptimizer::AlbedoOptimizer(AlbedoOptimizer&& other)
    : w_(other.w_)
    , h_(other.h_)
    , msAlpha_(other.msAlpha_)
    , msLambda_(other.msLambda_)
    , msMaxIter_(other.msMaxIter_)
    , albedo_opt(std::move(other.albedo_opt))
    , msFilter_(std::move(other.msFilter_))
{
}

AlbedoOptimizer::~AlbedoOptimizer() {}

void AlbedoOptimizer::init(const Image<float3>& image,
    const Image<uchar>& mask, const Image<float3>& shading)
{
    msFilter_.reset(new image::MumfordShahFilter<float3>(
        msAlpha_, msLambda_, msMaxIter_, mask));

    // Initialize albedo from scratch
    msFilter_->init(image);
    msFilter_->run(shading, albedo_opt, 10 * msMaxIter_);
}

void AlbedoOptimizer::initFrom(const Image<float3>& image,
    const Image<uchar>& mask, const Image<float3>& albedo)
{
    msFilter_.reset(new image::MumfordShahFilter<float3>(
        msAlpha_, msLambda_, msMaxIter_, mask));

    // Initialize albedo from lower resolution
    auto mask_lr = mask.resized(albedo.width(), albedo.height(), mask);
    albedo_opt = albedo.resized(w_, h_, mask_lr, cuimage::NEAREST);
    msFilter_->init(albedo_opt);
}

void AlbedoOptimizer::step(const Image<float3>& shading)
{
    assert(!shading.empty());

    // Run mumford shah with shading
    msFilter_->run(shading, albedo_opt);
}
