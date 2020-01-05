#include "solver/optimizer/lighting.h"

#include "image/processing.h"
#include "solver/optimizer/lighting_cu.h"

using namespace solver;

LightingOptimizer::LightingOptimizer(const int w, const int h, const int order)
    : w_(w)
    , h_(h)
    , p_((order + 1) * (order + 1))
{
    if (order > 1)
        throw std::runtime_error("Harmonics order > 1 not implemented!");

    lineq.reset(new LinearEquationSolver(p_, p_, 1));

    matrixA_ = Image<float>(p_, w_ * h_ * 3, 0.f);
    matrixb_ = Image<float>(1, w_ * h_ * 3, 0.f);

    // Allocate ATA und ATb matrices
    matrixATA_ = Image<float>(p_, p_, 0.f);
    matrixATb_ = Image<float>(1, p_, 0.f);

    cublasSafeCall(cublasCreate(&cublasHandle_));
}

LightingOptimizer::LightingOptimizer(LightingOptimizer&& other)
    : w_(other.w_)
    , h_(other.h_)
    , p_(other.p_)
    , shading_opt(std::move(other.shading_opt))
    , lighting_opt(std::move(other.lighting_opt))
    , matrixA_(std::move(other.matrixA_))
    , matrixb_(std::move(other.matrixb_))
    , matrixATA_(std::move(other.matrixATA_))
    , matrixATb_(std::move(other.matrixATb_))
    , lineq(std::move(other.lineq))
{
    cublasSafeCall(cublasCreate(&cublasHandle_));
}

LightingOptimizer::~LightingOptimizer()
{
    cublasSafeCall(cublasDestroy(cublasHandle_));
}

void LightingOptimizer::init(const Image<float3>& image,
    const Image<uchar>& mask, const Image<float4>& spherical_harmonics)
{
    cu_CalculateBMatrix<float3>(matrixb_, image, mask);
    lighting_opt.realloc(1, p_);
    shading_opt.realloc(w_, h_);

    // Initialize lighting and shading from scratch
    cu_InitializeLighting<float3>(lighting_opt);
    cu_CalculateShading<float3>(
        shading_opt, spherical_harmonics, lighting_opt, mask);
}

void LightingOptimizer::initFrom(const Image<float3>& image,
    const Image<uchar>& mask, const Image<float3>& lighting,
    const Image<float4>& spherical_harmonics)
{
    cu_CalculateBMatrix<float3>(matrixb_, image, mask);

    // Initialize lighting and shading from existing light
    lighting_opt.copyFrom(lighting);
    shading_opt.realloc(w_, h_);
    cu_CalculateShading<float3>(
        shading_opt, spherical_harmonics, lighting_opt, mask);
}

void LightingOptimizer::step(const Image<uchar>& mask,
    const Image<float4>& spherical_harmonics, const Image<float3>& albedo)
{
    // Calculate A matrix  = masked(albedo * spherical_harmonics))
    cu_CalculateAMatrix<float3>(matrixA_, albedo, spherical_harmonics, mask);

    // Calculate ATA
    cu_CalculateATA(matrixATA_, matrixA_, cublasHandle_);

    // Calculate ATb
    cu_CalculateATb(matrixATb_, matrixA_, matrixb_, cublasHandle_);

    Image<float> x;
    lineq->solve(x, matrixATA_, matrixATb_);

    // Broadcast to c channels
    cu_SetLighting<float3>(lighting_opt, x);

    // Calculate shading
    cu_CalculateShading<float3>(
        shading_opt, spherical_harmonics, lighting_opt, mask);
}

void LightingOptimizer::updateShading(
    const Image<uchar>& mask, const Image<float4>& spherical_harmonics)
{
    cu_CalculateShading<float3>(
        shading_opt, spherical_harmonics, lighting_opt, mask);
}