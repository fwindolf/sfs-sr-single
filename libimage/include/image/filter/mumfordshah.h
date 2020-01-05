/**
 * @file   mumfordshah.h
 * @brief  Image cartooning with mumford shah energy functional
 * @author Florian Windolf
 */
#ifndef IMAGE_FILTER_MUMFORD_SHAH_H
#define IMAGE_FILTER_MUMFORD_SHAH_H

#include "core/image.h"

#include "image/filter/mumfordshah_cu.h"

#include <iostream>

namespace image
{

/**
 * @class MumfordShahFilter
 * @brief Smooth images while keeping edges intact, resulting in cartoonish images
 */
template <typename T>
class MumfordShahFilter 
{
public:
    MumfordShahFilter(float alpha, float lambda, int maximumIterations, const Image<uchar>& mask);
    
    ~MumfordShahFilter(){};

    void compute(const Image<T>& in, Image<T>& out);

    /**
     * Initialize to run MS with same image target, but different shading
     */
    void init(const Image<T>& image);

    /**
     * Run MS with same image target, but different shading
     * Take the last result as initialization. If iterations is 0, maximumIterations from constructor are used
     */
    void run(const Image<T>& shading, Image<T>& output, const int iterations = 0);

private:
    int iterations_;
    const float alpha_;  // Penalizes smoothness of minimizer outside discontinuities of primal variable. 
    const float lambda_; // Penalizes length of discontinuity set

    // Lower level
    bool init();

    bool run(Image<T>& output);

    void updatePrimal();

    void updateStepSizes();

    void updateDual();

    bool checkEarlyStop();

    Image<T> intensity_;
    Image<T> px_, py_;
    Image<T> u_;
    Image<T> u_bar_;
    Image<T> u_diff_;
    Image<T> scalar_op_;
    Image<uchar> mask_;

    size_t width_;
    size_t height_;
    size_t channels_;

    float img_domain_;

    float sigma_;
    float gamma_;
    float tau_;
    float theta_;
    float tolerance_;
};

template <typename T>
MumfordShahFilter<T>::MumfordShahFilter(float alpha, float lambda, int maximumIterations, const Image<uchar>& mask)
 : alpha_(alpha), 
   lambda_(lambda),
   tolerance_(1e-5),
   tau_(0.25f),
   sigma_(0.5f),
   gamma_(2.f),
   theta_(0.f),
   iterations_(maximumIterations)
{  
    if(!mask.empty())
        mask_.copyFrom(mask);
}

template <typename T>
void MumfordShahFilter<T>::compute(const Image<T>& in, Image<T>& out)
{
    if (in.empty())
        throw std::runtime_error("Can not mumfordshah-filter empty/non-float image!");

    width_ = in.width();
    height_ = in.height();

    if(mask_.empty())
        mask_ = Image<uchar>(in.width(), in.height(), 255);
    
    intensity_.copyFrom(in);

    if(!init())
        throw std::runtime_error("Could not initialize mumford shah filter!");

    if(!run(out))
        throw std::runtime_error("Error during computation in mumford shah filter!");
}

template <typename T>
void MumfordShahFilter<T>::init(const Image<T>& in)
{
    if (in.empty())
        throw std::runtime_error("Can not mumfordshah-filter empty/non-float image!");

    width_ = in.width();
    height_ = in.height();

    if(mask_.empty())
        mask_ = Image<uchar>(width_, height_, 255);

    intensity_.copyFrom(in);

    if(!init())
        throw std::runtime_error("Could not initialize mumford shah filter!");
}

template <typename T>
void MumfordShahFilter<T>::run(const Image<T>& shading, Image<T>& out, const int iterations)
{
    if (shading.empty())
        throw std::runtime_error("Can not run initialized mumfordshah-filter without shading!");

    scalar_op_.copyFrom(shading);

    auto iters_original = iterations_;
    if (iterations > 0)
        iterations_ = iterations;

    // Reset
    px_.setTo(0.f);
    py_.setTo(0.f);
    u_diff_.setTo(0.f);

    // Reset result
    u_.copyFrom(intensity_);
    u_bar_.copyFrom(intensity_);

    tolerance_ = 1e-5;
    tau_ = 0.25f;
    sigma_ = 0.5f;
    gamma_ = 2.f;
    theta_= 0.f;

    if(!run(out))
        throw std::runtime_error("Error during computation in mumford shah filter!");

    iterations_ = iters_original;
}  

template <typename T>
bool MumfordShahFilter<T>::init()
{
    intensity_.mask(mask_);
    
    px_ = Image<T>(width_, height_, 0.f);
    py_ = Image<T>(width_, height_, 0.f);
    u_diff_ = Image<T>(width_, height_, 0.f);
    scalar_op_ = Image<T>(width_, height_, 1.f);
    
    u_.copyFrom(intensity_);
    u_bar_.copyFrom(intensity_);

    img_domain_ = cuimage::channels<T>() * mask_.valid();
    return true;
}


template <typename T>
bool MumfordShahFilter<T>::run(Image<T>& out)
{
    for (int i = 0; i < iterations_; i++)
    {
        updateDual();
        updateStepSizes();
        updatePrimal();
       
        if (i && i % cuimage::max(10, iterations_ / 10) == 0)
        {
            if(checkEarlyStop())
                break;
        }
    }

    out.copyFrom(u_);
    return true;
}

template <typename T>
void MumfordShahFilter<T>::updatePrimal()
{
    cu_UpdatePrimal<T>(u_, u_bar_, u_diff_, intensity_, px_, py_, scalar_op_, mask_, tau_, theta_);
}

template <typename T>
void MumfordShahFilter<T>::updateStepSizes()
{
    theta_ = 1.f / sqrt(1.f + 2.f * gamma_ * tau_);
    tau_ = tau_ * theta_;
    sigma_ = sigma_ / theta_;
}

template <typename T>
void MumfordShahFilter<T>::updateDual()
{
    cu_UpdateDual<T>(px_, py_, u_bar_, mask_, sigma_, alpha_, lambda_);
}

template <typename T>
bool MumfordShahFilter<T>::checkEarlyStop()
{
    float norm = u_diff_.norm1() / img_domain_;
    if (norm < tolerance_)
        return true;

    return false;
}

} // image

#endif // IMAGE_FILTER_MUMFORD_SHAH_H