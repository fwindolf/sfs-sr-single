#include <gtest/gtest.h>

#include "image/filter/filter.h"
#include "image/filter/gaussian/gaussian.h"
#include "image/filter/bilateral/bilateral.h"
#include "image/filter/guidedimage.h"

#include <cuda_runtime.h>

using namespace image;
using namespace testing;

#define floatT core::Image::Type::Float
#define doubleT core::Image::Type::Double
#define intT core::Image::Type::Int
#define ucharT core::Image::Type::UChar

#define cpuL core::Image::Location::CPU
#define gpuL core::Image::Location::GPU

struct ImageData{
    ImageData(int h, int w, int c, core::Image::Type type, core::Image::Location location)
        : h(h), w(w), c(c), type(type), loc(location) {}
    int h;
    int w;
    int c;
    core::Image::Type type;
    core::Image::Location loc;
};

class FilterTest : public Test
{
public:
    FilterTest(Filter* newFilter)
     : filter(newFilter)
    {
        
    }

    std::unique_ptr<Filter> filter;
    ImagePtr in, out;
    ImagePtr in_zeros, in_striped;

    // Some default tests that hold true for all filters
    void test_fails_for_empty_images()
    {
        internal::CaptureStderr();
        out = filter->compute(in);
        ASSERT_EQ(nullptr, out);
        EXPECT_FALSE(internal::GetCapturedStderr().empty());
    }

    void test_doesnt_change_image_dimensions()
    {
        out = filter->compute(in_zeros);
        ASSERT_NE(nullptr, out);
        EXPECT_EQ(in_zeros->height(), out->height());
        EXPECT_EQ(in_zeros->width(), out->width());
        EXPECT_EQ(in_zeros->channels(), out->channels());
    }

    void test_creates_data_for_output_image()
    {
        out = filter->compute(in_zeros);
        ASSERT_NE(nullptr, out);

        EXPECT_FALSE(out->empty());
        EXPECT_EQ(in->type(), out->type());
        EXPECT_TRUE(out->ownsData());
    }

    void test_doesnt_alter_zero_image()
    {
        out = filter->compute(in_zeros);
        ASSERT_NE(nullptr, out);
        ASSERT_TRUE(out->data());
        
        //EXPECT_TRUE(in_zeros->equals(out));
    }

};

class FilterCpuTest : public FilterTest
{ 
public:  
    FilterCpuTest(Filter* newFilter)
    : FilterTest(newFilter),
      d(6, 5, 3, floatT, cpuL)          
    {   
        in = std::make_shared<core::Image>(d.h, d.w, d.c, d.type, d.loc);

        auto data = new float[d.h * d.w * d.c]();
        in_zeros = std::make_shared<core::Image>(d.h, d.w, d.c, data, d.type, d.loc);

        auto cdata = new float[d.h * d.w * d.c];
        for(int c = 0; c < d.c; c++)       
        {
            for (int h = 0; h < d.h; h++)
            {
                int i = 0;
                for (int w = 0; w < d.w; w++)
                {
                    cdata[(c * d.h * d.w) + h * d.w + w] = i % 2; 
                    i++;
                }
            }
        }
        in_striped = std::make_shared<core::Image>(d.h, d.w, d.c, cdata, d.type, d.loc);
    }

    ~FilterCpuTest()
    {
    }

    ImageData d;    
};


class GaussianFilterCpu : public FilterCpuTest
{
public:
    GaussianFilterCpu()
        : FilterCpuTest(new GaussianFilter(3, 0.f))
    {
    }

    void setKernelSize(int ks)
    {
        filter.reset(new GaussianFilter(ks, 0.f));
    }

    void setSigma(float sigma)
    {
        filter.reset(new GaussianFilter(0, sigma));
    }
};

TEST_F(GaussianFilterCpu, fails_for_empty_images)
{
    test_fails_for_empty_images();
}

TEST_F(GaussianFilterCpu, doesnt_change_image_dimensions)
{
    test_doesnt_change_image_dimensions();
}

TEST_F(GaussianFilterCpu, creates_data_for_output_image)
{
    test_creates_data_for_output_image();
}

TEST_F(GaussianFilterCpu, doesnt_alter_zero_image)
{
    test_doesnt_alter_zero_image();
}

TEST_F(GaussianFilterCpu, doesnt_alter_image_with_no_kernel)
{
    int kernelSize = 1; // Only consider the center pixel
    setKernelSize(kernelSize);
    
    out = filter->compute(in_striped);
    ASSERT_NE(nullptr, out);
    ASSERT_TRUE(out->data());

    auto idata = (float*)in_striped->data();
    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_FLOAT_EQ(idata[i], fdata[i]);
}


TEST_F(GaussianFilterCpu, averages_stripes_except_on_borders)
{
    out = filter->compute(in_striped);
    ASSERT_NE(nullptr, out);
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    // Dont check borders
    int b = 1;

    for (int c = 0; c < d.c; c++)
        for(int h = b; h < d.h - b; h++)
            for(int w = b; w < d.w - b; w++)
                EXPECT_FLOAT_EQ(0.5, fdata[c * d.h * d.w + h * d.w + w]);
}


class BilateralFilterCpu : public FilterCpuTest
{
public:
    BilateralFilterCpu()
        : dia_(3), sc_(4.f), sp_(4.f),
        FilterCpuTest(new BilateralFilter(dia_, sc_, sp_))
    {
    }

    void setDiameter(int dia)
    {
        dia_ = dia;
        filter.reset(new BilateralFilter(dia_, sc_, sp_));
    }

    void setSigma(float sigmaColor, float sigmaSpace)
    {
        sc_ = sigmaColor;
        sp_ = sigmaSpace;
        filter.reset(new BilateralFilter(dia_, sc_, sp_));
    }
private:
    int dia_;
    float sc_, sp_;
};

TEST_F(BilateralFilterCpu, fails_for_empty_images)
{
    test_fails_for_empty_images();
}

TEST_F(BilateralFilterCpu, doesnt_change_image_dimensions)
{
    test_doesnt_change_image_dimensions();
}

TEST_F(BilateralFilterCpu, creates_data_for_output_image)
{
    test_creates_data_for_output_image();
}

TEST_F(BilateralFilterCpu, doesnt_alter_zero_image)
{
    test_doesnt_alter_zero_image();
}

TEST_F(BilateralFilterCpu, leaves_stripes_intact_except_for_border)
{
    setDiameter(3);
    setSigma(1.f, 1.f);

    out = filter->compute(in_striped);
    ASSERT_NE(nullptr, out);
    ASSERT_TRUE(out->data());

    float* fdata = (float*)out->data();
    int b = 1;
    for (int c = 0; c < d.c; c++)
    {
        for(int h = b; h < d.h - b; h++)
        {
            for(int w = b; w < d.w - b; w++)
            {
                float expected = 0.f;
                if(w % 2 == 1)
                    expected = 1.f;
                
                EXPECT_NEAR(expected, fdata[c * d.h * d.w + h * d.w + w], 0.01);
            }
        }
    }           

}

/**

TEST_F(MedianFilterCpu, fails_for_empty_images)
{
    int kernelSize = 5;
    internal::CaptureStderr();   
    EXPECT_FALSE(filter.median(*in, *out, kernelSize));
    EXPECT_FALSE(internal::GetCapturedStderr().empty());
}

TEST_F(MedianFilterCpu, doesnt_change_image_dimensions)
{
    int kernelSize = 5;

    auto data = new float[d.h * d.w * d.c]();
    in->update(data, cpuL);
    
    ASSERT_TRUE(filter.median(*in, *out, kernelSize));
    EXPECT_EQ(d.h, out->height());
    EXPECT_EQ(d.w, out->width());
    EXPECT_EQ(d.c, out->channels());
}

TEST_F(MedianFilterCpu, creates_data_for_output_image)
{
    int kernelSize = 5;

    ASSERT_TRUE(filter.median(*in_zeros, *out, kernelSize));
    EXPECT_FALSE(out->empty());
    EXPECT_EQ(in->type(), out->type());
    EXPECT_TRUE(out->ownsData());
}

TEST_F(MedianFilterCpu, doesnt_alter_zero_image)
{
    int kernelSize = 5;

    ASSERT_TRUE(filter.median(*in_zeros, *out, kernelSize));
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_FLOAT_EQ(0.f, fdata[i]);
}

TEST_F(MedianFilterCpu, doesnt_alter_image_with_no_kernel)
{
    int kernelSize = 1; // Only consider the center pixel

    ASSERT_TRUE(filter.median(*in_striped, *out, kernelSize));
    ASSERT_TRUE(out->data());

    auto idata = (float*)in_striped->data();
    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_FLOAT_EQ(idata[i], fdata[i]);
}

TEST_F(MedianFilterCpu, returns_only_values_from_image)
{
    int kernelSize = 3;
    ASSERT_TRUE(filter.median(*in_striped, *out, kernelSize));    
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_TRUE(fdata[i] == 1 || fdata[i] == 0);
}


TEST_F(BilateralFilterCpu, fails_for_empty_images)
{
    int kernelSize = 5;
    internal::CaptureStderr();   
    EXPECT_FALSE(filter.bilateral(*in, *out, kernelSize));
    EXPECT_FALSE(internal::GetCapturedStderr().empty());    
}

TEST_F(BilateralFilterCpu, doesnt_change_image_dimensions)
{
    int kernelSize = 5;

    auto data = new float[d.h * d.w * d.c]();
    in->update(data, cpuL);
    
    ASSERT_TRUE(filter.bilateral(*in, *out, kernelSize));
    EXPECT_EQ(d.h, out->height());
    EXPECT_EQ(d.w, out->width());
    EXPECT_EQ(d.c, out->channels());
}

TEST_F(BilateralFilterCpu, creates_data_for_output_image)
{
    int kernelSize = 5;

    ASSERT_TRUE(filter.bilateral(*in_zeros, *out, kernelSize));
    EXPECT_FALSE(out->empty());
    EXPECT_EQ(in->type(), out->type());
    EXPECT_TRUE(out->ownsData());
}

TEST_F(BilateralFilterCpu, doesnt_alter_zero_image)
{
    int kernelSize = 5;

    ASSERT_TRUE(filter.bilateral(*in_zeros, *out, kernelSize));
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_FLOAT_EQ(0.f, fdata[i]);
}

TEST_F(BilateralFilterCpu, works_with_pixel_kernel)
{
    int kernelSize = 1; // Only consider the center pixel

    ASSERT_TRUE(filter.bilateral(*in_striped, *out, kernelSize));
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_TRUE(fdata[i] >= 0.f && fdata[i] < 1.f);
}


TEST_F(BilateralFilterCpu, works_with_bigger_kernel)
{
    int kernelSize = 3;
    ASSERT_TRUE(filter.bilateral(*in_striped, *out, kernelSize));    
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_TRUE(fdata[i] >= 0.f && fdata[i] < 1.f);
}


class MedianFilterCpu : public FilterCpuTest{};
class BilateralFilterCpu : public FilterCpuTest{};


*/
class FilterGpuTest : public FilterTest
{
public:  
    FilterGpuTest(Filter* newFilter)
        : d(5, 6, 2, floatT, gpuL),
          FilterTest(newFilter)
    {   
        in = std::make_shared<core::Image>(d.h, d.w, d.c, d.type, d.loc);

        float* data = new float[d.h * d.w * d.c]();
        in_zeros = std::make_shared<core::Image>(d.h, d.w, d.c, data, d.type, cpuL);
        in_zeros->upload();

        auto cdata = new float[d.h * d.w * d.c];
        for(int c = 0; c < d.c; c++)        
            for (int i = 0; i < d.h * d.w; i++)
                cdata[c * d.h * d.w + i] = i % 2;
 
        in_striped = std::make_shared<core::Image>(d.h, d.w, d.c, cdata, d.type, cpuL);
        in_striped->upload();
    }

    ~FilterGpuTest()
    {    
    }

    ImageData d;
};

class GaussianFilterGpu : public FilterGpuTest
{
public:
    GaussianFilterGpu()
    : FilterGpuTest(new GaussianFilter(3, 0.f))
    {

    }

};

TEST_F(GaussianFilterGpu, fails_for_empty_images)
{
    test_fails_for_empty_images();
}

TEST_F(GaussianFilterGpu, doesnt_change_image_dimensions)
{
    test_doesnt_change_image_dimensions();
}

TEST_F(GaussianFilterGpu, creates_data_for_output_image)
{
    test_creates_data_for_output_image();
}

TEST_F(GaussianFilterGpu, doesnt_alter_zero_image)
{
    test_doesnt_alter_zero_image();
}

/*
TEST_F(GaussianFilterGpu, averages_stripes_except_on_border)
{
    ASSERT_TRUE(filter->compute(in_striped, out));
    ASSERT_TRUE(out->data());

    out->download();
    out->print();
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    
    // Dont check borders
    int b = 1;

    for (int c = 0; c < d.c; c++)
        for(int h = b; h < d.h - b; h++)
            for(int w = b; w < d.w - b; w++)
                EXPECT_FLOAT_EQ(0.5, fdata[c * d.h * d.w + h * d.w + w]);
}

class MedianFilterGpu : public FilterGpuTest{};
class BilateralFilterGpu : public FilterGpuTest{};


TEST_F(GaussianFilterGpu, fails_for_empty_images)
{
    int kernelSize = 5;
    internal::CaptureStderr();   
    EXPECT_FALSE(filter.gaussian(*in, *out, kernelSize));
    EXPECT_FALSE(internal::GetCapturedStderr().empty());    
}

TEST_F(GaussianFilterGpu, doesnt_change_image_dimensions)
{
    int kernelSize = 5;

    auto data = new float[d.h * d.w * d.c]();
    in->update(data, gpuL);

    ASSERT_TRUE(filter.gaussian(*in, *out, kernelSize));
    EXPECT_EQ(d.h, out->height());
    EXPECT_EQ(d.w, out->width());
    EXPECT_EQ(d.c, out->channels());
}

TEST_F(GaussianFilterGpu, creates_data_for_output_image)
{
    int kernelSize = 5;

    ASSERT_TRUE(filter.gaussian(*in_zeros, *out, kernelSize));
    EXPECT_FALSE(out->empty());
    EXPECT_EQ(in->type(), out->type());
    EXPECT_TRUE(out->ownsData());
}

TEST_F(GaussianFilterGpu, doesnt_alter_zero_image)
{
    int kernelSize = 5;

    ASSERT_TRUE(filter.gaussian(*in_zeros, *out, kernelSize));
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_FLOAT_EQ(0.f, fdata[i]);
}

TEST_F(GaussianFilterGpu, doesnt_alter_image_with_no_kernel)
{
    int kernelSize = 1; // Only consider the center pixel

    ASSERT_TRUE(filter.gaussian(*in_striped, *out, kernelSize));
    ASSERT_TRUE(out->data());

    auto idata = (float*)in_striped->data();
    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_FLOAT_EQ(idata[i], fdata[i]);
}


TEST_F(GaussianFilterGpu, averages_stripes)
{
    int kernelSize = 3;

    ASSERT_TRUE(filter.gaussian(*in_striped, *out, kernelSize));
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_FLOAT_EQ(0.5, fdata[i]);
}


TEST_F(MedianFilterGpu, fails_for_empty_images)
{
    int kernelSize = 5;
    EXPECT_FALSE(filter.median(*in, *out, kernelSize));
}

TEST_F(MedianFilterGpu, doesnt_change_image_dimensions)
{
    int kernelSize = 5;

    auto data = new float[d.h * d.w * d.c]();
    in->update(data, gpuL);
    
    ASSERT_TRUE(filter.median(*in, *out, kernelSize));
    EXPECT_EQ(d.h, out->height());
    EXPECT_EQ(d.w, out->width());
    EXPECT_EQ(d.c, out->channels());
}

TEST_F(MedianFilterGpu, creates_data_for_output_image)
{
    int kernelSize = 5;

    ASSERT_TRUE(filter.median(*in_zeros, *out, kernelSize));
    EXPECT_FALSE(out->empty());
    EXPECT_EQ(in->type(), out->type());
    EXPECT_TRUE(out->ownsData());
}

TEST_F(MedianFilterGpu, doesnt_alter_zero_image)
{
    int kernelSize = 5;

    ASSERT_TRUE(filter.median(*in_zeros, *out, kernelSize));
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_FLOAT_EQ(0.f, fdata[i]);
}

TEST_F(MedianFilterGpu, doesnt_alter_image_with_no_kernel)
{
    int kernelSize = 1; // Only consider the center pixel

    ASSERT_TRUE(filter.median(*in_striped, *out, kernelSize));
    ASSERT_TRUE(out->data());

    auto idata = (float*)in_striped->data();
    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_FLOAT_EQ(idata[i], fdata[i]);
}

TEST_F(MedianFilterGpu, returns_only_values_from_image)
{
    int kernelSize = 3;
    ASSERT_TRUE(filter.median(*in_striped, *out, kernelSize));    
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_TRUE(fdata[i] == 1 || fdata[i] == 0);
}


TEST_F(BilateralFilterGpu, fails_for_empty_images)
{
    int kernelSize = 5;
    EXPECT_FALSE(filter.bilateral(*in, *out, kernelSize));
}

TEST_F(BilateralFilterGpu, doesnt_change_image_dimensions)
{
    int kernelSize = 5;

    auto data = new float[d.h * d.w * d.c]();
    in->update(data, gpuL);
    
    ASSERT_TRUE(filter.bilateral(*in, *out, kernelSize));
    EXPECT_EQ(d.h, out->height());
    EXPECT_EQ(d.w, out->width());
    EXPECT_EQ(d.c, out->channels());
}

TEST_F(BilateralFilterGpu, creates_data_for_output_image)
{
    int kernelSize = 5;

    ASSERT_TRUE(filter.bilateral(*in_zeros, *out, kernelSize));
    EXPECT_FALSE(out->empty());
    EXPECT_EQ(in->type(), out->type());
    EXPECT_TRUE(out->ownsData());
}

TEST_F(BilateralFilterGpu, doesnt_alter_zero_image)
{
    int kernelSize = 5;

    ASSERT_TRUE(filter.bilateral(*in_zeros, *out, kernelSize));
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_FLOAT_EQ(0.f, fdata[i]);
}

TEST_F(BilateralFilterGpu, works_with_pixel_kernel)
{
    int kernelSize = 1; // Only consider the center pixel

    ASSERT_TRUE(filter.bilateral(*in_striped, *out, kernelSize));
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_TRUE(fdata[i] >= 0.f && fdata[i] < 1.f);
}


TEST_F(BilateralFilterGpu, works_with_bigger_kernel)
{
    int kernelSize = 3;
    ASSERT_TRUE(filter.bilateral(*in_striped, *out, kernelSize));    
    ASSERT_TRUE(out->data());

    auto fdata = (float*)out->data();
    for (int i = 0; i < d.h * d.w * d.c; i++)
        EXPECT_TRUE(fdata[i] >= 0.f && fdata[i] < 1.f);
}

*/