#include <gtest/gtest.h>

#if TEST_INPAINT

#include "solver/optimizer/inpaint.h"

#include "image/processing.h"
#include "image/data.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace solver;

class InpaintTest : public ::testing::Test
{
public:
    InpaintTest()
    {
        clean = cv::Mat::ones(h, w, CV_32FC1);
        nans = cv::Mat::ones(h, w, CV_32FC1);
        
        cv::rectangle(nans, cv::Point(4, 4), cv::Point(15, 15), cv::Scalar(std::nanf("")), cv::FILLED);
        cv::rectangle(nans, cv::Point(10, 28), cv::Point(40, 31), cv::Scalar(std::nanf("")), cv::FILLED);
        cv::circle(nans, cv::Point(40, 12), 10, cv::Scalar(std::nanf("")), cv::FILLED);

        imageClean.upload((float*)clean.data, w, h);
        imageNan.upload((float*)nans.data, w, h);
    }

    int w = 64;
    int h = 32;

    cv::Mat clean, nans;
    Image<float> imageClean, imageNan;
};

TEST_F(InpaintTest, returns_image_if_no_nans_found)
{
    Image<float> out(imageClean);
    Image<uchar> mask(w, h, 1.f);
    InpaintFilter filter(mask);

    EXPECT_TRUE(filter.removeNans(out));
    ASSERT_FALSE(out.empty());

    float* h_out = out.download();

    for (int i = 0; i < w * h; i ++)
        EXPECT_FLOAT_EQ(clean.at<float>(i), h_out[i]);
    
    delete h_out;
}

TEST_F(InpaintTest, doesnt_change_existing_values)
{
    Image<float> out(imageClean);
    Image<uchar> mask(w, h, 1.f);
    InpaintFilter filter(mask);

    EXPECT_TRUE(filter.removeNans(out));
    ASSERT_FALSE(out.empty());

    float* h_out = out.download();

    for (int i = 0; i < w * h; i ++)
    {
        float cmp = nans.at<float>(i);
        if(!std::isnan(cmp))
            EXPECT_NEAR(cmp, h_out[i], 1e-4);
    }
    delete h_out;
}

TEST_F(InpaintTest, fills_nan_values)
{
    Image<float> out(imageNan);
    Image<uchar> mask(w, h, 1.f);
    InpaintFilter filter(mask);

    EXPECT_TRUE(filter.removeNans(out));
    ASSERT_FALSE(out.empty());

    float* h_out = out.download();

    for (int i = 0; i < w * h; i ++)
    {
        const float val = h_out[i];

        if(std::isnan(nans.at<float>(i)))
        {
            EXPECT_NE(0.f, val);
            EXPECT_FALSE(std::isnan(val));        
        }        
    }
    delete h_out;
}

TEST_F(InpaintTest, inpaints_with_mask)
{
    cv::Mat m = cv::Mat::zeros(h, w, CV_8UC1);
    cv::rectangle(m, cv::Point(5, 5), cv::Point(w-5, h-5), cv::Scalar(1), cv::FILLED);

    Image<float> out(imageNan);
    Image<uchar> mask;
    mask.upload((uchar*)m.data, w, h);

    InpaintFilter filter(mask);
    filter.removeNans(out);

    float* h_out = out.download();
    for (int x = 10; x < 53; x++)
        for (int y = 10; y < 21; y++)
            EXPECT_NE(0.f, h_out[y * w + x]);  

    delete h_out;
}

TEST_F(InpaintTest, inpaints_depth)
{
    w = 640;
    h = 480;
    image::DataSet d(std::string(SOURCE_DIR) + "/data/", w, h, 2);

    float* h_depth = d.depth_star.download();
    cv::Mat depth_(h, w, CV_32FC1, h_depth);
    cv::circle(depth_, cv::Point(320, 100), 10, cv::Scalar(std::nanf("")), cv::FILLED);
    cv::circle(depth_, cv::Point(320, 150), 20, cv::Scalar(std::nanf("")), cv::FILLED);
    cv::circle(depth_, cv::Point(320, 240), 30, cv::Scalar(std::nanf("")), cv::FILLED);
    cv::circle(depth_, cv::Point(320, 300), 10, cv::Scalar(std::nanf("")), cv::FILLED);

    Image<float> depth;
    depth.upload(h_depth, w, h);

    InpaintFilter filter(d.mask);
    filter.removeNans(depth);

    ASSERT_FALSE(depth.empty());

    float* h_out = depth.download();

    for (int i = 0; i < depth_.total(); i++)
        if(std::isnan(depth_.at<float>(i)))
            EXPECT_FALSE(std::isnan(h_out[i]));
    
    delete h_depth;
    delete h_out;
}

#endif