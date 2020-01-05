#include "image/processing/depth.h"

#include <gtest/gtest.h>
#include <iostream>

using namespace image;
using namespace ::testing;

TEST(Blurring, smoothes_depth)
{
    cv::Mat depth_x(9, 9, CV_32FC1);
    cv::Mat depth_y(9, 9, CV_32FC1);
    cv::Mat depth_ref;

    for (int x = 0; x < 9; x++)
    {
        for (int y = 0; y < 9; y++)
        {
            depth_y.at<float>(x, y) = y % 2 + 1;
            depth_x.at<float>(x, y) = x % 2 + 1;
        }
    }

    Image<float> d(9, 9);
    Image<float> d_out(9, 9);
    Image<float> d_ref(9, 9);
    Image<uchar> mask(9, 9, 1);

    // Check in X direction
    d.upload((float*)depth_x.data, 9, 9);
    DepthProcessing p(d);
    p.blur(d_out, mask, 1, 2.f);

    cv::blur(depth_x, depth_ref, cv::Size(3, 3), cv::Point(-1, -1),
        cv::BORDER_REPLICATE);
    d_ref.upload((float*)depth_ref.data, 9, 9);

    EXPECT_LE((d_out - d_ref).norm2(), 1);

    // Check in Y direction
    d.upload((float*)depth_y.data, 9, 9);
    p.blur(d_out, mask, 1, 2.f);

    cv::blur(depth_y, depth_ref, cv::Size(3, 3), cv::Point(-1, -1),
        cv::BORDER_REPLICATE);
    d_ref.upload((float*)depth_ref.data, 9, 9);

    EXPECT_LE((d_out - d_ref).norm2(), 1);
}

TEST(Blurring, creates_no_invalid_values_from_valid_image)
{
    Image<float> d(100, 100, 2.f);
    Image<uchar> mask(100, 100, 1);
    Image<float> d_out;
    DepthProcessing p(d);
    p.blur(d_out, mask, 1, 2.f);

    EXPECT_EQ(d_out.nonzero(), d_out.size());
    EXPECT_EQ(d_out.valid(), d_out.size());
}

TEST(Blurring, doesnt_interpolate_with_zeros)
{
    cv::Mat depth_x(9, 9, CV_32FC1);
    cv::Mat depth_y(9, 9, CV_32FC1);
    cv::Mat depth_ref;

    for (int x = 0; x < 9; x++)
    {
        for (int y = 0; y < 9; y++)
        {
            depth_y.at<float>(x, y) = y % 2;
            depth_x.at<float>(x, y) = x % 2;
        }
    }

    Image<float> d(9, 9);
    Image<float> d_out(9, 9);
    Image<uchar> mask(9, 9, 1);

    // Check in X direction
    d.upload((float*)depth_x.data, 9, 9);
    DepthProcessing p(d);
    p.blur(d_out, mask, 1, 2.f);

    EXPECT_GE(d_out.min(), 1);

    // Check in Y direction
    d.upload((float*)depth_y.data, 9, 9);
    p.blur(d_out, mask, 1, 2.f);

    EXPECT_GE(d_out.min(), 1);
}

TEST(Blurring, doesnt_interpolate_with_nans)
{
    cv::Mat depth(9, 9, CV_32FC1);
    for (int x = 0; x < 9; x++)
    {
        for (int y = 0; y < 9; y++)
        {
            if (x < 3 && y < 3)
                depth.at<float>(x, y) = std::nanf("");
            else if (x >= 6 && y >= 6)
                depth.at<float>(x, y) = 0;
            else
                depth.at<float>(x, y) = 1;
        }
    }

    Image<float> d(9, 9);
    Image<float> d_out(9, 9);
    Image<uchar> mask(9, 9, 1);

    d.upload((float*)depth.data, 9, 9);
    DepthProcessing p(d);
    p.blur(d_out, mask, 1, 2.f);

    int nan = 2 * 4;
    int valid = d_out.size() - nan;
    EXPECT_EQ(d_out.nan(), nan);
    EXPECT_EQ(d_out.valid(), valid);
    EXPECT_EQ(d_out.nonzero(), valid);
}

TEST(Blurring, doesnt_introduce_artifacts)
{
    float val = 0.001f;
    Image<float> d(1000, 1000, val);
    Image<uchar> mask(1000, 1000, 1);
    Image<float> d_out;

    DepthProcessing p(d);
    p.blur(d_out, mask, 31, 4.f);

    float* d_data = d_out.download();
    for (int x = 0; x < 1000; x++)
    {
        for (int y = 0; y < 1000; y++)
        {
            EXPECT_FLOAT_EQ(d_data[x + y * 1000], val);
        }
    }   

    delete[] d_data;
}

TEST(Bilateral, smoothes_depth)
{
    cv::Mat depth_x(9, 9, CV_32FC1);
    cv::Mat depth_y(9, 9, CV_32FC1);
    cv::Mat depth_ref;

    float sigmaS = 2.f;
    float sigmaC = 1.f;

    for (int x = 0; x < 9; x++)
    {
        for (int y = 0; y < 9; y++)
        {
            depth_y.at<float>(x, y) = y % 2 + 1;
            depth_x.at<float>(x, y) = x % 2 + 1;
        }
    }

    Image<float> d(9, 9);
    Image<float> d_out(9, 9);
    Image<float> d_ref(9, 9);
    Image<uchar> mask(9, 9, 1);

    // Check in X direction
    d.upload((float*)depth_x.data, 9, 9);
    DepthProcessing p(d);
    p.bilateral(d_out, mask, 1, sigmaS, sigmaC);

    EXPECT_LE(d_out.mean(), 1.8);
    EXPECT_GE(d_out.mean(), 1.2);

    // Check in Y direction
    d.upload((float*)depth_y.data, 9, 9);
    p.bilateral(d_out, mask, 1, sigmaS, sigmaC);

    EXPECT_LE(d_out.mean(), 1.8);
    EXPECT_GE(d_out.mean(), 1.2);
}

TEST(Bilateral, creates_no_invalid_values_from_valid_image)
{
    Image<float> d(1000, 1000, 0.005);
    Image<uchar> mask(1000, 1000, 1);

    // Check in X direction
    DepthProcessing p(d);
    Image<float> d_out;
    p.bilateral(d_out, mask, 31, 4.f, 5.f);

    EXPECT_LE(d_out.nonzero(), d_out.size());
    EXPECT_GE(d_out.valid(), d_out.size());
}

TEST(Bilateral, doesnt_interpolate_with_zeros)
{
    cv::Mat depth_x(9, 9, CV_32FC1);
    cv::Mat depth_y(9, 9, CV_32FC1);
    cv::Mat depth_ref;

    for (int x = 0; x < 9; x++)
    {
        for (int y = 0; y < 9; y++)
        {
            depth_y.at<float>(x, y) = y % 2;
            depth_x.at<float>(x, y) = x % 2;
        }
    }

    Image<float> d(9, 9);
    Image<float> d_out(9, 9);
    Image<uchar> mask(9, 9, 1);

    // Check in X direction
    d.upload((float*)depth_x.data, 9, 9);
    DepthProcessing p(d);
    p.bilateral(d_out, mask, 3, 1.f, 2.f);

    EXPECT_GE(d_out.min(), 1);

    // Check in Y direction
    d.upload((float*)depth_y.data, 9, 9);
    p.bilateral(d_out, mask, 3, 4.f, 1.f);


    EXPECT_GE(d_out.min(), 1);
}

TEST(Bilateral, doesnt_interpolate_with_nans)
{
    cv::Mat depth(9, 9, CV_32FC1);
    for (int x = 0; x < 9; x++)
    {
        for (int y = 0; y < 9; y++)
        {
            if (x < 3 && y < 3)
                depth.at<float>(x, y) = std::nanf("");
            else if (x >= 6 && y >= 6)
                depth.at<float>(x, y) = 0;
            else
                depth.at<float>(x, y) = 1;
        }
    }

    Image<float> d(9, 9);
    Image<float> d_out(9, 9);
    Image<uchar> mask(9, 9, 1);

    d.upload((float*)depth.data, 9, 9);
    DepthProcessing p(d);
    p.bilateral(d_out, mask, 1, 4.f, 10.f);

    int nan = 2 * 4;
    int valid = d_out.size() - nan;
    EXPECT_EQ(d_out.nan(), nan);
    EXPECT_EQ(d_out.valid(), valid);
    EXPECT_EQ(d_out.nonzero(), valid);
}