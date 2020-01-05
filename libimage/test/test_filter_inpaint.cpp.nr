#include "test.h"

#include "image/filter/inpaint.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace image;

const std::string dataPath = "../libimage/test/data/";

TEST(FilterInpaint, fails_for_empty_images)
{
    auto img = std::make_shared<core::Image>(5, 4, 3, floatT, cpuL);
    InpaintFilter f;
    EXPECT_ANY_THROW(f.compute(img));
}

TEST(FilterInpaint, works_with_nans)
{
    cv::Mat I = cv::imread(dataPath + "cat.png", cv::IMREAD_GRAYSCALE); // From build
    I.convertTo(I, CV_32FC1, 1.f/255.f);
    ASSERT_FALSE(I.empty());

    // corrupt image
    float percentage_intact = 80.f;
    cv::Mat rnd(I.cols, I.rows, CV_8UC1);
    cv::randu(rnd, cv::Scalar(0), cv::Scalar(100)); 

    cv::Mat Icorrupt = I.clone();
    Icorrupt.setTo(nanf(""), (rnd >= percentage_intact));
    auto img = std::make_shared<core::Image>(Icorrupt);

    InpaintFilter f;
    auto result = f.compute(img);

    cv::Mat reconstructed;
    result->to(reconstructed);
    
    ASSERT_TRUE((bool)result);
    EXPECT_FALSE(result->empty());

    cv::imshow("Corrupt", Icorrupt);
    cv::imshow("Reconstructed", reconstructed);    
    cv::waitKey(0);
    exit(-1);
}