#include <gtest/gtest.h>

#include "image/filter/mumfordshah.h"

using namespace image;
using namespace core;

const std::string dataPath = "/home/flo/projects/depthsfs/mv-depth-srfs/libimage/test/data/";

TEST(MumfordShah, fails_for_empty_images)
{
    Image<float3> img;
    Image<float3> out;
    Image<uchar> mask;
    MumfordShahFilter<float3> f(100.f, 0.5f, 100, mask);
    EXPECT_ANY_THROW(f.compute(img, out));
}

TEST(MumfordShah, works_for_greyscale_images)
{
    Image<uchar> orig(dataPath + "cat.png");
    Image<float> img = orig.as<float>() / 255.f;
    Image<uchar> mask(img.width(), img.height(), 255);
    Image<float> result;

    MumfordShahFilter<float> f(-1.f, 0.5f, 2000, mask); // alpha=inf
    
    EXPECT_NO_THROW(f.compute(img, result));
    EXPECT_FALSE(result.empty());

    // result.show("Grey", cuimage::COLOR_TYPE_GREY_F);
}


TEST(MumfordShah, works_for_masked_greyscale_images)
{
    Image<uchar> orig(dataPath + "toy.png");
    Image<float> img = orig.as<float>() / 255.f;
    Image<uchar> mask(dataPath + "toy-mask.png");
    Image<float> result;
    
    MumfordShahFilter<float> f(1000.f, 0.5f, 1000, mask);
    
    EXPECT_NO_THROW(f.compute(img, result));
    EXPECT_FALSE(result.empty());

    // result.show("Grey", cuimage::COLOR_TYPE_GREY_F);
}



TEST(MumfordShah, works_for_color_images)
{
    Image<uchar3> orig(dataPath + "tulips.png");
    Image<float3> img = orig.as<float3>() / make_float3(255.f, 255.f, 255.f);
    Image<uchar> mask(img.width(), img.height(), 255);
    Image<float3> result;

    MumfordShahFilter<float3> f(-1.f, 0.5f, 200, mask); // alpha = inf
    
    EXPECT_NO_THROW(f.compute(img, result));
    EXPECT_FALSE(result.empty());

    // result.show("Output", cuimage::COLOR_TYPE_RGB_F);
}

TEST(MumfordShah, works_for_masked_color_images)
{
    Image<uchar3> orig(dataPath + "toy.png");
    Image<float3> img = orig.as<float3>() / make_float3(255.f, 255.f, 255.f);
    Image<uchar> mask(dataPath + "toy-mask.png");
    Image<float3> result;

    MumfordShahFilter<float3> f(-1.f, 0.5f, 200, mask); // alpha = inf
    
    EXPECT_NO_THROW(f.compute(img, result));
    EXPECT_FALSE(result.empty());

    // result.show("Output", cuimage::COLOR_TYPE_RGB_F);
}

TEST(MumfordShah, works_for_shading_updates)
{
    Image<uchar3> orig(dataPath + "toy.png");
    Image<float3> img = orig.as<float3>() / make_float3(255.f, 255.f, 255.f);
    Image<uchar> mask(dataPath + "toy-mask.png");

    Image<float3> result0, result1;
    MumfordShahFilter<float3> f(-1.f, 0.5f, 1000, mask); // alpha = inf
    f.init(img);

    Image<float3> shading0(img.width(), img.height(), make_float3(0.f, 0.f, 0.f));
    Image<float3> shading1(img.width(), img.height(), make_float3(1.f, 1.f, 1.f));

    EXPECT_NO_THROW(f.run(shading0, result0));
    EXPECT_FALSE(result0.empty());

    EXPECT_NO_THROW(f.run(shading1, result1));
    EXPECT_FALSE(result1.empty());

    EXPECT_GT((result0 - result1).norm2(), 0);
}

TEST(MumfordShah, multiple_shading_updates_produce_same_output)
{
    Image<uchar3> orig(dataPath + "toy.png");
    Image<float3> img = orig.as<float3>() / make_float3(255.f, 255.f, 255.f);
    Image<uchar> mask(dataPath + "toy-mask.png");

    Image<float3> result0, result1;
    MumfordShahFilter<float3> f(-1.f, 0.5f, 1000, mask); // alpha = inf
    f.init(img);    

    Image<float3> shading(img.width(), img.height(), make_float3(1.f, 1.f, 1.f));

    EXPECT_NO_THROW(f.run(shading, result0));
    EXPECT_FALSE(result0.empty());

    EXPECT_NO_THROW(f.run(shading, result1));
    EXPECT_FALSE(result1.empty());

    EXPECT_NEAR(0, (result0 - result1).norm2(), 1e-6);
}