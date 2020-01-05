#include <gtest/gtest.h>

#if TEST_ALBEDO

#include "image/data.h"
#include "solver/optimizer/albedo.h"

using namespace solver;

class AlbedoTest : public testing::Test
{
public:
    AlbedoTest()
     : w_(640), h_(480), 
       msAlpha_(-1.f),
       msLambda_(.5f),
       msMaxIter_(1000),
       d(std::string(SOURCE_DIR) + "/data/", w_, h_, 1),
       optimizer(w_, h_, msAlpha_, msLambda_, msMaxIter_),
       mask(d.mask),
       image(d.image),
       albedo(d.albedo),
       shading(d.shading)
    {        
    }

    const int w_, h_;
    const float msAlpha_, msLambda_;
    const int msMaxIter_;

    image::DataSet d;

    const Image<float3>& image;
    const Image<float3>& shading;
    const Image<float3>& albedo;
    const Image<uchar>& mask;

    AlbedoOptimizer optimizer;

};

/*
TEST_F(AlbedoTest, finds_solution_with_perfect_shading_grayscale)
{
    Image<float> image_gray = image.asGray<float>();
    Image<float> albedo_gray = albedo.asGray<float>();

    float msAlpha = -1.f;
    float msLambda = 0.5f;
    int msMaxIter = 1000;

    AlbedoOptimizer optimizer(msAlpha, msLambda, msMaxIter, image, mask);
    Image<float> albedo_out(image_gray.width(), image_gray.height());
    optimizer.step(albedo_out, shading);


    // Expect the image and the image created from albedo and shading to match inside mask
    Image<float> diff = image -  (shading * albedo_out);
    diff.mask(mask);

    EXPECT_LT(diff.norm2(), 10);    
}
*/

/*
TEST_F(AlbedoTest, matches_reference_result)
{
    float msAlpha = -1.f;
    float msLambda = .1f;
    int msMaxIter = 5000;

    AlbedoOptimizer optimizer(msAlpha, msLambda, msMaxIter, image, mask);
    Image<float3> albedo_out;
    Image<float3> shading(std::string(SOURCE_DIR) + "/data/shading.png");
    shading.resize(image.width(), image.height());
    shading.mask(mask);

    optimizer.step(albedo_out, shading);

    Image<float3> albedo_ref(std::string(SOURCE_DIR) + "/data/albedo_ref.png");
    albedo_ref.resize(image.width(), image.height());
    albedo_ref.mask(mask);

    albedo_out.show<cuimage::COLOR_TYPE_RGB_F>("Out", false);
    albedo_ref.show<cuimage::COLOR_TYPE_RGB_F>("Ref", false);
    auto diff = albedo_out - albedo_ref;
    diff.abs();
    diff.show<cuimage::COLOR_TYPE_RGB_F>("Diff");
    EXPECT_NEAR(0.f, diff.norm2(), 1.f);
}
*/

TEST_F(AlbedoTest, finds_solution_in_color_with_perfect_shading)
{   
    optimizer.init(image, mask, shading);
    optimizer.step(shading);

    Image<float3> diff = image - (shading * optimizer.albedo_opt);
    diff.mask(mask);
    EXPECT_LT(diff.norm2(), 25);
}

TEST_F(AlbedoTest, finds_solution_without_shading)
{
    Image<float3> shading_1(image.width(), image.height(), 1.f);
    optimizer.init(image, mask, shading_1);
    optimizer.step(shading_1);

    Image<float3> diff = image - (shading * optimizer.albedo_opt);
    diff.mask(mask);
    EXPECT_LT(diff.norm2(), 75);        
}

TEST_F(AlbedoTest, solution_better_when_using_correct_shading)
{
    Image<float3> shading_1(image.width(), image.height(), 1.f);
    optimizer.init(image, mask, shading_1);
    optimizer.step(shading_1);

    AlbedoOptimizer optimizer_with(w_, h_, msAlpha_, msLambda_, msMaxIter_);
    optimizer_with.init(image, mask, shading);
    optimizer_with.step(shading);
    
    Image<float3> diff_without = image - (shading * optimizer.albedo_opt);
    diff_without.mask(mask);
    Image<float3> diff_with = image - (shading * optimizer_with.albedo_opt);
    diff_with.mask(mask);
    
    // Expect the output with shading to be closer to orignal image than without shading information
    EXPECT_LT(diff_with.norm2(), diff_without.norm2());
}

#endif
