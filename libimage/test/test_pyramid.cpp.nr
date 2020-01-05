#include <gtest/gtest.h>

#include "image/pyramid.h"

#include "core/image.h"

using namespace testing;
using namespace image;
using namespace core;

struct ImageData
{
    ImageData(int w, int h, int c)
        : w(w), h(h), c(c) {}
    
    int w, h, c;
};

class ImagePyramidCPU : public Test
{
public:
    ImagePyramidCPU()
    : c(4, 5, 2), 
      f(32, 40, 2),
      levels(3),
      pyramid(levels)
    {
        cv::Mat cvCoarseZeros = cv::Mat::zeros(c.h, c.w, CV_32FC(c.c));
        cv::Mat cvCoarseOnes = cv::Mat(c.h, c.w, CV_32FC(c.c), cv::Scalar(1.f, 1.f, 1.f));
        cv::Mat cvCoarseEye = cv::Mat::eye(c.h, c.w, CV_32FC1);
        cv::Mat cvCoarseId = cv::Mat(cvCoarseEye.size(), CV_32FC(c.c));
        int coarse_from_to[2 * c.c];
        for (int i = 0; i < c.c; i++)
        {
            coarse_from_to[2 * i] = 0;
            coarse_from_to[2 * i + 1] = i;
        }
        cv::mixChannels(&cvCoarseEye, 1, &cvCoarseId, 1, coarse_from_to, c.c);

        cv::Mat cvFineZeros = cv::Mat::zeros(f.h, f.w, CV_32FC(f.c));
        cv::Mat cvFineOnes = cv::Mat(f.h, f.w, CV_32FC(f.c), cv::Scalar(1.f, 1.f, 1.f));
        cv::Mat cvFineEye = cv::Mat::eye(f.h, f.w, CV_32FC1);
        cv::Mat cvFineId = cv::Mat(cvFineEye.size(), CV_32FC(f.c));
        int fine_from_to[2 * f.c];
        for (int i = 0; i < f.c; i++)
        {
            fine_from_to[2 * i] = 0;
            fine_from_to[2 * i + 1] = i;
        }
        cv::mixChannels(&cvFineEye, 1, &cvFineId, 1, fine_from_to, f.c);
        
        coarse_zeros = std::make_shared<Image>(cvCoarseZeros);
        coarse_ones = std::make_shared<Image>(cvCoarseOnes);
        coarse_id = std::make_shared<Image>(cvCoarseId);

        fine_zeros = std::make_shared<Image>(cvFineZeros);
        fine_ones = std::make_shared<Image>(cvFineOnes);
        fine_id = std::make_shared<Image>(cvFineId);
    }

    ~ImagePyramidCPU(){}

    ImageData c, f;

    int levels;
    ImagePyramid pyramid;

    ImagePtr coarse_zeros, coarse_ones, coarse_id;
    ImagePtr fine_zeros, fine_ones, fine_id;
};

TEST_F(ImagePyramidCPU, is_created_empty_and_will_not_return_content)
{
    EXPECT_FALSE(pyramid.isBuilt());
    EXPECT_EQ(nullptr, pyramid.finest());
    EXPECT_EQ(nullptr, pyramid.coarsest());
}

TEST_F(ImagePyramidCPU, can_be_built_downwards_from_coarsest)
{
    EXPECT_TRUE(pyramid.buildDownwards(coarse_zeros));
    EXPECT_TRUE(pyramid.isBuilt());

    EXPECT_NE(nullptr, pyramid.finest());
    EXPECT_NE(nullptr, pyramid.coarsest());
}

TEST_F(ImagePyramidCPU, doubles_image_size_when_building_downwards)
{
    ASSERT_TRUE(pyramid.buildDownwards(coarse_zeros));
    ASSERT_TRUE(pyramid.isBuilt());

    int w = c.w;
    int h = c.h;
    for (int i = levels - 1; i >= 0; i--)
    {
        auto img = pyramid.at(i);
        EXPECT_NE(nullptr, img);

        EXPECT_EQ(w, img->width());
        EXPECT_EQ(h, img->height());
        EXPECT_EQ(c.c, img->channels());

        w *= 2;
        h *= 2;
    }
}

TEST_F(ImagePyramidCPU, keeps_zeros_when_building_downwards)
{
    ASSERT_TRUE(pyramid.buildDownwards(coarse_zeros));
    ASSERT_TRUE(pyramid.isBuilt());

    auto finest_zeros = pyramid.finest();

    ASSERT_NE(nullptr, finest_zeros);

    auto data = (float*)finest_zeros->data();
    for (int i = 0; i < finest_zeros->height() * finest_zeros->width() * finest_zeros->channels(); i++)
    {
        EXPECT_FLOAT_EQ(0.f, data[i]);
    }
}

TEST_F(ImagePyramidCPU, keeps_ones_when_building_downwards)
{
    ASSERT_TRUE(pyramid.buildDownwards(coarse_ones));
    ASSERT_TRUE(pyramid.isBuilt());

    auto finest_ones = pyramid.finest();

    ASSERT_NE(nullptr, finest_ones);

    auto data = (float*)finest_ones->data();
    for (int i = 0; i < finest_ones->height() * finest_ones->width() * finest_ones->channels(); i++)
    {
        EXPECT_FLOAT_EQ(1.f, data[i]);
    }
}

TEST_F(ImagePyramidCPU, keeps_identity_shape_when_building_downwards)
{
    ASSERT_TRUE(pyramid.buildDownwards(coarse_id));
    ASSERT_TRUE(pyramid.isBuilt());

    auto finest_id = pyramid.finest();

    ASSERT_NE(nullptr, finest_id);
    auto data = (float*)finest_id->data();
    
    // UR and LL corners
    EXPECT_FLOAT_EQ(0.f, data[f.w - 1]);
    EXPECT_FLOAT_EQ(0.f, data[f.w * (f.h - 1)]);

    // UL and LR corners
    EXPECT_EQ(1.f, data[0]);
    EXPECT_GT(0.5f, data[f.w * f.h - 1]);

}

TEST_F(ImagePyramidCPU, can_be_built_upwards_from_finest)
{
    EXPECT_TRUE(pyramid.buildUpwards(fine_zeros));
    EXPECT_TRUE(pyramid.isBuilt());

    EXPECT_NE(nullptr, pyramid.finest());
    EXPECT_NE(nullptr, pyramid.coarsest());
}

TEST_F(ImagePyramidCPU, halves_image_size_when_building_upwards)
{
    ASSERT_TRUE(pyramid.buildUpwards(fine_zeros));
    ASSERT_TRUE(pyramid.isBuilt());

    int w = f.w;
    int h = f.h;
    for (int i = 0; i < levels; i++)
    {
        auto img = pyramid.at(i);
        EXPECT_NE(nullptr, img);

        EXPECT_EQ(w, img->width());
        EXPECT_EQ(h, img->height());
        EXPECT_EQ(c.c, img->channels());

        w /= 2;
        h /= 2;
    }
}

TEST_F(ImagePyramidCPU, keeps_zeros_when_building_upwards)
{
    ASSERT_TRUE(pyramid.buildUpwards(fine_zeros));
    ASSERT_TRUE(pyramid.isBuilt());

    auto coarsest_zeros = pyramid.coarsest();

    ASSERT_NE(nullptr, coarsest_zeros);

    auto data = (float*)coarsest_zeros->data();
    for (int i = 0; i < coarsest_zeros->height() * coarsest_zeros->width() * coarsest_zeros->channels(); i++)
    {
        EXPECT_FLOAT_EQ(0.f, data[i]);
    }
}

TEST_F(ImagePyramidCPU, keeps_ones_when_building_upwards)
{
    ASSERT_TRUE(pyramid.buildUpwards(fine_ones));
    ASSERT_TRUE(pyramid.isBuilt());

    auto coarsest_zeros = pyramid.coarsest();

    ASSERT_NE(nullptr, coarsest_zeros);

    auto data = (float*)coarsest_zeros->data();
    for (int i = 0; i < coarsest_zeros->height() * coarsest_zeros->width() * coarsest_zeros->channels(); i++)
    {
        EXPECT_FLOAT_EQ(1.f, data[i]);
    }
}

TEST_F(ImagePyramidCPU, keeps_identity_shape_when_building_upwards)
{
    ASSERT_TRUE(pyramid.buildUpwards(fine_id));
    ASSERT_TRUE(pyramid.isBuilt());

    auto coarsest_id = pyramid.coarsest();

    ASSERT_NE(nullptr, coarsest_id);
    auto data = (float*)coarsest_id->data();
    
    for (int i = 0; i < std::min(coarsest_id->height(), coarsest_id->width()); i++)
    {
        int idx = i * coarsest_id->width() + i;
        EXPECT_GT(data[idx], 0.f);
    }
}


class IntrinsicsPyramidCPU : public Test
{
public:
    IntrinsicsPyramidCPU() 
     : levels(4),
       pyramid(levels)
    {
        intrinsics = std::make_shared<PinholeIntrinsics>(640, 480, 520, 520, 320, 240);
    }

    int levels;
    IntrinsicsPyramid pyramid;
    IntrinsicsPtr intrinsics;
};

TEST_F(IntrinsicsPyramidCPU, is_created_empty_and_will_not_return_valid_intrinsics)
{
    EXPECT_FALSE(pyramid.isBuilt());
    EXPECT_EQ(nullptr, pyramid.finest());
    EXPECT_EQ(nullptr, pyramid.coarsest());
}

TEST_F(IntrinsicsPyramidCPU, can_be_built_upwards_from_finest)
{
    EXPECT_TRUE(pyramid.buildUpwards(intrinsics));
    EXPECT_TRUE(pyramid.isBuilt());
}

TEST_F(IntrinsicsPyramidCPU, has_valid_entries_at_all_levels)
{
    ASSERT_TRUE(pyramid.buildUpwards(intrinsics));
    ASSERT_TRUE(pyramid.isBuilt());

    for (int i = 0; i < levels; i++)
    {
        auto lvl_intrinsics = pyramid.at(i);
        EXPECT_NE(nullptr, lvl_intrinsics);
        EXPECT_FALSE(lvl_intrinsics->matrix().isZero());
    }
}
