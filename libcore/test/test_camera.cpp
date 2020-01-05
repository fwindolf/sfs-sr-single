#include <gtest/gtest.h>

#include "core/camera.h"

using namespace core;
using namespace testing;

class IntrinsicsTest : public Test
{
public:
    IntrinsicsTest()
     : K(Eigen::Matrix3f::Identity()),
       width(640),
       height(480)
    {
        K(0, 0) = fx;
        K(1, 1) = fy;
        K(0, 2) = width/2.f;
        K(1, 2) = height/2.f;
    }

    int width;
    int height;

    float fx = 600;
    float fy = 400;
    Eigen::Matrix3f K;
};


TEST_F(IntrinsicsTest, can_be_created_from_matrix)
{
    Intrinsics i(width, height, K);
    EXPECT_EQ(height, i.height());
    EXPECT_EQ(width, i.width());
    EXPECT_EQ(fx, i.fx());
    EXPECT_EQ(fy, i.fy());
    EXPECT_EQ(width/2.f, i.cx());
    EXPECT_EQ(height/2.f, i.cy());
}

TEST_F(IntrinsicsTest, can_be_set_later)
{
    Intrinsics i;
    i.setDimensions(width, height);
    i.setFocalLength(fx, fy);
    i.setCenter(width/2, height/2);

    EXPECT_EQ(height, i.height());
    EXPECT_EQ(width, i.width());
    EXPECT_EQ(fx, i.fx());
    EXPECT_EQ(fy, i.fy());
    EXPECT_EQ(width/2.f, i.cx());
    EXPECT_EQ(height/2.f, i.cy());
}


class CameraTest : public Test
{
public:
    CameraTest() 
     : ic(new Intrinsics(width, height, 600, 500, 320, 240)),
       id(new Intrinsics(width, height, 400, 450, 320, 240))
    {
    }
    int width = 640;
    int height = 480;

    IntrinsicsPtr ic, id;
};


TEST_F(CameraTest, can_be_created_empty)
{
    Camera c;
    EXPECT_EQ(c.colorIntrinsics()->matrix(), Eigen::Matrix3f::Identity());
    EXPECT_EQ(c.depthIntrinsics()->matrix(), Eigen::Matrix3f::Identity());
}

TEST_F(CameraTest, can_be_created_from_two_intrinsics)
{
    Camera c(ic, id);
    EXPECT_EQ(c.colorIntrinsics(), ic);
    EXPECT_EQ(c.depthIntrinsics(), id);
}

TEST_F(CameraTest, can_be_created_from_single_intrinsics)
{
    Camera c(ic);
    EXPECT_EQ(c.colorIntrinsics(), ic);
    EXPECT_EQ(c.depthIntrinsics(), ic);
}








