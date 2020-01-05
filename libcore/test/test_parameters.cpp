#include <gtest/gtest.h>

#include <cuda_runtime.h>

/*

#include "core/util/parameters.h"

using namespace core;
using namespace testing;

TEST(IntParameter, can_be_created_from_temporary_variable)
{
    IntParameter * p;
    {
        int i = 3;
        p = new IntParameter(i);
    }
    
    int* i = (int*)p->data();
    EXPECT_EQ(*i, 3);
}

TEST(IntParameter, does_not_leak)
{
    int* i;
    {
        IntParameter p(5);
        i = (int*)p.data();
        EXPECT_EQ(*i, 5);
    }
    
    EXPECT_NE(*i, 5); // might just get lucky?
}

TEST(ImageParameter, can_access_image_data)
{
    int w = 10, h = 5, c = 2;
    auto data = new float[w * h * c]{1.f, 2.f, 3.f};
    auto ip = std::make_shared<Image>(h, w, c, data, Image::Type::Float, Image::Location::CPU);

    auto idata = (float*)ip->data();
    EXPECT_EQ(data[0], idata[0]);
    EXPECT_EQ(0, idata[w * h * c - 1]);

    // Image deletes data
}

TEST(ImageParameter, can_access_image_data_after_upload)
{
    int w = 10, h = 5, c = 2;
    auto data = new float[w * h * c]{1.f, 2.f, 3.f};
    auto ip = std::make_shared<Image>(h, w, c, data, Image::Type::Float, Image::Location::CPU);
    ip->upload();

    auto idata = new float[w * h * c];
    cudaMemcpy(idata, (float*)ip->data(), w * h * c * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_EQ(data[0], idata[0]);
    EXPECT_EQ(0, idata[w * h * c - 1]);

    // Image deletes data
    delete idata;
}

TEST(ImageParameter, can_access_image_data_after_download)
{
    int w = 10, h = 5, c = 2;
    auto data = new float[w * h * c]{1.f, 2.f, 3.f};

    auto ip = std::make_shared<Image>(h, w, c, data, Image::Type::Float, Image::Location::CPU);
    ip->upload();
    ip->download();

    auto idata = (float*)ip->data();

    EXPECT_EQ(data[0], idata[0]);
    EXPECT_EQ(0, idata[w * h * c - 1]);

    // Image deletes data and idata
}


TEST(NamedParameters, can_store_different_types)
{
    NamedParameters params;
    
    int myint = 5;
    params.add("myint", myint);


    int w = 10, h = 5, c = 2;
    auto myimage = std::make_shared<Image>(h, w, c, Image::Type::Float, Image::Location::CPU);
    params.add("myimage", myimage);

    EXPECT_EQ(2, params.size());
}

TEST(NamedParameters, can_be_retrieved_by_name)
{
    NamedParameters params;
    params.add("myint0", 3);    
    params.add("myint1", 1);

    EXPECT_EQ("myint0", params.at("myint0").name());
    EXPECT_EQ("myint1", params.at("myint1").name());
}

TEST(NamedParameters, can_be_retrieved_by_position)
{
    NamedParameters params;
    params.add("myint0", 3);    
    params.add("myint1", 1);

    EXPECT_EQ("myint0", params.at(0).name());
    EXPECT_EQ("myint1", params.at(1).name());
}

TEST(NamedParameters, can_insert_in_different_order)
{
    NamedParameters params;
    params.add("myint3", 3, 3);    
    params.add("myint2", 1, 1);
    params.add("myint1", 2, 2);
    params.add("myint0", 0, 0);

    EXPECT_EQ(4, params.size());
    // Check if the data is correct by casting void* data to int* and accessing
    EXPECT_EQ(0, *(int*)params.at(0).data());
    EXPECT_EQ(1, *(int*)params.at(1).data());
    EXPECT_EQ(2, *(int*)params.at(2).data());
    EXPECT_EQ(3, *(int*)params.at(3).data());
}

TEST(NamedParameters, return_all_parameter_names_in_correct_order)
{
    NamedParameters params;
    std::vector<std::string> names = { "i1", "i2", "i3", "i4"};
    for(auto& name : names)
        params.add(name, 0);
    
    EXPECT_EQ(names.size(), params.size());
    
    auto pnames = params.names();
    ASSERT_EQ(names.size(), pnames.size());

    for(size_t i = 0; i < pnames.size(); i++)
        EXPECT_EQ(pnames.at(i), names.at(i));
}

TEST(NamedParameters, return_data_in_correct_order)
{
    NamedParameters params;
    params.add("i3", 3, 3);

    int w = 10, h = 5, c = 2;
    auto myimage = std::make_shared<Image>(h, w, c, Image::Type::Float, Image::Location::CPU);
    params.add("i1", myimage, 1);

    params.add("i0", 0, 0);
    params.add("i2", 2, 2);

    auto data = params.data();
    EXPECT_EQ(0, *(int*)data[0]);
    EXPECT_EQ(myimage->data(), data[1]);
    EXPECT_EQ(2, *(int*)data[2]);
    EXPECT_EQ(3, *(int*)data[3]);
}
*/