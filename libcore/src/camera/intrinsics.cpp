#include "core/camera/intrinsics.h"

#include <fstream>

using namespace core;

Intrinsics::Intrinsics()
    : matrix_(Eigen::Matrix3f::Identity())
    , width_(0)
    , height_(0)
{
}

int testResolution(float dimension)
{
    std::vector<int> resolutions = {1920, 1600, 1296, 1280, 1200, 1080, 1024, 968, 960,
        800, 768, 720, 640, 512, 480, 400, 320, 240};

    for (int i = 1; i < resolutions.size(); i++)
    {
        if (dimension > (resolutions[i - 1] + resolutions[i]) / 2.f)
            return resolutions[i - 1];
    }
    return 0;
}

Intrinsics::Intrinsics(const std::string& fileName)
    : matrix_(Eigen::Matrix3f::Identity())
{
    std::ifstream f(fileName);
    if (!f.is_open())
        throw std::runtime_error(
            "Could not read intrinsics: Failed to open " + fileName);

    // Get the number of lines
    int numLines = 0;
    std::string unused;
    while ( std::getline(f, unused) )
        ++numLines;

    f.clear(); // In case we reached EOF
    f.seekg(0, std::ios::beg);

    int w, h;
    float fx, fy, cx, cy;
    float discard;

    if (numLines == 5)
    {
        // w, h plus 4x4
        f >> w >> h;
        f >> fx >> discard >> cx >> discard;
        f >> discard >> fy >> cy >> discard;
        w = testResolution(2 * cx);
        h = testResolution(2 * cy);
    }
    else if (numLines == 4)
    {
        // Try to read w, h
        f >> w >> h;
        if (w < 100 || w > 2000 || h < 100 || h > 2000)
        {
            // Not successful
            f.clear(); // In case we reached EOF
            f.seekg(0, std::ios::beg);

            // 4x4
            f >> fx >> discard >> cx >> discard;
            f >> discard >> fy >> cy >> discard;
            w = testResolution(2 * cx);
            h = testResolution(2 * cy);
        }
        else
        {
            // w, h plus 3x3 afterwards
            f >> fx >> discard >> cx;
            f >> discard >> fy >> cy;
        }
    }
    else if (numLines == 3)
    {
        // 3x3
        f >> fx >> discard >> cx;
        f >> discard >> fy >> cy;

        w = testResolution(2 * cx);
        h = testResolution(2 * cy);
    }

    // Update this
    setDimensions(w, h);
    setFocalLength(fx, fy);
    setCenter(cx, cy);

    print();
}

Intrinsics::Intrinsics(const int width, const int height)
    : matrix_(Eigen::Matrix3f::Identity())
    , width_(width)
    , height_(height)
{
}

Intrinsics::Intrinsics(
    const int width, const int height, const Eigen::Matrix3f& matrix)
    : matrix_(matrix)
    , width_(width)
    , height_(height)
{
}

Intrinsics::Intrinsics(const int width, const int height, const float fx,
    const float fy, const float cx, const float cy)
    : Intrinsics(width, height)
{
    setFocalLength(fx, fy);
    setCenter(cx, cy);
}

Intrinsics::~Intrinsics() {}

void Intrinsics::set(
    const int width, const int height, const Eigen::Matrix3f& matrix)
{
    width_ = width;
    height_ = height;
    matrix_ = matrix;
}

void Intrinsics::setDimensions(const int width, const int height)
{
    width_ = width;
    height_ = height;
}

void Intrinsics::setFocalLength(const float fx, const float fy)
{
    matrix_(0, 0) = fx;
    matrix_(1, 1) = fy;
}

void Intrinsics::setCenter(const float cx, const float cy)
{
    matrix_(0, 2) = cx;
    matrix_(1, 2) = cy;
}

void Intrinsics::setMatrix(const Eigen::Matrix3f& matrix) { matrix_ = matrix; }

Eigen::Matrix3f Intrinsics::matrix() const { return matrix_; }

Eigen::Matrix3f Intrinsics::inverse() const { return matrix_.inverse(); }

int Intrinsics::width() const { return width_; }

int Intrinsics::height() const { return height_; }

float Intrinsics::fx() const { return matrix_(0, 0); }

float Intrinsics::fy() const { return matrix_(1, 1); }

float Intrinsics::cx() const { return matrix_(0, 2); }

float Intrinsics::cy() const { return matrix_(1, 2); }

void Intrinsics::print() const
{
    std::cout << "Pinhole Intrinsics " << width_ << "x" << height_ << std::endl
              << "fx=" << fx() << " fy=" << fy() << std::endl
              << "cx=" << cx() << " cy=" << cy() << std::endl;
}