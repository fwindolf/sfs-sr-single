#include "image/data.h"

using namespace image;

DataSet::DataSet(const std::string path, const int w, const int h,
    const int num, bool usecolor)
{
    std::cout << "Opening dataset " << path << std::endl;
    const std::string f_depth_lr = path + "depth.png";
    const std::string f_depth = path + "depth.exr";
    const std::string f_albedo = path + "albedo"
        + (num > 0 ? "_" + std::to_string(num) : "") + ".png";
    const std::string f_color
        = path + "color" + (num > 0 ? "_" + std::to_string(num) : "") + ".png";
    const std::string f_mask = path + "mask.png";
    const std::string f_light = path + "light.txt";

    // Default Intrinsics
    fx = 1399.19 / (1280 / (float)w);
    fy = 1399.19 / (960 / (float)h);
    cx = 640 / (1280 / (float)w);
    cy = 480 / (960 / (float)h);

    const std::string f_k_lr = path + "K_lr.txt";
    const std::string f_k_hr = path + "K_hr.txt";
    try
    {
        K_lr = std::make_shared<core::Intrinsics>(f_k_lr);
    }
    catch(const std::exception& e)
    {
        std::cout << "Could not read HR intrinsics from file: " << e.what() << std::endl;
        K_lr = std::make_shared<core::Intrinsics>(
            w / 2, h / 2, fx / 2., fy / 2., cx / 2., cy / 2.);
    }
    cam_lr = std::make_shared<core::Camera>(K_lr);

    try
    {
        K_hr = std::make_shared<core::Intrinsics>(f_k_hr);
    }
    catch(const std::exception& e)
    {
        std::cout << "Could not read LR intrinsics from file: " << e.what() << std::endl;
        K_hr = std::make_shared<core::Intrinsics>(w, h, fx, fy, cx, cy);
    }
    cam_hr = std::make_shared<core::Camera>(K_hr);

    // Light
    std::ifstream lFile(f_light);
    if (lFile.is_open())
    {
        double tmp;
        lFile >> tmp;
        l1 = static_cast<float>(tmp);
        lFile >> tmp;
        l2 = static_cast<float>(tmp);
        lFile >> tmp;
        l3 = static_cast<float>(tmp);
        lFile >> tmp;
        l4 = static_cast<float>(tmp);
        std::cout << "Read light from file (" << l1 << ", " << l2 << ", " << l3
                  << "," << l4 << ")" << std::endl;
    }
    else
    {
        l1 = 0.1;
        l2 = 0.1;
        l3 = -0.4;
        l4 = 0.2;
    }

    // Try to open all images
    Image<float> depth_lr_orig;
    bool has_depth_lr_orig = false;
    try
    {
        depth_lr_orig = Image<float>(f_depth_lr);
        has_depth_lr_orig = true;
    }
    catch (const std::exception& e)
    {
        std::cout << "Could not open LR depth: " << e.what() << std::endl;
    }

    Image<float> depth_orig;
    bool has_depth_orig = false;
    try
    {
        depth_orig = Image<float>(f_depth);
        has_depth_orig = true;
    }
    catch (const std::exception& e)
    {
        std::cout << "Could not open GT depth: " << e.what() << std::endl;
    }

    Image<uchar> mask_orig;
    bool has_mask_orig = false;
    try
    {
        mask_orig = Image<uchar>(f_mask);
        has_mask_orig = true;
    }
    catch (const std::exception& e)
    {
        std::cout << "Could not open mask: " << e.what() << std::endl;
    }

    Image<float3> albedo_orig;
    bool has_albedo_orig = false;
    try
    {
        albedo_orig = Image<float3>(f_albedo);
        has_albedo_orig = true;
    }
    catch (const std::exception& e)
    {
        std::cout << "Could not open albedo: " << e.what() << std::endl;
    }

    Image<float3> color_orig;
    bool has_color_orig = false;
    try
    {
        color_orig = Image<float3>(f_color);
        has_color_orig = true;
        std::cout << "Read color image" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "Could not open color image: " << e.what() << std::endl;
    }

    // Check for completeness
    if (!has_depth_lr_orig && !has_depth_orig)
    {
        std::cerr << "Could not load any depth!" << std::endl;
        exit(0);
    }

    // If no color was loaded and not both depth and albedo to generate
    if (!has_color_orig && !(has_depth_orig && has_albedo_orig))
    {
        std::cerr << "Unable to load/generate the color image!" << std::endl;
        exit(0);
    }

    // Init based on the opened depths
    if (!has_depth_lr_orig)
    {
        std::cout << "Using GT depth for initialization of LR depth"
                  << std::endl;
        depth_lr_orig = depth_orig;
    }
    else if (!has_depth_orig)
    {
        std::cout << "Using LR depth for initialization of GT depth"
                  << std::endl;
        depth_orig = depth_lr_orig;
    }

    // Create a mask from valid depth pixels if none was loaded
    if (!has_mask_orig)
    {
        auto depth_mask = depth_orig;
        depth_mask.threshold(0.1f, 0.f, 1.f);
        mask_orig = depth_mask.as<uchar>() * (uchar)255;
    }

    // If no albedo provided, use the color image
    if (!has_albedo_orig)
    {
        std::cout << "Using color image as albedo!" << std::endl;
        albedo_orig.copyFrom(color_orig);
    }

    depth_lr.copyFrom(depth_lr_orig);
    if (depth_lr.height() > h || depth_lr.width() > w)
    {
        std::cout << "Depth LR currently contains " << depth_lr.nan()
                  << " NaNs " << std::endl;
        depth_lr.resize(w, h, cuimage::LINEAR_NONZERO);
    }

    albedo.copyFrom(albedo_orig);
    albedo.resize(w, h, cuimage::LINEAR);

    mask_lr.copyFrom(mask_orig);
    mask_lr.resize(depth_lr.width(), depth_lr.height(), mask_orig);

    mask.copyFrom(mask_orig);
    mask.resize(w, h, mask_orig);

    depth_star.copyFrom(depth_orig);
    if (depth_star.height() != h || depth_star.width() != w)
    {
        std::cout << "Depth GT currently contains " << depth_star.nan()
                  << " NaNs " << std::endl;
        depth_star.resize(w, h, cuimage::LINEAR_NONZERO);
    }

    image::DepthProcessing pDepth(depth_star);
    float3* l = new float3[4]{make_float3(l1, l1, l1), make_float3(l2, l2, l2),
        make_float3(l3, l3, l3), make_float3(l4, l4, l4)};
    light.upload(l, 1, 4);
    pDepth.shading(shading, light, K_hr);

    if (has_color_orig
        && (color_orig.width() != w || color_orig.height() != h))
    {
        std::cout << "Color image is not in requested resolution of (" << w
                  << "x" << h << ")!" << std::endl;
        // Masked resize to not introduce darker seams around valid region of
        // image in case outside is black
        auto tmp_mask = mask.resized(color_orig.width(), color_orig.height(),
            mask, cuimage::NEAREST);
        color_orig.resize(w, h, tmp_mask, cuimage::LINEAR);
    }
    color.copyFrom(color_orig);

    if (usecolor && has_color_orig)
    {
        std::cout << "Prefer using color image" << std::endl;
        image = color;
    }
    else if (has_albedo_orig && has_depth_orig)
    {
        std::cout << "Using generated image (from shading * albedo)"
                  << std::endl;
        image = shading * albedo;
    }
    else
    {
        std::cout << "Using color image" << std::endl;
        image = color;
    }

    albedo.mask(mask);
    image.mask(mask);
    shading.mask(mask);
    depth_star.mask(mask);
    depth_lr.mask(mask_lr);

    delete[] l;
}