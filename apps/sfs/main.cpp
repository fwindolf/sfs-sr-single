#include "image/data.h"
#include "optimizer.h"
#include "parameters.h"

#include <CLI.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace core;
using namespace image;
using namespace solver;

int main(int argc, char* argv[])
{
    SfsParameters params;

    CLI::App app{"SFS Application"};

    app.add_option("-a, --optim_alpha", params.alpha,
        "Albedo Update : Smoothness of albedo, -1 (=infinity) for piecewise "
        "constant",
        true);
    app.add_option("-l, --optim_lambda", params.lambda,
        "Albedo Update : Tradeoff between smoothness and number of jumps",
        true);
    app.add_option("-i, --optim_iter", params.maxIter,
        "Albedo Update : Maximum number of iterations per step", true);

    app.add_option("-g, --optim_gamma", params.gamma,
        "Theta Update  : Influence of the SfS term", true);
    app.add_option("-n, --optim_nu", params.nu,
        "Theta Update  : Minimal surface weight for the output depth", true);
    app.add_option("-m, --optim_mu", params.mu,
        "Depth Update  : Weight controlling the influence of the original "
        "depth",
        true);

    app.add_option("-k, --admm_kappa", params.kappa,
        "ADMM Parameter: Initial Step size for the dual update", true);
    app.add_option("-t, --admm_tau", params.tau,
        "ADMM Parameter: Penalty parameter, by which kappa is increased per "
        "iteration",
        true);
    app.add_option("-o, --admm_tolerance", params.tolerance,
        "ADMM Parameter: Tolerance of the relative error between theta and "
        "thetaZ",
        true);
    app.add_option("-e, --admm_tolerance_EL", params.toleranceEL,
        "ADMM Parameter: Tolerance of the residual of the primal dual update",
        true);

    std::string dataPath = std::string(SOURCE_DIR) + "/data/android/";
    int num = 0;
    std::vector<int> resolution = { 640, 480 };

    app.add_option("-d, --dataset_path", dataPath,
        "Data Parameter: Path to the root directory of data", true);
    app.add_option("-f, --dataset_frame_num", num,
        "Data Parameter: Number/Frame of in a set of data", true);
    app.add_option("-r, --dataset_resolution", resolution,
           "Data Parameter: Width of the upsampled data", true);

    float sigma = 0.f;
    app.add_option("-b, --dataset_depth_sigma", sigma, 
        "Data Parameter: Blur the initial depth");

    bool use_gt_depth = false;
    bool use_gt_albedo = false;
    bool use_gt_light = false;
    bool smooth_depth = false;
    bool prefer_image = false;
    app.add_flag("--dataset_gt_depth", use_gt_depth,
        "Data Parameter: Use optimal depth for LR depth (will be bilaterally "
        "filtered before usage)");
    app.add_flag("--dataset_gt_albedo", use_gt_albedo,
        "Data Parameter: Use optimal albedo as input");
    app.add_flag("--dataset_gt_light", use_gt_light,
        "Data Parameter: Use optimal light as input");
    app.add_flag("--dataset_smooth_depth", smooth_depth,
        "Data Parameter: Smooth depth initialization");
    app.add_flag("--dataset_prefer_image", prefer_image,
        "Data Parameter: Use the loaded image over the generated");

    std::string resultsFolder = "";
    std::string run = "";
    app.add_option("-s, --output_results_folder", resultsFolder,
        "Out Parameter : Path to save output images, results.");
    app.add_option("-p, --output_run_folder", run,
        "Out Parameter : Run folder to save output to.");

    int it_theta_outer = 2;
    int it_theta_inner = 3;
    int it_depth_outer = 1;
    int it_depth_inner = 3;
    app.add_option("--iter_theta_outer", it_theta_outer,
        "Iterations Parameter: Number of outer theta iterations");
    app.add_option("--iter_theta_inner", it_theta_inner,
        "Iterations Parameter: Number of inner theta iterations");
    app.add_option("--iter_depth_outer", it_depth_outer,
        "Iterations Parameter: Number of outer depth iterations");
    app.add_option("--iter_depth_inner", it_depth_inner,
        "Iterations Parameter: Number of inner depth iterations");

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    std::cout << "Use GT depth: " << use_gt_depth << std::endl;
    std::cout << "Use GT albedo: " << use_gt_albedo << std::endl;
    std::cout << "Use GT light: " << use_gt_light << std::endl;

    DataSet d(dataPath, resolution[0], resolution[1], num, prefer_image);
    Image<float> depth;
    if (use_gt_depth)
        depth.copyFrom(d.depth_star);

    std::cout << "Optimizer with theta (" << it_theta_outer << "/"
              << it_theta_inner << ") and  depth (" << it_depth_outer << "/"
              << it_depth_inner << ")" << std::endl;


    std::cout << "Sigma: " << sigma << std::endl;
    if (sigma > 0.f)
    {   
        Image<float> depth_init;
        if (depth.empty())
            depth_init = d.depth_lr.resized(resolution[0], resolution[1], cuimage::LINEAR_NONZERO);
        else
            depth_init = depth;

        DepthProcessing p(depth_init);
        //p.blur(depth, d.mask, 0.01 * depth_init.width(), sigma);
        p.bilateral(depth, d.mask, 0.01 * depth_init.width(), sigma, sigma);
    }

    //d.albedo.show<cuimage::COLOR_TYPE_RGB_F>("Albedo");

    SfSOptimizer optimizer(d.image, d.mask, d.depth_lr, depth, d.K_hr, params,
        it_theta_outer, it_theta_inner, it_depth_outer, it_depth_inner,
        smooth_depth);

    optimizer.init();

    if (use_gt_albedo)
        optimizer.useAlbedo(d.albedo);
    if (use_gt_light)
        optimizer.useLight(d.light);

    optimizer.run();

    if (resultsFolder.empty())
    {
        optimizer.evaluate(d.depth_star, "");
        optimizer.visualize();
    }
    else
    {
        optimizer.evaluate(d.depth_star, resultsFolder, run);
        optimizer.save(resultsFolder, run);
    }
}