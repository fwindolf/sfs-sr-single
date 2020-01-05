import os
import subprocess
import json
from tqdm import tqdm
import argparse

build_dir = "../../../build/"
data_dir = "/data/real/"
output_dir = "/data/eval/sfs/real/"

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='/data/synthetic/')
parser.add_argument('-o', '--output', default='/data/eval/sfs/synthetic/')

args = parser.parse_args()

data_dir = args.dataset
output_dir = args.output

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

folder = "{res_x}x{res_y}_{tau}_{kappa}__{gamma}_{nu}_{mu}__{lambda}_{iters}"

def make_sfs():
    args = [
        "make", "-j", "8", "AppSfs"
    ]
    subprocess.Popen(
        args,
        cwd=os.path.abspath(build_dir)
    ).wait()

def run_sfs(run):
    run_folder = folder.format(**run)

    if not os.path.exists(os.path.join(output_dir, run_folder)):
        # Create output directory
        os.mkdir(os.path.join(output_dir, run_folder))
        
        # Run AppSfs with the right parameters
        args = [
            os.path.abspath(build_dir + "/bin/AppSfs"),
            "--dataset_path", os.path.abspath(data_dir) + "/",
            "--dataset_resolution", run['res_x'], run['res_y'],
            "--dataset_frame_num", 0,
            "--admm_kappa", run['kappa'],
            "--admm_tau", run['tau'],
            "--optim_gamma", run['gamma'],
            "--optim_mu", run['mu'],
            "--optim_nu", run['nu'],
            "--optim_lambda", run['lambda'],
            "--optim_iter", run['iters'],
            "--output_results_folder", os.path.abspath(output_dir) + "/",
            "--output_run_folder", run_folder
        ]
        args = [str(a) for a in args]

        with open(os.path.join(output_dir, run_folder, 'output.txt'), 'w') as f:
            subprocess.Popen(
                args,
                cwd=os.path.abspath(build_dir), 
                stdout=f, 
                stderr=f
            ).wait()

    # Open the evaluation files
    output = dict()
    with open(os.path.join(output_dir, run_folder, "results.txt")) as f_res:
        line = f_res.readline().strip().split(';')        
        output['meanErr'] = float(line[0])
        output['medianErr'] = float(line[1])
        output['rmsErr'] = float(line[2])
    
    return output

# General
resolutions = [[1280, 960], [640, 480]]

# ADMM
kappa_taus = [
    (1e-2, 1.5),
    (1e-3, 2),
    (1e-4, 2),
    (1e-5, 2),
    (1e-5, 5),
    (1e-6, 5),
    (1e-6, 10)
]

# SFS
#gammas = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20, 50]
#nus = [0, 0.0001, 0.001, 0.01, 0.1,]
#mus = [0, 0.0001, 0.001, 0.01, 0.1, 1]

gammas = [0.1, 0.2, 0.5, 1]
nus = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
mus = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

# Albedo
lambdas = [0.05, 0.1, 0.2]
iters = [200, 400]

# SFS 1280x680 GAMMA SWEEP 
#resolutions = [[1280, 960]]
#kappa_taus = [(1e-4, 5), (1e-2, 2)]
#gammas = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20, 50, 100]
#mus = [0.001]
#nus = [0.0001]
#lambdas = [0.5]
#iters = [400]

# SFS 640x480 GAMMA SWEEP 
#resolutions = [[640, 480]]
#kappa_taus = [(1e-5, 10), (1e-3, 2)]
#gammas = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20, 50, 100]
#mus = [0.01]
#nus = [0.0001]
#lambdas = [0.5]
#iters = [400]

# SFS 1280x960 NU SWEEP 
#resolutions = [[1280, 960]]
#kappa_taus = [(1e-4, 5), (1e-2, 2)]
#gammas = [1.0]
#mus = [0.001]
#nus = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]
#lambdas = [0.5]
#iters = [400]

# SFS 640x480 NU SWEEP 
#resolutions = [[640, 480]]
#kappa_taus = [(1e-5, 10), (1e-3, 2)]
#gammas = [1]
#mus = [0.01]
#nus = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]
#lambdas = [0.5]
#iters = [400]

# SFS 1280x960 MU SWEEP 
#resolutions = [[1280, 960]]
#kappa_taus = [(1e-4, 5), (1e-2, 2)]
#gammas = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20, 50, 100]
#mus = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]
#nus = [0.0001]
#lambdas = [0.5]
#iters = [400]

# SFS 640x480 MU SWEEP 
#resolutions = [[640, 480]]
#kappa_taus = [(1e-5, 10), (1e-3, 2)]
#gammas = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20, 50, 100]
#mus = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]
#nus = [0.0001]
#lambdas = [0.5]
#iters = [400]

# Make application
make_sfs()

# Run evaluations
runs = list()
evaluations = dict()

for res in resolutions:
    r = "{}x{}".format(res[0], res[1])

    if not r in evaluations:
        evaluations[r] = dict()

    for k, t in kappa_taus:
        if not k in evaluations[r]:
            evaluations[r][k] = dict()
        if not t in evaluations[r][k]:
            evaluations[r][k][t] = dict()

        for g in gammas:
            if not g in evaluations[r][k][t]:
                evaluations[r][k][t][g] = dict()

            for n in nus:
                if not n in evaluations[r][k][t][g]:
                    evaluations[r][k][t][g][n] = dict()

                for m in mus:
                    if not m in evaluations[r][k][t][g][n]:
                        evaluations[r][k][t][g][n][m] = dict()

                    for l in lambdas:
                        if not l in evaluations[r][k][t][g][n][m]:
                            evaluations[r][k][t][g][n][m][l] = dict()

                        for i in iters:
                            if not t in evaluations[r][k][t][g][n][m][l]:
                                evaluations[r][k][t][g][n][m][l][i] = dict()

                            runs.append({
                                'res_x' : res[0], 
                                'res_y' : res[1],
                                'tau' : t, 
                                'kappa' : k,
                                'gamma' : g,
                                'nu' : n,
                                'mu' : m,
                                'lambda' : l, 
                                'iters' : i,
                            })


def write_evaldict(run, result):
    r = "{}x{}".format(run['res_x'], run['res_y'])
    t = run['tau']
    k = run['kappa']
    g = run['gamma']
    n = run['nu']
    m = run['mu']
    l = run['lambda']
    i = run['iters']

    evaluations[r][k][t][g][n][m][l][i] = result

for run in tqdm(runs):
    try:
        result = run_sfs(run) 
        write_evaldict(run, result)
    except KeyboardInterrupt:
        print "Aborted!"
        break
    except Exception as e:
        print "Error during execution:", e
    

with open(os.path.join(output_dir, "evaluation.json"), 'w') as f_json:
    json.dump(evaluations, f_json)