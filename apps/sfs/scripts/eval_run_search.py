import os
import subprocess
import json
from tqdm import tqdm
import argparse

build_dir = '../../../build/'
data_dir = '/data/real/'
output_dir = '/data/eval/sfs/real/'

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='/data/synthetic/')
parser.add_argument('-o', '--output', default='/data/eval/sfs/synthetic/')

args = parser.parse_args()

data_dir = args.dataset
output_dir = args.output

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

folder = '{res_x}x{res_y}_{tau}_{kappa}__{gamma}_{nu}_{mu}__{lambda}_{iters}'

def make_sfs():
    args = [
        'make', '-j', '8', 'AppSfs'
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
            os.path.abspath(build_dir + '/bin/AppSfs'),
            '--dataset_path', os.path.abspath(data_dir) + '/',
            '--dataset_resolution', run['res_x'], run['res_y'],
            '--dataset_frame_num', 1,
            '--admm_kappa', run['kappa'],
            '--admm_tau', run['tau'],
            '--optim_gamma', run['gamma'],
            '--optim_mu', run['mu'],
            '--optim_nu', run['nu'],
            '--optim_lambda', run['lambda'],
            '--optim_iter', run['iters'],
            '--output_results_folder', os.path.abspath(output_dir) + '/',
            '--output_run_folder', run_folder
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
    with open(os.path.join(output_dir, run_folder, 'results.txt')) as f_res:
        line = f_res.readline().strip().split(';')        
        output['meanErr'] = float(line[0])
        output['medianErr'] = float(line[1])
        output['rmsErr'] = float(line[2])
    
    return output

params = {
    '640x480' : {
        'fast' : {
            'k' : 1e-4,
            't' : 5
        }, 
        'best' : {
            'k' : 1e-4,
            't' : 1.5
        },
        'mu' : [0.00001, 0.001, 0.01, 0.1, 1.0],
        'nu' : [0.00001, 0.001, 0.01, 0.1, 1.0],
        'lambda' : 0.3,
        'iters' : 200
    },
    '1280x960' : {
        'fast' : {
            'k' : 1e-4,
            't' : 5
        }, 
        'best' : {
            'k' : 1e-5,
            't' : 1.5
        },
        'mu' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
        'nu' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0,],
        'lambda' : 0.5,
        'iters' : 300,
    },    
    'gamma' : [0.1, 0.2, 0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]    
}

# Make application
make_sfs()

# Run evaluations
runs = list()
evaluations = dict()

resolutions = ['640x480', '1280x960']
types = ['best', 'fast']
iterate = ['mu', 'nu', 'gamma']


obj = params.keys()

def combinations(lol):
    num = len(lol)
    assert(num >= 2)

    l1 = lol[0]
    for i in range(1, num):
        l2 = lol[i]

        combs = []
        for l2_el in l2:
            assert not isinstance(l2_el, list)
            for l1_el in l1:
                if isinstance(l1_el, list):
                    combs.append(l1_el + [l2_el, ])
                else:
                    combs.append([l1_el, l2_el])
        
        l1 = combs

    return l1


for r in resolutions:
    p = params[r]

    if 'lambda' not in iterate:
        l = p['lambda']

    if 'iters' not in iterate:
        i = p['iters']

    if 'gamma' not in iterate:
        g = params['gamma']
    
    for t in types:
        k = p[t]['k']
        t = p[t]['t']

        if 'mu' not in iterate:
            m = p['mu']
        
        if 'nu' not in iterate:
            n = p['nu']

        iterates = [params[k_iter] if k_iter in params else p[k_iter] for k_iter in iterate]
        
        if len(iterates) > 1:
            iterates = combinations(iterates)
        else:
            iterates = iterates[0]

        # Get the parameters for this iterable
        for tpl in iterates:
            if isinstance(tpl, tuple) or isinstance(tpl, list):
                get = lambda d, k: d[iterate.index(k)]
            else:
                get = lambda d, k: d

            if 'mu' in iterate:
                m = get(tpl, 'mu')

            if 'nu' in iterate:
                n = get(tpl, 'nu')

            if 'lambda' in iterate:
                l = get(tpl, 'lambda')

            if 'iters' in iterate:
                i = get(tpl, 'iters')

            if 'gamma' in iterate:
                g = get(tpl, 'gamma')

            rx, ry = r.split('x')
            runs.append({
                'res_x' : rx, 
                'res_y' : ry,
                'tau' : t, 
                'kappa' : k,
                'gamma' : g,
                'nu' : n,
                'mu' : m,
                'lambda' : l, 
                'iters' : i,
            })

for run in tqdm(runs):
    try:
        result = run_sfs(run)         
    except KeyboardInterrupt:
        print 'Aborted!'
        break
    except Exception as e:
        print 'Error during execution:', e
