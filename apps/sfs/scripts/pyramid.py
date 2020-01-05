import os
import subprocess
import json
import shutil
from tqdm import tqdm
import argparse

import numpy as np
import cv2

build_dir = '../../../build/'
cuimage_build_dir = '../../../third_party/cuda-image/build/'

TARGET_RES = (1280, 960)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='/data/synthetic/')
parser.add_argument('-o', '--output', default='/data/eval/sfs/pyramid/synthetic')
parser.add_argument('-t', '--tau', default=1.5)
parser.add_argument('-l', '--lambd', default=0.3)
parser.add_argument('-g', '--gamma', default=1.0)
parser.add_argument('-n', '--nu', default=0.5)
parser.add_argument('--levels', default=4)


args = parser.parse_args()

levels = int(args.levels)

data_dir = args.dataset
output_dir = args.output

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

run_data_dir = os.path.join(output_dir, 'data')
if not os.path.exists(run_data_dir):
    os.makedirs(run_data_dir)

resolutions = [(TARGET_RES[0] / 2**l, TARGET_RES[1] / 2**l) for l in range(levels)]
print resolutions



def run_sfs(run_args):
        
    # Run AppSfs with the right parameters
    args = [
        os.path.abspath(build_dir + '/bin/AppSfs'),
        '--dataset_path', run_args['data_dir'] + '/',
        '--dataset_frame_num', '0',
        '--dataset_resolution', run_args['res_x'], run_args['res_y'],
        '--admm_kappa', run_args['kappa'],
        '--admm_tau', run_args['tau'],
        '--admm_tolerance', run_args['tolerance'],
        '--optim_gamma', run_args['gamma'],
        '--optim_mu', run_args['mu'],
        '--optim_nu', run_args['nu'],
        '--optim_lambda', run_args['lambda'],
        '--optim_iter', run_args['iters'],
        '--output_results_folder', run_args['out_dir'] + '/',
        '--output_run_folder', run_args['folder']
    ]
    if run_args['gt_albedo']:
        args.append('--dataset_gt_albedo')
    
    if run_args['gt_light']:
        args.append('--dataset_gt_light')
    
    if run_args['gt_depth']:
        args.append('--dataset_gt_depth')
    
    if run_args['smooth']:
        args.extend(['--dataset_depth_sigma', str(run_args['smooth'])])

    if run_args['image']:
        args.append('--dataset_prefer_image')

    args = [str(a) for a in args]
    print args

    with open(os.path.join(run_args['data_dir'], run_args['folder'], 'output.txt'), 'w') as f:
        subprocess.Popen(
            args,
            cwd=os.path.abspath(build_dir), 
            stdout=f, 
            stderr=f
        ).wait()

    # Open the evaluation files
    output = dict()
    with open(os.path.join(run_args['out_dir'], run_args['folder'], 'results.txt')) as f_res:
        line = f_res.readline().strip().split(';')        
        output['meanErr'] = float(line[0])
        output['medianErr'] = float(line[1])
        output['rmsErr'] = float(line[2])

    with open(os.path.join(run_args['out_dir'], run_args['folder'], 'output.txt')) as f_out:
        lines = [l.strip().split('|') for l in f_out.readlines() if str(l).startswith("|")]
        steps = 0
        time_ms = 0

        for l in lines:
            if len(l) < 8 or 'Iter' in l[1]:
                continue
            
            steps += 1
            ts = [float(t[:-3]) for t in l[3:7]] # t_a, t_l, t_t, t_d, t_lg
            for t in ts:
                time_ms += t

        output['time'] = time_ms / 1000
        output['steps'] = steps

    return output

def run_viewer(dir, steps=2):
    args = [
        os.path.abspath(cuimage_build_dir + '/bin/Viewer'),
        os.path.abspath(dir),
        'h', 'depth.png',
    ]
    # Extra halvings for each additional level
    args.extend(['h', 'halved.exr'] * (steps - 3)) # start at 640x480 and already downsampled once with h depth.png
    args = [str(a) for a in args]
    print args
    subprocess.Popen(
            args
        ).wait()

# Generate the coarsest image
run_viewer(data_dir, levels)

# Construct dataset for finest levels
try:
    shutil.copy2(os.path.join(data_dir, 'color_1.png'), os.path.join(run_data_dir, 'color.png'))
except:
    shutil.copy2(os.path.join(data_dir, 'color.png'), os.path.join(run_data_dir, 'color.png'))

shutil.move(os.path.join(data_dir, 'halved.exr'), os.path.join(run_data_dir, 'depth_init.exr'))
shutil.copy2(os.path.join(data_dir, 'depth.png'), os.path.join(run_data_dir, 'depth.png'))

try:
    shutil.copy2(os.path.join(data_dir, 'mask.png'), os.path.join(run_data_dir, 'mask.png'))
except:
    d = cv2.imread(os.path.join(run_data_dir, 'depth.png'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    dm = np.zeros_like(d, dtype=np.uint8)
    dm[d > 0] = 255
    dm = cv2.cvtColor(dm, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(run_data_dir, 'mask.png'), dm)

i = cv2.imread(os.path.join(run_data_dir, 'depth_init.exr'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
i = i.astype(np.uint16).astype(np.float32) # create artifacts
i[i == 0] = np.nan
h, w = i.shape

m = cv2.imread(os.path.join(run_data_dir,'mask.png'))
m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
m = cv2.resize(m, (w, h), cv2.INTER_NEAREST)
m[m < 127] = 0
m[m >= 127] = 255

i[m == 0] = 0
cv2.imwrite(os.path.join(run_data_dir, 'depth_init.exr'), i)

res = resolutions[-1]
next_res = resolutions[-2]
wdir = os.path.join(output_dir, "{}x{}".format(*res))
if not os.path.exists(wdir):
    os.mkdir(wdir)

odir = os.path.join(wdir, 'output')
if not os.path.exists(odir):
    os.mkdir(odir)

# Run initial Sfs
initial_args = {
    'data_dir' : os.path.abspath(wdir),
    'out_dir' : os.path.abspath(wdir),
    'folder' : 'output',
    'res_x' : next_res[0],
    'res_y' : next_res[1],
    'gamma' : 1.0,
    'kappa' : 1e-4,
    'tau' : args.tau,
    'mu' : 0.001,
    'nu' : float(args.nu) / levels,
    'lambda' : float(args.lambd) / levels,
    'iters' : 300,
    'gt_albedo' : False,
    'gt_depth' : True,
    'gt_light' : False,
    'smooth' : 5.0,
    'image' : True,
    'tolerance' : 1e-9
}

# Copy necessary data

shutil.copy2(os.path.join(run_data_dir, 'color.png'), os.path.join(wdir, 'color.png'))
shutil.copy2(os.path.join(run_data_dir, 'depth.png'), os.path.join(wdir, 'depth.png'))
shutil.copy2(os.path.join(run_data_dir, 'mask.png'), os.path.join(wdir, 'mask.png'))
shutil.copy2(os.path.join(run_data_dir, 'depth_init.exr'), os.path.join(wdir, 'depth.exr'))

outputs = list()
outputs.append(run_sfs(initial_args))
print outputs[-1]

res_last = res
out_last = odir

for i in reversed(range(1, levels - 1)):
    res = resolutions[i]
    next_res = resolutions[i - 1]
    wdir = os.path.join(output_dir, "{}x{}".format(*res))
    odir = os.path.join(wdir, 'output')

    if not os.path.exists(wdir):
        os.mkdir(wdir)

    if not os.path.exists(odir):
        os.mkdir(odir)

    print "Level {}: Working at {} in {}".format(i, res, wdir)

    # Copy from last level
    shutil.copy2(os.path.join(out_last, 'albedo.png'), os.path.join(wdir, 'albedo.png'))
    shutil.copy2(os.path.join(out_last, 'depth_refined.exr'), os.path.join(wdir, 'depth.exr'))
    shutil.copy2(os.path.join(out_last, 'light.txt'), os.path.join(wdir, 'light.txt'))
    shutil.copy2(os.path.join(run_data_dir, 'color.png'), os.path.join(wdir, 'color.png'))
    shutil.copy2(os.path.join(run_data_dir, 'mask.png'), os.path.join(wdir, 'mask.png'))
    shutil.copy2(os.path.join(run_data_dir, 'depth.png'), os.path.join(wdir, 'depth.png'))

    # Run Sfs for this level
    this_args = {
        'data_dir' : os.path.abspath(wdir),
        'out_dir' : os.path.abspath(wdir),
        'folder' : 'output',
        'res_x' : next_res[0],
        'res_y' : next_res[1],
        'gamma' : 1.0,
        'kappa' : 1e-4,
        'tau' : float(args.tau),
        'mu' : 0.001,
        'nu' : float(args.nu) / (levels - i),
        'lambda' : float(args.lambd) / (levels - i),
        'iters' : 300 / (levels - i),
        'gt_albedo' : True,
        'gt_depth' : True,
        'gt_light' : True,
        'smooth' : 0.1,
        'image' : True,
        'tolerance' : 1e-9
    }
    outputs.append(run_sfs(this_args))
    print outputs[-1]


    res_last = res
    out_last = odir