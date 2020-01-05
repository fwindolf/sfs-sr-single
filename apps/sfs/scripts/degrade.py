import os
import cv2
import numpy as np
import shutil
import subprocess

orig_dir = '/data/synthetic/'
data_dir = '/data/synthetic_degraded_wo_smoothing/'
build_dir = '../../../build/'

d_orig = cv2.imread(os.path.join(orig_dir, 'depth.png'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

d_orig = np.asarray(d_orig, dtype=np.float32)
d_orig[np.isnan(d_orig)] = 0

d_vals = d_orig[d_orig > 0]
d_range = (np.max(d_vals) - np.min(d_vals))

print d_range

degradations = np.logspace(-5, 1, 20)

# degrade the depth image with N(d(x)| degradation * d(x) / range)

depths_deg = list()
for deg in degradations:
    d = np.random.normal(d_orig, deg * (d_orig / d_range))

    out_dir = os.path.join(data_dir, str(deg))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print deg, np.linalg.norm(d_orig - d)

    cv2.imwrite(os.path.join(out_dir, 'depth.png'.format(deg)), d.astype(np.uint16))

    shutil.copy2(os.path.join(orig_dir, 'color_1.png'), os.path.join(out_dir, 'color.png'))
    shutil.copy2(os.path.join(orig_dir, 'mask.png'), os.path.join(out_dir, 'mask.png'))
    shutil.copy2(os.path.join(orig_dir, 'depth.exr'), os.path.join(out_dir, 'depth.exr'))

def run_sfs_best(dir, folder):
    args = [
        os.path.abspath(build_dir + '/bin/AppSfs'),
        '--dataset_path', os.path.abspath(dir) + '/',
        '--dataset_frame_num', '0',
        '--dataset_resolution', '1280', '960',
        '--admm_kappa', '1e-5',
        '--admm_tau', '1.5',
        '--optim_gamma', '0.2',
        '--optim_mu', '0.001',
        '--optim_nu', '0.5',
        '--optim_lambda', '0.5',
        '--optim_iter', '400',
        '--output_results_folder', os.path.abspath(dir) + '/',
        '--output_run_folder', folder,
        #'--dataset_smooth_depth'
    ]

    with open(os.path.join(dir, folder, 'output.txt'), 'w') as f:
        subprocess.Popen(
            args,
            cwd=os.path.abspath(build_dir), 
            stdout=f, 
            stderr=f
        ).wait()

def run_sfs_fast(dir, folder):
    args = [
        os.path.abspath(build_dir + '/bin/AppSfs'),
        '--dataset_path', os.path.abspath(dir) + '/',
        '--dataset_frame_num', '0',
        '--dataset_resolution', '1280', '960',
        '--admm_kappa', '1e-4',
        '--admm_tau', '5',
        '--optim_gamma', '0.3',
        '--optim_mu', '0.001',
        '--optim_nu', '1e-5',
        '--optim_lambda', '0.3',
        '--optim_iter', '200',
        '--output_results_folder', os.path.abspath(dir) + '/',
        '--output_run_folder', folder,
        #'--dataset_smooth_depth'
    ]


    with open(os.path.join(dir, folder, 'output.txt'), 'w') as f:
        subprocess.Popen(
            args,
            cwd=os.path.abspath(build_dir), 
            stdout=f, 
            stderr=f
        ).wait()

def read_results(dir, folder):
    # Open the evaluation files
    output = dict()
    with open(os.path.join(dir, folder, 'results.txt')) as f_res:
        line = f_res.readline().strip().split(';')        
        output['meanErr'] = float(line[0])
        output['medianErr'] = float(line[1])
        output['rmsErr'] = float(line[2])

    with open(os.path.join(dir, folder, 'output.txt')) as f_out:
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


results = list()
for deg in reversed(degradations):
    out_dir = os.path.join(data_dir, str(deg))

    print "Running {} for {}".format(deg, out_dir)

    if os.path.exists(os.path.join(out_dir, 'best')):
        shutil.rmtree(os.path.join(out_dir, 'best'))

    os.makedirs(os.path.join(out_dir, 'best'))

    run_sfs_best(out_dir, 'best')
    res_best = read_results(out_dir, 'best')
    
    if os.path.exists(os.path.join(out_dir, 'fast')):
        shutil.rmtree(os.path.join(out_dir, 'fast'))

    os.makedirs(os.path.join(out_dir, 'fast'))

    run_sfs_fast(out_dir, 'fast')    
    res_fast =  read_results(out_dir, 'fast')

    results.append({
        'best' : res_best, 
        'fast' : res_fast
    })


with open(os.path.join(data_dir, 'results.csv'), 'w') as f_out:
    f_out.write('Deg;Type;Steps;Time;MAE;RMS\n') 

    for res, deg in zip(results, reversed(degradations)):
        f_out.write('{};best;{};{};{}\n'.format(deg, res['best']['steps'], res['best']['time'], res['best']['meanErr'], res['best']['rmsErr'])) 
        f_out.write('{};fast;{};{};{}\n'.format(deg, res['fast']['steps'], res['fast']['time'], res['fast']['meanErr'], res['fast']['rmsErr'])) 



import matplotlib.pyplot as plt
import seaborn as sns

deg_fast = list()
mae_fast = list()
steps_fast = list()
times_fast = list()

deg_best = list()
mae_best = list()
steps_best = list()
times_best = list()

for deg in reversed(degradations):
    out_dir = os.path.join(data_dir, str(deg))

    print "Evaluating {} for {}".format(deg, out_dir)
    res_best = read_results(out_dir, 'best')
    res_fast =  read_results(out_dir, 'fast')

    deg_fast.append(deg)
    mae_fast.append(res_fast['meanErr'])
    steps_fast.append(res_fast['steps'])
    times_fast.append(res_fast['time'])

    deg_best.append(deg)
    mae_best.append(res_best['meanErr'])
    steps_best.append(res_best['steps'])
    times_best.append(res_best['time'])


sns.set_style('ticks')

fig, axes = plt.subplots(1, 2, sharex=True)
#fig.suptitle(r'Influence of intial depth on quality and runtime')
fig.set_figheight(4)
fig.set_figwidth(10)

# Quality
axes[0].set_ylabel('Mean Error (mm)')
axes[0].set_xlabel(r'Degradation ($\sigma$)')
axes[0].set_xscale('log')
axes[0].set_ylim([0, 25])
#axes[0].set_yscale('log')
axes[0].plot(deg_fast, mae_fast, c='r', label='fast')
axes[0].plot(deg_best, mae_best, c='b', label='best')


# Runtime
axes[1].set_ylabel('Runtime (s)')
axes[1].set_xlabel(r'Degradation ($\sigma$)')
axes[1].set_xscale('log')
axes[1].set_ylim([0, 30])
#axes[1].set_yscale('log')
axes[1].plot(deg_fast, steps_fast, c='r', label='fast')
axes[1].plot(deg_best, steps_best, c='b', label='best')

plt.legend(loc='best')

plt.subplots_adjust(bottom=0.15)
f_output = os.path.join(data_dir, 'result.png')
print "Writing graph to", f_output
plt.savefig(f_output)