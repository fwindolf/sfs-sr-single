import os
import numpy as np
import shutil
import subprocess

orig_dir = '/data/synthetic/'
data_dir = '/data/synthetic_smoothed/'
build_dir = '../../../build/'


sigmas = np.logspace(-2, 2, 20)

def run_sfs_best(dir, folder, sigma):
    args = [
        os.path.abspath(build_dir + '/bin/AppSfs'),
        '--dataset_path', '/data/synthetic/',
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
        '--dataset_gt_depth',
        '--dataset_depth_sigma', str(sigma)
    ]

    with open(os.path.join(dir, folder, 'output.txt'), 'w') as f:
        subprocess.Popen(
            args,
            cwd=os.path.abspath(build_dir), 
            stdout=f, 
            stderr=f
        ).wait()

def run_sfs_fast(dir, folder, sigma):
    args = [
        os.path.abspath(build_dir + '/bin/AppSfs'),
        '--dataset_path', '/data/synthetic/',
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
        '--dataset_gt_depth',
        '--dataset_depth_sigma', str(sigma)
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
for sigma in []:#sigmas:
    out_dir = os.path.join(data_dir, str(sigma))

    print "Running {} for {}".format(sigma, out_dir)

    if os.path.exists(os.path.join(out_dir, 'best')):
        shutil.rmtree(os.path.join(out_dir, 'best'))

    os.makedirs(os.path.join(out_dir, 'best'))

    run_sfs_best(out_dir, 'best', sigma)
    res_best = read_results(out_dir, 'best')
    
    if os.path.exists(os.path.join(out_dir, 'fast')):
        shutil.rmtree(os.path.join(out_dir, 'fast'))

    os.makedirs(os.path.join(out_dir, 'fast'))

    run_sfs_fast(out_dir, 'fast', sigma)    
    res_fast =  read_results(out_dir, 'fast')

    results.append({
        'best' : res_best, 
        'fast' : res_fast
    })


#with open(os.path.join(data_dir, 'results.csv'), 'w') as f_out:
#    f_out.write('Sigma;Type;Steps;Time;MAE;RMS\n') 
#    for res, sigma in zip(results, sigmas):
#        f_out.write('{};best;{};{};{};{}\n'.format(sigma, res['best']['steps'], res['best']['time'], res['best']['meanErr'], res['best']['rmsErr'])) 
#        f_out.write('{};fast;{};{};{};{}\n'.format(sigma, res['fast']['steps'], res['fast']['time'], res['fast']['meanErr'], res['fast']['rmsErr'])) 


import matplotlib.pyplot as plt
import seaborn as sns

sigma_fast = list()
mae_fast = list()
steps_fast = list()
times_fast = list()

sigma_best = list()
mae_best = list()
steps_best = list()
times_best = list()

for sigma in reversed(sigmas):
    out_dir = os.path.join(data_dir, str(sigma))

    print "Evaluating {} for {}".format(sigma, out_dir)
    res_best = read_results(out_dir, 'best')
    res_fast =  read_results(out_dir, 'fast')

    sigma_fast.append(sigma)
    mae_fast.append(res_fast['meanErr'])
    steps_fast.append(res_fast['steps'])
    times_fast.append(res_fast['time'])

    sigma_best.append(sigma)
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
axes[0].set_xlabel(r'Smoothing ($\sigma$)')
axes[0].set_xscale('log')
axes[0].set_ylim([0, 5])
#axes[0].set_yscale('log')
axes[0].plot(sigma_fast, mae_fast, c='r', label='fast')
axes[0].plot(sigma_best, mae_best, c='b', label='best')


# Runtime
axes[1].set_ylabel('Runtime (s)')
axes[1].set_xlabel(r'Smoothing ($\sigma$)')
axes[1].set_xscale('log')
axes[1].set_ylim([0, 30])
#axes[1].set_yscale('log')
axes[1].plot(sigma_fast, steps_fast, c='r', label='fast')
axes[1].plot(sigma_best, steps_best, c='b', label='best')

plt.legend(loc='best')

plt.subplots_adjust(bottom=0.20)
f_output = os.path.join(data_dir, 'initialization.png')
print "Writing graph to", f_output
plt.savefig(f_output)