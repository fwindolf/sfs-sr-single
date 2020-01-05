import os
import subprocess
import numpy as np
import argparse

build_dir = '../../../build/'

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='/data/synthetic/')
parser.add_argument('-o', '--output', default='/data/eval/sfs/admm_sweep')

args = parser.parse_args()

data_dir = args.dataset
output_dir = args.output

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

folder = '{res_x}x{res_y}_{tau}_{kappa}__{gamma}_{nu}_{mu}__{lambda}_{iters}'

def run_sfs(output_dir, run):
    run_folder = run['folder']
    
    out_folder = os.path.join(output_dir, run_folder)
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    if len(os.listdir(out_folder)) <= 1:
        # Run AppSfs with the right parameters
        args = [
            os.path.abspath(build_dir + '/bin/AppSfs'),
            '--dataset_path', os.path.abspath(data_dir) + '/',
            '--dataset_resolution', run['res_x'], run['res_y'],
            '--dataset_frame_num', run['f'],
            '--admm_kappa', run['kappa'],
            '--admm_tau', run['tau'],
            '--optim_gamma', run['gamma'],
            '--optim_mu', run['mu'],
            '--optim_nu', run['nu'],
            '--optim_lambda', run['lambda'],
            '--optim_iter', run['iters'],
            '--output_results_folder', os.path.abspath(output_dir) + '/',
            '--output_run_folder', run_folder,
            '--dataset_depth_sigma', 5.0,
            #'--dataset_prefer_image'
        ]
        args = [str(a) for a in args]
        print args

        with open(os.path.join(out_folder, 'output.txt'), 'w') as f:
            subprocess.Popen(
                args,
                cwd=os.path.abspath(build_dir), 
                stdout=f, 
                stderr=f
            ).wait()

    # Open the evaluation files
    output = dict()
    with open(os.path.join(out_folder, 'results.txt')) as f_res:
        line = f_res.readline().strip().split(';')        
        output['meanErr'] = float(line[0])
        output['medianErr'] = float(line[1])
        output['rmsErr'] = float(line[2])

    with open(os.path.join(out_folder, 'output.txt')) as f_out:
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


taus = np.arange(1.5, 8.5, step=0.5)
kappas = [7.74263682e-8, 2.78255940e-7] + list(np.logspace(-6, -1, 10)) + [3.59381366e-01, 1.29154967e-00]
#kappas = np.logspace(-6, -1, 21)

print kappas
print taus

print len(kappas) * len(taus)


run = {
    'res_x' : 1280,
    'res_y' : 960,
    'f' : 0,
    'gamma' : 0.8,
    'mu' : 0.001,
    'nu' : 0.5,
    'lambda' : 0.5, 
    'iters' : 300,
    'data_dir' : os.path.abspath(data_dir),
    'out_dir' : os.path.abspath(output_dir),
}

results = []
maes_t, maes_k = {}, {}
rms_t, rms_k = {}, {}
times_t, times_k = {}, {}

labels = {0 : 'stars', 1: 'stripes', 2 : 'circles'}
colors = {0 : 'blue', 1: 'green', 2 : 'red'}

for i in range(3):        
    for k in kappas:
        for t in sorted(taus):
            run['kappa'] = k
            run['tau'] = t
            run['f'] = i
            
            folder = '{}_{:.6f}'.format(t, k)
            run['folder'] = folder

            out_dir = "{}_{}".format(output_dir, i)

            if not os.path.exists(os.path.join(out_dir, folder)):
                os.makedirs(os.path.join(out_dir, folder))

            output = run_sfs(out_dir, run)

            results.append(output)

            if not i in maes_t:
                maes_t[i] = {}
                rms_t[i] = {}
                times_t[i] = {}

            if not i in maes_k:
                maes_k[i] = {}
                rms_k[i] = {}
                times_k[i] = {}                
            
            if t in maes_t[i]:
                maes_t[i][t].append(output['meanErr'])
                rms_t[i][t].append(output['rmsErr'])
                times_t[i][t].append(output['time'])
            else:
                maes_t[i][t] = [output['meanErr']]
                rms_t[i][t] = [output['rmsErr']]
                times_t[i][t] = [output['time']]

            if k in maes_k[i]:
                maes_k[i][k].append(output['meanErr'])
                rms_k[i][k].append(output['rmsErr'])
                times_k[i][k].append(output['time'])
            else:
                maes_k[i][k] = [output['meanErr']]
                rms_k[i][k] = [output['rmsErr']]
                times_k[i][k] = [output['time']]

import seaborn as sns
import matplotlib.pyplot as plt


def intervals(data_dict, quantil=0.95):
    d_ = np.array([data_dict[k] for k in sorted(data_dict.keys())])
    d_mean = np.mean(d_, axis=0)
    d_std = np.std(d_, axis=0)
    d_low = d_mean - quantil * d_std
    d_hi = d_mean + quantil * d_std

    return d_mean, d_low, d_hi


# Create figure
sns.set_style('ticks')
fig, axes = plt.subplots(1, 3, sharex=True)
fig.suptitle(r'Influence of $\tau$ parameter')
fig.set_figheight(5)
fig.set_figwidth(15)

# MAAE
axes[0].set_ylabel(r'Mean Angular Error ($^\circ$)')
axes[0].set_xlabel(r'$\tau$')
axes[0].set_xscale('log')
axes[0].set_xlim([8e-8, 1.2e-0])

for i in range(3):
    maes_t_mean, maes_t_low, maes_t_hi = intervals(maes_t[i])
    print len(kappas), len(maes_t_mean)
    axes[0].plot(kappas, maes_t_mean, color=colors[i], label=labels[i])
    axes[0].fill_between(kappas, maes_t_low, maes_t_hi, color=colors[i], alpha=0.2)


# RMS
axes[1].set_ylabel('RMS Error (mm)')
axes[1].set_xlabel(r'$\tau$')
axes[1].set_xscale('log')
axes[1].set_xlim([8e-8, 1.2e-0])

for i in range(3):
    rms_t_mean, rms_t_low, rms_t_hi = intervals(rms_t[i])
    
    axes[1].plot(kappas, rms_t_mean, color=colors[i], label=labels[i])
    axes[1].fill_between(kappas, rms_t_low, rms_t_hi, color=colors[i], alpha=0.2)


# Times
axes[2].set_ylabel('Runtime (s)')
axes[2].set_xlabel(r'$\tau$')
axes[2].set_xscale('log')
axes[2].set_xlim([8e-8, 1.2e-0])

for i in range(3):
    times_t_mean, times_t_low, times_t_hi = intervals(times_t[i])

    axes[2].plot(kappas, times_t_mean, color=colors[i], label=labels[i])
    axes[2].fill_between(kappas, times_t_low, times_t_hi, color=colors[i], alpha=0.2)

plt.legend()
plt.subplots_adjust(bottom=0.15, right=0.98, left=0.05)
f_output = os.path.join(output_dir, 'sweep_kappa.png')
print "Writing graph to", f_output
plt.savefig(f_output)

################

# Create figure
fig, axes = plt.subplots(1, 3, sharex=True)
fig.suptitle(r'Influence of $\kappa$ parameter')
fig.set_figheight(5)
fig.set_figwidth(15)

# MAAE
axes[0].set_ylabel(r'Mean Angular Error ($^\circ$)')
axes[0].set_xlabel(r'$\kappa$')
axes[0].set_xlim([1.5, 8.0])

for i in range(3):
    maes_k_mean, maes_k_low, maes_k_hi = intervals(maes_k[i])

    print len(taus), len(maes_k_mean)

    axes[0].plot(taus, maes_k_mean, color=colors[i], label=labels[i])
    axes[0].fill_between(taus, maes_k_low, maes_k_hi, color=colors[i], alpha=0.2)


# RMS
axes[1].set_ylabel('RMS Error (mm)')
axes[1].set_xlabel(r'$\kappa$')
axes[1].set_xlim([1.5, 8.0])

for i in range(3):
    rms_k_mean, rms_k_low, rms_k_hi = intervals(rms_k[i])

    axes[1].plot(taus, rms_k_mean, color=colors[i], label=labels[i])
    axes[1].fill_between(taus, rms_k_low, rms_k_hi, color=colors[i], alpha=0.2)


# Times
axes[2].set_ylabel('Runtime (s)')
axes[2].set_xlabel(r'$\kappa$')
axes[2].set_xlim([1.5, 8.0])

for i in range(3):
    times_k_mean, times_k_low, times_k_hi = intervals(times_k[i])

    axes[2].plot(taus, times_k_mean, color=colors[i], label=labels[i])
    axes[2].fill_between(taus, times_k_low, times_k_hi, color=colors[i], alpha=0.2)

plt.legend()
plt.subplots_adjust(bottom=0.15, right=0.98, left=0.05)
f_output = os.path.join(output_dir, 'sweep_tau.png')
print "Writing graph to", f_output
plt.savefig(f_output)


