import os
import sqlite3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d

output_dir = "/data/eval/sfs/synthetic/graphs"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

conn = sqlite3.connect('eval.db')
c = conn.cursor()

sql_query_tau_kappa = '''
    SELECT 
        tau, kappa, time, steps, rms_error, mean_error
    FROM 
        runs 
    WHERE
        resolution LIKE ?;
'''

c.execute(sql_query_tau_kappa, ['640x480', ])
res_640 = c.fetchall()
c.execute(sql_query_tau_kappa, ['1280x960', ])
res_1280 = c.fetchall()

tau_labels = np.unique([r[0] for r in res_640 + res_1280])
kappa_labels = np.unique([r[1] for r in res_640 + res_1280])
print tau_labels, kappa_labels

tau_640, kappa_640, time_640, rms_640 = [], [], [], []
for r in res_640:
    tau_640.append(r[0])
    kappa_640.append(r[1])
    time_640.append(r[2])
    rms_640.append(r[4])

tau_1280, kappa_1280, time_1280, rms_1280 = [], [], [], []
for r in res_1280:
    tau_1280.append(r[0])
    kappa_1280.append(r[1])
    time_1280.append(r[2])
    rms_1280.append(r[4])


def plot_1over1(x, y, title, resolution='1280x960'):
    assert('name' in x and 'name' in y)

    x_name = x['name']
    y_name = y['name']
    x_label = x.get('label', '')
    y_label = y.get('label', '')
    x_lim = x.get('limits', None)
    y_lim = y.get('limits', None)
    x_scale = x.get('scale', 'linear')
    y_scale = y.get('scale', 'linear')
    

    if isinstance(resolution, str):
        sql = "SELECT {}, {} FROM runs WHERE resolution LIKE ?;".format(x_name, y_name)
        c.execute(sql, [resolution, ])
        res = c.fetchall()
        x_data = [r[0] for r in res]
        y_data = [r[1] for r in res]
    elif isinstance(resolution, list):
        res = []
        for r in resolution:
            sql = "SELECT {}, {} FROM runs WHERE resolution LIKE ?;".format(x_name, y_name)
            c.execute(sql, [r, ])
            res.append(c.fetchall())

        x_data = [[r[0] for r in re] for re in res]
        y_data = [[r[1] for r in re] for re in res]


    # Y over X
    sns.set_style('ticks')
    fig, axes = plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    fig.suptitle(title, fontsize=16)

    # Add data
    if isinstance(resolution, str): 
        axes.scatter(x_data, y_data, c='b', marker='o', edgecolor='b', s=50)
    elif isinstance(resolution, list):
        colors = ['b', 'r', 'g', 'v']
        for i in range(len(resolution)):
            axes.scatter(x_data[i], y_data[i], c=colors[i], marker='o', edgecolor=colors[i], s=50, label=resolution[i])

        axes.legend()

    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)

    if x_lim:
        axes.set_xlim(x_lim)
    if y_lim:
        axes.set_ylim(y_lim)

    axes.set_xscale(x_scale)
    axes.set_yscale(y_scale)
    
    #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.98, wspace=0.2, hspace=0.1)

    if isinstance(resolution, str): 
        plt.savefig(os.path.join(output_dir, '{}_{}_over_{}.png'.format(resolution, y_name, x_name)))
    elif isinstance(resolution, list): 
        plt.savefig(os.path.join(output_dir, '{}_{}_over_{}.png'.format('multi', y_name, x_name)))

d_tau = {'name' : 'tau', 'label' : r'$\tau$'}#'ADMM Step Size Multiplier'}
d_kappa = {'name' : 'kappa', 'label' : r'$\kappa$', 'scale' : 'log', 'limits' : [1e-7, 1e-1]}#'ADMM Initial Step Size'}
d_gamma ={'name' : 'gamma', 'label' : r'$\gamma$'}#'SFS Fitting Weight'}
d_nu= {'name' : 'nu', 'label' : r'$\nu$'}#'SFS Surface Weight'}
d_mu = {'name' : 'lambda', 'label' : r'$\mu$'}#SFS Depth Weight'}
d_lambda = {'name' : 'lambda', 'label' : r'$\lambda$'}#Albedo Smoothness Weight'}
d_aiter = {'name' : 'lambda', 'label' : 'Albedo Iterations'}
d_time = {'name' : 'time', 'label' : 'Duration (s)'}
d_rms = {'name' : 'rms_error', 'label' : 'RMS Error (mm)'}
d_mean = {'name' : 'mean_error', 'label' : 'Mean Error(mm)'}
d_steps = {'name' : 'steps', 'label' : 'Steps until Convergence'}


plot_1over1(d_tau, d_rms, 'Influence of ADMM Step Size Multiplier on Quality')
plot_1over1(d_kappa, d_rms, 'Influence of Initial ADMM Step Size on Quality')
plot_1over1(d_gamma, d_rms, 'Influence SFS Fitting Weight on Quality')
plot_1over1(d_nu, d_rms, 'Influence of SFS Surface Weight on Quality', resolution=['1280x960', '640x480'])
plot_1over1(d_mu, d_rms, 'Influence of SFS Depth Weight on Quality')
plot_1over1(d_lambda, d_rms, 'Influence of Albedo Smoothness Weight on Quality')
plot_1over1(d_aiter, d_rms, 'Influence of Albedo Iterations on Quality')
plot_1over1(d_steps, d_rms, 'Influence of Steps until Convergence on Quality', resolution=['1280x960', '640x480'])
plot_1over1(d_steps, d_mean, 'Influence of Steps until Convergence on Quality', resolution=['1280x960', '640x480'])



# All measurements of time/duration over kappa/tau
fig, axes = plt.subplots(2, 2, sharex='col')
fig.set_figheight(10)
fig.set_figwidth(10)
fig.suptitle('Influence of ADMM step parameters on Quality and Duration (640x480)', fontsize=16)
axes[0, 0].scatter(tau_640, rms_640, c='r', marker='.', s=50)
axes[0, 0].set_ylabel('RMS Error (mm)')
axes[0, 0].set_xscale('Log')
axes[0, 0].set_xlim([1e-7, 1e-1])
axes[0, 1].scatter(kappa_640, rms_640, c='r', marker='.', s=50)
axes[0, 1].set_xscale('Log')
axes[0, 1].set_xlim([1, 12])
axes[1, 0].scatter(tau_640, time_640, c='b', marker='.', s=50)
axes[1, 0].set_ylabel('Duration (s)')
axes[1, 0].set_xlabel('Tau')
axes[1, 0].set_xlim([1e-7, 1e-1])
axes[1, 1].scatter(kappa_640, time_640, c='b', marker='.', s=50)
axes[1, 1].set_xlabel('Kappa')
axes[1, 1].set_xlim([1, 10])
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.98, wspace=0.2, hspace=0.1)
plt.savefig(os.path.join(output_dir, 'tau_kappa_640.png'))

fig, axes = plt.subplots(2, 2, sharex='col')
fig.set_figheight(10)
fig.set_figwidth(10)
fig.suptitle('Influence of ADMM step parameters on Quality and Duration (1280x960)', fontsize=16)
axes[0, 0].scatter(tau_1280, rms_1280, c='r', marker='.', s=50)
axes[0, 0].set_ylabel('RMS Error (mm)')
axes[0, 0].set_xscale('Log')
axes[0, 0].set_xlim([1e-7, 1e-1])
axes[0, 1].scatter(kappa_1280, rms_1280, c='r', marker='.', s=50)
axes[0, 1].set_xscale('Log')
axes[0, 1].set_xlim([1, 12])
axes[1, 0].scatter(tau_1280, time_1280, c='b', marker='.', s=50)
axes[1, 0].set_ylabel('Duration (s)')
axes[1, 0].set_xlabel('Tau')
axes[1, 0].set_xlim([1e-7, 1e-1])
axes[1, 1].scatter(kappa_1280, time_1280, c='b', marker='.', s=50)
axes[1, 1].set_xlabel('Kappa')
axes[1, 1].set_xlim([1, 10])
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.98, wspace=0.2, hspace=0.1)
plt.savefig(os.path.join(output_dir, 'tau_kappa_1280.png'))



# Quality over time
fig, axes = plt.subplots(1, 1, sharex='col')
fig.set_figheight(10)
fig.set_figwidth(10)
fig.suptitle('Influence of Duration on Quality (1280x480)', fontsize=16)
axes.scatter(time_1280, rms_1280, c='b', marker='o', s=50)
axes.set_ylabel('RMS Error (mm)')
axes.set_xlabel('Duration (s)')
#plt.subplots_adjust(left=0.1, bottom=0.1, right=0.98, wspace=0.2, hspace=0.1)
plt.savefig(os.path.join(output_dir, 'quality_time_1280.png'))
