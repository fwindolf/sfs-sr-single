import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3

output_dir = "/data/eval/sfs/synthetic/graphs"

params = {
    "640x480" : {
        "fast" : {
            "k" : 1e-5,
            "t" : 10
        }, 
        "best" : {
            "k" : 1e-3,
            "t" : 2
        },
        "mu" : 0.01, 
        "nu" : 0.0001,
    },
    "1280x960" : {
        "fast" : {
            "k" : 1e-4,
            "t" : 5
        }, 
        "best" : {
            "k" : 1e-2,
            "t" : 2
        },
        "mu" : 0.001, 
        "nu" : 0.0001,
    },
    "lambda" : 0.5,
    "iters" : 400,
    "gamma" : [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20]
}
sql_mus = """
    SELECT DISTINCT
        mu
    FROM 
        runs
    WHERE
        kappa = ? AND
        tau = ? AND
        nu = ? AND
        lambda = ? AND
        iterations = ? AND
        resolution LIKE ? AND
        mu <= 1e-2
    ORDER BY 
        mu ASC;
"""

sql_gammas = """ 
    SELECT 
        gamma, mean_error, rms_error, time, steps 
    FROM 
        runs 
    WHERE
        kappa = ? AND
        tau = ? AND
        mu = ? AND
        nu = ? AND
        lambda = ? AND
        iterations = ? AND
        resolution LIKE ?
    ORDER BY 
        gamma ASC;
    """


conn = sqlite3.connect('eval.db')
c = conn.cursor()

def get_params_gamma(res, run, mu):
    return [
        params[res][run]['k'],
        params[res][run]['t'],
        mu,
        params[res]['nu'],
        params['lambda'],
        params['iters'],
        res
    ]

def get_params_mu(res):
    return [
        params[res]['best']['k'],
        params[res]['best']['t'],
        params[res]['nu'],
        params['lambda'],
        params['iters'],
        res
    ]

# Get taus
c.execute(sql_mus, get_params_mu("1280x960"))
mus_1280 = [mu[0] for mu in c.fetchall()]

c.execute(sql_mus, get_params_mu("640x480"))
mus_640 = [mu[0] for mu in c.fetchall()]

colors_1280 = [
    'darkred', # 0
    'firebrick', # 1e-5
    'red', #1e-4
    'indianred', #1e-3
    'lightcoral', #1e-2
]

colors_640 = [
    'midnightblue', # 0
    'darkblue', # 1e-5
    'blue', #1e-4
    'royalblue', #1e-3
    'cornflowerblue' #1e-2
]

sns.set_style('ticks')
fig, axes = plt.subplots(2, 2, sharex=True, sharey='row')
fig.suptitle(r'Influence of $\gamma$ and $\mu$ parameter on quality and runtime', fontsize='large')
fig.set_figheight(10)
fig.set_figwidth(15)

# Quality Best 
axes[0, 0].text(.5,.9,'Best Quality Configuration', fontsize='medium', horizontalalignment='center', transform=axes[0, 0].transAxes)
axes[0, 0].set_ylabel('Mean Error (mm)')
axes[0, 0].set_xlabel(r'$\gamma$')
axes[0, 0].set_xscale('Log')
axes[0, 0].set_xlim([0.01, 100])


for i, mu in enumerate(mus_1280):
    c.execute(sql_gammas, get_params_gamma("1280x960", "best", mu))
    res_1280 = c.fetchall()
    axes[0, 0].plot(
        [r[0] for r in res_1280], 
        [r[1] for r in res_1280], 
        c=colors_1280[i], label=r'1280x960 ($\mu$ = {})'.format(mu))

for i, mu in enumerate(mus_640):
    c.execute(sql_gammas, get_params_gamma("640x480", "best", mu))
    res_640 = c.fetchall()
    axes[0, 0].plot(
        [r[0] for r in res_640], 
        [r[1] for r in res_640], 
        c=colors_640[i], label=r'640x480 ($\mu$ = {})'.format(mu))
    
#axes[0, 0].legend(loc='center right', fontsize='small')

# Quality Fastest
axes[0, 1].text(.5,.9,'Fastest Runtime Configuration', fontsize='medium', horizontalalignment='center', transform=axes[0, 1].transAxes)
axes[0, 1].set_ylabel('Mean Error (mm)')
axes[0, 1].set_xlabel(r'$\gamma$')
axes[0, 1].set_xscale('Log')
axes[0, 0].set_xlim([0.01, 100])

for i, mu in enumerate(mus_1280):
    c.execute(sql_gammas, get_params_gamma("1280x960", "fast", mu))
    res_1280 = c.fetchall()
    axes[0, 1].plot(
        [r[0] for r in res_1280], 
        [r[1] for r in res_1280], 
        c=colors_1280[i], label=r'1280x960 ($\mu$ = {})'.format(mu))

for i, mu in enumerate(mus_640):
    c.execute(sql_gammas, get_params_gamma("640x480", "fast", mu))
    res_640 = c.fetchall()
    axes[0, 1].plot(
        [r[0] for r in res_640], 
        [r[1] for r in res_640], 
        c=colors_640[i], label=r'640x480 ($\mu$ = {})'.format(mu))
    
#axes[0, 1].legend(loc='center right', fontsize='small')

# Time Best 
axes[1, 0].text(.5,.9,'Best Quality Configuration', fontsize='medium', horizontalalignment='center', transform=axes[1, 0].transAxes)
axes[1, 0].set_ylabel('Duration (s)')
axes[1, 0].set_xlabel(r'$\gamma$')
axes[1, 0].set_xscale('Log')
axes[1, 0].set_xlim([0.01, 100])


for i, mu in enumerate(mus_1280):
    c.execute(sql_gammas, get_params_gamma("1280x960", "best", mu))
    res_1280 = c.fetchall()
    axes[1, 0].plot(
        [r[0] for r in res_1280], 
        [r[3] for r in res_1280], 
        c=colors_1280[i], label=r'1280x960 ($\mu$ = {})'.format(mu))

for i, mu in enumerate(mus_640):
    c.execute(sql_gammas, get_params_gamma("640x480", "best", mu))
    res_640 = c.fetchall()
    axes[1, 0].plot(
        [r[0] for r in res_640], 
        [r[3] for r in res_640], 
        c=colors_640[i], label=r'640x480 ($\mu$ = {})'.format(mu))
    
#axes[1, 0].legend(loc='center right', fontsize='small')

# Time Fastest
axes[1, 1].text(.5,.9,'Fastest Runtime Configuration', fontsize='medium', horizontalalignment='center', transform=axes[1, 1].transAxes)
axes[1, 1].set_ylabel('Duration (s)')
axes[1, 1].set_xlabel(r'$\gamma$')
axes[1, 1].set_xscale('Log')
axes[1, 1].set_xlim([0.01, 100])

for i, mu in enumerate(mus_1280):
    c.execute(sql_gammas, get_params_gamma("1280x960", "fast", mu))
    res_1280 = c.fetchall()
    axes[1, 1].plot(
        [r[0] for r in res_1280], 
        [r[3] for r in res_1280], 
        c=colors_1280[i], label=r'1280x960 ($\mu$ = {})'.format(mu))

for i, mu in enumerate(mus_640):
    c.execute(sql_gammas, get_params_gamma("640x480", "fast", mu))
    res_640 = c.fetchall()
    axes[1, 1].plot(
        [r[0] for r in res_640], 
        [r[3] for r in res_640], 
        c=colors_640[i], label=r'640x480 ($\mu$ = {})'.format(mu))
    
#axes[1, 1].legend(loc='center right', fontsize='small')
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.94, wspace=0.1, hspace=0.12)
handles, labels = plt.gca().get_legend_handles_labels()

# Reorder legend to be column wise
order = np.array(zip(range(5), range(5, 10))).flatten()
handles, labels = [handles[i] for i in order], [labels[i] for i in order]
plt.legend(handles, labels, loc='lower right', fontsize='small', ncol=5, bbox_to_anchor=(0.875, 0), bbox_transform=fig.transFigure)

plt.savefig(os.path.join(output_dir, 'gamma_mu.png'))