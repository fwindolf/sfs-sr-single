import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interpolate
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
    },
    "lambda" : 0.5,
    "iters" : 400,
    "gamma" : 1.0,
    "nus" : [0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]
}


sql_gammas = """ 
    SELECT 
        nu, mean_error, rms_error, time, steps 
    FROM 
        runs 
    WHERE
        kappa = ? AND
        tau = ? AND
        gamma = ? AND
        mu = ? AND        
        lambda = ? AND
        iterations = ? AND
        resolution LIKE ?
    ORDER BY 
        nu ASC;
    """


conn = sqlite3.connect('eval.db')
c = conn.cursor()

def get_params(res, run):
    return [
        params[res][run]['k'],
        params[res][run]['t'],
        params['gamma'],
        params[res]['mu'],
        params['lambda'],
        params['iters'],
        res
    ]

# Data
## 1280x960
c.execute(sql_gammas, get_params('1280x960', 'best'))
res_1280_best =  c.fetchall()

c.execute(sql_gammas, get_params('1280x960', 'fast'))
res_1280_fast =  c.fetchall()

## 640x480
c.execute(sql_gammas, get_params('640x480', 'best'))
res_640_best =  c.fetchall()

c.execute(sql_gammas, get_params('640x480', 'fast'))
res_640_fast =  c.fetchall()

def plot_interpolated(axis, x, y, color, label):
    x_new = np.arange(np.amin(x), np.amax(x), 100)
    tck = interpolate.splrep(x, y, s=0)
    y_new = interpolate.splev(x_new, tck, der=0)
    axis.plot(x, y, x, y, "x", x_new, y_new, c=color, label=label)


# Create figure
sns.set_style('ticks')
fig, axes = plt.subplots(1, 2, sharex=True)
fig.suptitle(r'Influence of $\nu$ parameter on quality and runtime')
fig.set_figheight(4)
fig.set_figwidth(15)

# Quality
axes[0].set_ylabel('Mean Error (mm)')
axes[0].set_xlabel(r'$\nu$')
axes[0].set_xscale('log')
axes[0].set_xlim([1e-4, 1])
axes[0].plot(
    [r[0] for r in res_1280_best], 
    [r[1] for r in res_1280_best],
    c='red', label='1280x960 best')
axes[0].plot(
    [r[0] for r in res_1280_fast], 
    [r[1] for r in res_1280_fast],
    c='darkred', label='1280x960 fastest')
axes[0].plot(
    [r[0] for r in res_640_best], 
    [r[1] for r in res_640_best],
    c='cornflowerblue', label='640x480 best')
axes[0].plot(
    [r[0] for r in res_640_fast], 
    [r[1] for r in res_640_fast],
    c='midnightblue', label='640x480 fastest')

axes[0].legend(loc='upper left', fontsize='small')

# Runtime
axes[1].set_ylabel('Runtime (s)')
axes[1].set_xlabel(r'$\nu$')
axes[1].plot(
    [r[0] for r in res_1280_best], 
    [r[3] for r in res_1280_best],
    c='red', label='1280x960 best')
axes[1].plot(
    [r[0] for r in res_1280_fast], 
    [r[3] for r in res_1280_fast],
    c='darkred', label='1280x960 fastest')
axes[1].plot(
    [r[0] for r in res_640_best], 
    [r[3] for r in res_640_best],
    c='cornflowerblue', label='640x480 best')
axes[1].plot(
    [r[0] for r in res_640_fast], 
    [r[3] for r in res_640_fast],
    c='midnightblue', label='640x480 fastest')

plt.subplots_adjust(bottom=0.15)
plt.savefig(os.path.join(output_dir, 'nu.png'))
