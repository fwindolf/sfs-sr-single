import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir', default='/data/eval/sfs/synthetic/graphs/')
parser.add_argument('-t', '--table', default='runs_synthetic')

args = parser.parse_args()

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

params = {
    "runs_synthetic" :  {
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
    },
    "runs_real" : {
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
                "k" : 1e-6,
                "t" : 2
            },
            "mu" : 0.01, 
            "nu" : 0.01,
        },
        "lambda" : 0.5,
        "iters" : 400,
        "gamma" : [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20]
    }
}

sql_gammas = """ 
    SELECT 
        gamma, mean_error, rms_error, time, steps 
    FROM 
        {} 
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
    """.format(args.table)


conn = sqlite3.connect('eval.db')
c = conn.cursor()

def get_params(table, res, run):
    return [
        params[table][res][run]['k'],
        params[table][res][run]['t'],
        params[table][res]['mu'],
        params[table][res]['nu'],
        params[table]['lambda'],
        params[table]['iters'],
        res
    ]

print get_params(args.table,  '1280x960', 'best')

# Data
## 1280x960
c.execute(sql_gammas, get_params(args.table, '1280x960', 'best'))
res_1280_best =  c.fetchall()

c.execute(sql_gammas, get_params(args.table, '1280x960', 'fast'))
res_1280_fast =  c.fetchall()

## 640x480
c.execute(sql_gammas, get_params(args.table, '640x480', 'best'))
res_640_best =  c.fetchall()

c.execute(sql_gammas, get_params(args.table, '640x480', 'fast'))
res_640_fast =  c.fetchall()

xs = np.unique([r[0] for r in res_1280_best + res_1280_fast + res_640_best + res_640_fast])
xmin = np.argmin(xs)
xmax = np.argmax(xs)

# Create figure
sns.set_style('ticks')
fig, axes = plt.subplots(1, 2, sharex=True)
fig.suptitle(r'Influence of $\gamma$ parameter on quality and runtime')
fig.set_figheight(4)
fig.set_figwidth(15)

# Quality
axes[0].set_ylabel('Mean Error (mm)')
axes[0].set_xlabel(r'$\gamma$')
axes[0].set_xscale('log')
axes[0].set_xlim([xmin, xmax])
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

axes[0].legend(loc='center right', fontsize='small')

# Runtime
axes[1].set_ylabel('Runtime (s)')
axes[1].set_xlabel(r'$\gamma$')
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
f_output = os.path.join(output_dir, 'gamma.png')
print "Writing graph to", f_output
plt.savefig(f_output)
