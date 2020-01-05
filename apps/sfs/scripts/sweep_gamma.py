import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir', default='/data/eval/sfs/synthetic/graphs/')
parser.add_argument('-t', '--table', default='runs_synthetic')
parser.add_argument('-r', '--resolution', default='640x480')
parser.add_argument('-m', '--mu', required=True)
parser.add_argument('-n', '--nu', required=True)
parser.add_argument('-k', '--kappa', required=True)

args = parser.parse_args()

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Sanity checks
conn = sqlite3.connect('eval.db')
c = conn.cursor()

c.execute("SELECT DISTINCT lambda FROM {}".format(args.table))
assert(len(c.fetchall()) == 1)

c.execute("SELECT DISTINCT iterations FROM {}".format(args.table))
assert(len(c.fetchall()) == 1)

c.execute("SELECT DISTINCT tau FROM {} WHERE kappa=?".format(args.table), [args.kappa])
assert(len(c.fetchall()) == 1)

sql_gammas = """ 
    SELECT 
        gamma, mean_error, rms_error, time, steps 
    FROM 
        {} 
    WHERE
        kappa = ? AND
        mu = ? AND
        nu = ? AND
        resolution LIKE ?
    ORDER BY 
        gamma ASC;
    """.format(args.table)

# Data
c.execute(sql_gammas, [args.kappa, args.mu, args.nu, args.resolution])
res =  c.fetchall()

xs = np.unique([r[0] for r in res])
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
#axes[0].set_xscale('log')
#axes[0].set_xlim([xmin, xmax])
axes[0].plot(
    [r[0] for r in res], 
    [r[1] for r in res],
    c='red', label='gamma')

# Runtime
axes[1].set_ylabel('Runtime (s)')
axes[1].set_xlabel(r'$\gamma$')
axes[1].plot(
    [r[0] for r in res], 
    [r[3] for r in res],
    c='red', label='gamma')
#axes[1].set_xlim([xmin, xmax])

plt.subplots_adjust(bottom=0.15)
f_output = os.path.join(output_dir, 'gamma.png')
print "Writing graph to", f_output
plt.savefig(f_output)
