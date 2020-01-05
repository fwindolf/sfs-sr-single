import os
import re
import json
import numpy as np
import sqlite3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='/data/eval/sfs/synthetic/')
parser.add_argument('-t', '--table', default='runs_synthetic')

args = parser.parse_args()

output_dir = args.dataset

run_re = '(\d+x\d+)_(.*?)_(.*?)__(.*?)_(.*?)_(.*?)__(.*?)_(\d+)'
final_re = '\|\s+(\d+) Iterations in (.*?) s'
iter_re = '\|\s+\d+\s+\|.*?\|\s+(\d+) ms \|\s+(\d+) ms \|\s+(\d+) ms \|\s+(\d+) ms \|\s+(\d+) ms \|'


def get_val(dictionary, key_list):
    '''
    Get the value for the leaf entry in key_list
    '''
    tmp = dictionary
    for k in key_list:
        if k in tmp:
            tmp = tmp[k]
        else:
            print 'Didnt find {}!'.format(k)
            return None
    return tmp

def set_val(dictionary, key_list, val):
    '''
    Set the value of the leaf of key_list
    '''
    tmp = dictionary
    for k in key_list[:-1]:
        if k in tmp:
            tmp = tmp[k]
        else:
            tmp[k] = dict()
            tmp = tmp[k]

    k = key_list[-1]
    if k in tmp and type(tmp[k]) is not type(val):
        raise AttributeError('Overwriting with different type!')
    else:
        tmp[k] = val

def merge(dict1, dict2):
    output = dict()
    keys = dict1.keys() + dict2.keys()
    for k in keys:
        if k in dict1 and k in dict2:
            output[k] = merge(dict1[k], dict2[k])
        elif k in dict1:
            output[k] = dict1[k]
        elif k in dict2:
            output[k] = dict2[k]
    
    return output

def iterate(data, leaf_fn, prev_keys=[]):
    def is_leaf(data):
        for v in data.values():
            if isinstance(v, dict):
                return False
        return True

    if is_leaf(data):
        #print 'Leaf:', prev_keys, data
        try:
            leaf_fn(prev_keys, data)
        except Exception as e:
            print ('Failed to execute leaf function:', e)
    else:
        #print 'Dict:', prev_keys, data
        for k, v in data.items():
            iterate(v, leaf_fn, prev_keys + [k, ])



# Put into a database for easier queries
conn = sqlite3.connect('eval.db')
c = conn.cursor()

c.execute('''
    DROP TABLE IF EXISTS {}
'''.format(args.table))


c.execute('''
    CREATE TABLE IF NOT EXISTS {} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        resolution TEXT,
        tau REAL,
        kappa REAL,
        gamma REAL,
        nu REAL,
        mu REAL,
        lambda INTEGER,
        iterations INTEGER,
        rms_error REAL,
        median_error REAL,
        mean_error REAL,
        time REAL,
        steps INTEGER
    )
'''.format(args.table))

# {res_x}x{res_y}_{tau}_{kappa}__{gamma}_{nu}_{mu}__{lambda}_{iters}
sql_insert_runs = '''
    INSERT INTO {} (resolution, tau, kappa, gamma, nu, mu, 
    lambda, iterations, rms_error, median_error, mean_error, time, steps)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
'''.format(args.table)

params = list()

# Gather all output.txt and results.txt data
runs = os.listdir(output_dir)
for run in runs:
    if os.path.isfile(os.path.join(output_dir, run)) or len(run) < 20:
        continue

    try:
        # Process output.txt
        with open(os.path.join(output_dir, run, 'output.txt')) as f_out:
            lines = [l.strip() for l in f_out.readlines() if l.startswith('|')]

            for l in lines:
                match_final = re.search(final_re, l)
                if match_final:
                    steps = int(match_final.group(1))
                    time = float(match_final.group(2))

        # Process results.txt
        with open(os.path.join(output_dir, run, 'results.txt')) as f_res:
            lines = f_res.readlines()
            assert(len(lines)) == 1
            err_mean, err_median, err_rms = lines[0].strip().split(';')
    except Exception as e:
        print 'Failed to open output/result in {}'.format(run)
        continue

    # Write to dictionary
    match_run = re.search(run_re, run)
    run_keys = [match_run.group(i) for i in range(1, 9)]
    params.append(run_keys + [err_rms, err_median, err_mean, time, steps])

print "Inserting {} elements to {}".format(len(params), args.table)
try:
    c.executemany(sql_insert_runs, params)
except Exception as e:
    print "Could not insert: ", e

conn.commit()
c.execute("SELECT COUNT(*) FROM {}".format(args.table))
num, = c.fetchone()
print "Done, {} now contains {} rows".format(args.table, num)

conn.close()





    