import os

data_dir = '/data/eval/sfs/'

levels = [1, 2, 3, 4]
datasets = ['real', 'synthetic']
configs = ['best', 'fast']

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

resolutions = ['640x480', '320x240', '160x120']

for dataset in datasets:
    print dataset
    for level in levels:

        res = dict()
        for config in configs:
            if level == 1:
                res_dir = os.path.join(data_dir, 'pyramid_{}'.format(level), '{}_{}'.format(dataset, config))
                # Simply take output
                output = read_results(res_dir, 'output')
                res[config] = {
                    'mae' : output['meanErr'],
                    'rmse' : output['rmsErr'],
                    't' : output['time']
                }    
            else:
                # Combine times
                time = 0
                for l in reversed(range(level - 1)):
                    res_dir = os.path.join(data_dir, 'pyramid_{}'.format(level), '{}_{}'.format(dataset, config), resolutions[l])
                    output = read_results(res_dir, 'output')

                    mae = output['meanErr']
                    rmse = output['rmsErr']
                    time += output['time']

                res[config] = {
                    'mae' : mae,
                    'rmse' : rmse,
                    't' : time
                }            

        print "{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
            level,
            res['best']['t'], res['best']['mae'], res['best']['rmse'],
            res['fast']['t'], res['fast']['mae'], res['fast']['rmse'],
        )