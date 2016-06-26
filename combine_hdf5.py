import h5py
import os
import glob
from pdb import set_trace

data_loc = '/Users/alexialewis/research/PHAT/dustvar'
emcee_loc = os.path.join(data_loc, 'emcee_runs')
file_list = glob.glob(emcee_loc + '/*.h5')

single_file = os.path.join(data_loc, 'all_runs_newred.h5')

with h5py.File(single_file, 'w') as hf:
    for infile in file_list:
        with h5py.File(infile, 'r') as sf:
            group = sf.get(sf.keys()[0])

            g = hf.create_group(group.name)
            for k in group.keys():
                g.create_dataset(k, data=group[k])


