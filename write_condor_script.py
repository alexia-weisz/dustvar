import os
import numpy as np


data_loc = '/Users/alexialewis/research/PHAT/dustvar/'
run_loc = '/astro/store/phat/arlewis/dustvar/'
condorfile = os.path.join(data_loc, 'model_rv_fbump_newred_sfh_all_condor.cfg')


f = open(condorfile, 'w')
f.write('Notification = never \n')
f.write('getenv       = true \n')
f.write('universe     = vanilla \n')
f.write('Requirements = (Machine != "ullr.astro.washington.edu" && Machine != "acidrain.astro.washington.edu") \n')
f.write('Executable   = /astro/users/arlewis/research/model_rv_condor.sh \n')
f.write(' \n')
f.write('Initialdir = ' + run_loc + ' \n')
f.write(' \n')


finite_reg_num = 10#10124
count = 0
for j in range(0, finite_reg_num):
    region = str(j+1).zfill(5)

    out = run_loc + 'newred_sfh_data_region_' + region + '.h5'
    scrnfile = run_loc + 'newred_sfh_screen_' + region + '.out'
    run = str(j)

    f.write('Arguments = ' + run + ' \n')
    #f.write('Output = ' + out + '\n')
    f.write('Log = ' + scrnfile + '\n')
    f.write('Queue \n')
    f.write(' \n')
    count += 1

f.close()
print count
