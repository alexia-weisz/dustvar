import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.stats import norm

from pdb import set_trace


data_loc = '/Users/alexialewis/research/PHAT/dustvar/'

group_labels = ['R_V', 'f_bump', 'sampler_chain', 'sampler_flatchain', 'sampler_lnprob']
infile = os.path.join(data_loc, 'data.h5')

fig = plt.figure()

xmin, xmax = 0, 11
rvgrid = np.linspace(xmin,xmax, 101)
x = np.linspace(xmin, xmax, 100)
ps = []
lps= []
with h5py.File(infile, 'r') as hf:
    nregs = len(hf.keys())

    n = (hf.get(hf.keys()[0]))['sampler_flatchain'].shape[0]

    for i in range(50):
        group = hf.get(hf.keys()[i])
        rv_pdf = np.asarray(group['sampler_flatchain'][:,0])
        mu, std = norm.fit(rv_pdf)
        p = norm.pdf(x, mu, std)
        ps.append(p)

        color = next(plt.gca()._get_lines.prop_cycler)
        c = color['color']
        n, bins, patches = plt.hist(rv_pdf, bins=rvgrid, lw=1,
                                    histtype='step', normed=True, color=c)
        #plt.plot(x, p, lw=1, color=c)
        nn = np.log(n)
        nn[nn == -np.inf] = 0.
        lps.append(nn)

    lps = np.asarray(lps)
    lps[lps == 0.] = -100.
    lp_pdf = np.sum(lps, axis=0)
    #lp_pdf /= 100.
    #lp_pdf += np.max(lp_pdf)*2

    plt.grid()
    plt.xlabel('$R_V$')
    plt.ylabel('$\ln f$')
    plt.xlim(0.2, 10.2)

    plt.figure()
    plt.hist(x, bins=len(x), weights=np.exp(lp_pdf), lw=2, histtype='step', color='k', normed=True)
    #plt.grid()

    #plt.savefig(data_loc + 'plots/reg_rv_pdfs.pdf')
    plt.show()


## add up the lnprobs to get the single curve to fit them all.
